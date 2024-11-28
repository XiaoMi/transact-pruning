# Copyright (C) 2024 Xiaomi Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, 
# software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and limitations under the License.
#

import gc
from abc import abstractmethod

import torch

from pruning.module_dict import *


class Scorer():
    def __init__(self, model_type, num_layers):
        self.model_type = model_type
        self.num_layers = num_layers
        self.mha_scores = None  # torch.tensor(layer, inter_size)
        self.mlp_scores = None  # torch.tensor(layer, inter_size)

    @abstractmethod
    def accumulate(self):
        pass

    @abstractmethod
    def compute_score(self):
        pass

    def get_score(self):
        assert self.mha_scores.shape is not None and self.mlp_scores.shape is not None
        return self.mha_scores, self.mlp_scores

    def free_memory(self):
        gc.collect()
        torch.cuda.empty_cache()


class ActsScorer(Scorer):
    def __init__(self, model_type, num_layers):
        super().__init__(model_type, num_layers)
        self.mha_score_list = [0.0 for _ in range(num_layers)]
        self.mlp_score_list = [0.0 for _ in range(num_layers)]
        # HACK: use 0.0 fot broadcast in tensor op

    def accumulate(self, mha_pruning_acts, mlp_pruning_acts):
        # n_layer * (bsz, sl, inter_size)
        for i in range(self.num_layers):
            # norm sl (pow2, sum, root2), then pow2 and sum bsz
            # root2 later in compute_score to finish norm bsz (pow2, sum, root2)
            # fused to sum(sum(pow2)) -> n_layer * (inter_size)
            self.mha_score_list[i] += \
                (mha_pruning_acts[i].to(torch.float32).pow(2).sum(-2).sum(0))
            self.mlp_score_list[i] += \
                (mlp_pruning_acts[i].to(torch.float32).pow(2).sum(-2).sum(0))
            

    def compute_score(self):
        # stack -> (n_layer, inter_size)
        self.mha_scores = torch.stack(self.mha_score_list, dim=0)
        self.mlp_scores = torch.stack(self.mlp_score_list, dim=0)
        # root2
        self.mha_scores = (self.mha_scores.to(torch.float32).sqrt())
        self.mlp_scores = (self.mlp_scores.to(torch.float32).sqrt())

    def free_memory(self):
        self.mha_score_list = None
        self.mlp_score_list = None
        super().free_memory()


class TaylorScorer(Scorer):
    def __init__(self, model, model_type, num_layers):
        super().__init__(model_type, num_layers)
        self.init_memory(model)

    def init_memory(self, model):
        self.mha_proj_grad = [{} for _ in range(self.num_layers)]
        self.mlp_proj_grad = [{} for _ in range(self.num_layers)]

        for layer_id in range(self.num_layers):
            layer = layer_dict[self.model_type](model)[layer_id]

            self.mha_proj_grad[layer_id]["q"] = torch.zeros_like(
                mha_proj_dict[self.model_type]["q"](layer).weight.data)
            if mha_proj_dict[self.model_type]["k"] != "same_as_q":
                self.mha_proj_grad[layer_id]["k"] = torch.zeros_like(
                    mha_proj_dict[self.model_type]["k"](layer).weight.data)
            if mha_proj_dict[self.model_type]["v"] != "same_as_q":
                self.mha_proj_grad[layer_id]["v"] = torch.zeros_like(
                    mha_proj_dict[self.model_type]["v"](layer).weight.data)
            self.mha_proj_grad[layer_id]["o"] = torch.zeros_like(
                mha_proj_dict[self.model_type]["o"](layer).weight.data)

            if "gate" in mlp_proj_dict[self.model_type]:
                self.mlp_proj_grad[layer_id]["gate"] = torch.zeros_like(
                    mlp_proj_dict[self.model_type]["gate"](layer).weight.data)
            self.mlp_proj_grad[layer_id]["up"] = torch.zeros_like(
                mlp_proj_dict[self.model_type]["up"](layer).weight.data)
            self.mlp_proj_grad[layer_id]["down"] = torch.zeros_like(
                mlp_proj_dict[self.model_type]["down"](layer).weight.data)

    def accumulate(self, model):
        for layer_id in range(self.num_layers):
            layer = layer_dict[self.model_type](model)[layer_id]

            self.mha_proj_grad[layer_id]["q"] += \
                mha_proj_dict[self.model_type]["q"](layer).weight.grad.data.abs()
            if mha_proj_dict[self.model_type]["k"] != "same_as_q":
                self.mha_proj_grad[layer_id]["k"] += \
                    mha_proj_dict[self.model_type]["k"](layer).weight.grad.data.abs()
            if mha_proj_dict[self.model_type]["v"] != "same_as_q":
                self.mha_proj_grad[layer_id]["v"] += \
                    mha_proj_dict[self.model_type]["v"](layer).weight.grad.data.abs()
            self.mha_proj_grad[layer_id]["o"] += \
                mha_proj_dict[self.model_type]["o"](layer).weight.grad.data.abs()

            if "gate" in mlp_proj_dict[self.model_type]:
                self.mlp_proj_grad[layer_id]["gate"] += \
                    mlp_proj_dict[self.model_type]["gate"](layer).weight.grad.data.abs()
            self.mlp_proj_grad[layer_id]["up"] += \
                mlp_proj_dict[self.model_type]["up"](layer).weight.grad.data.abs()
            self.mlp_proj_grad[layer_id]["down"] += \
                mlp_proj_dict[self.model_type]["down"](layer).weight.grad.data.abs()

    def compute_score(self, model):
        for layer_id in range(self.num_layers):
            layer = layer_dict[self.model_type](model)[layer_id]

            q_taylor = torch.abs(self.mha_proj_grad[layer_id]["q"]
                                 .mul(mha_proj_dict[self.model_type]["q"](layer).weight.data))
            if mha_proj_dict[self.model_type]["k"] == "same_as_q" and \
               mha_proj_dict[self.model_type]["v"] == "same_as_q":
                q_taylor, k_taylor, v_taylor = q_taylor.chunk(3, dim=0)
            else:
                if mha_proj_dict[self.model_type]["k"] == "same_as_q":
                    q_taylor, k_taylor = q_taylor.chunk(2, dim=0)
                else:
                    k_taylor = torch.abs(self.mha_proj_grad[layer_id]["k"]
                                         .mul(mha_proj_dict[self.model_type]["k"](layer).weight.data))
                v_taylor = torch.abs(self.mha_proj_grad[layer_id]["v"]
                                     .mul(mha_proj_dict[self.model_type]["v"](layer).weight.data))
            o_taylor = torch.abs(self.mha_proj_grad[layer_id]["o"]
                                 .mul(mha_proj_dict[self.model_type]["o"](layer).weight.data))

            if "gate" in mlp_proj_dict[self.model_type]:
                gate_taylor = torch.abs(self.mlp_proj_grad[layer_id]["gate"]
                                        .mul(mlp_proj_dict[self.model_type]["gate"](layer).weight.data))
            up_taylor = torch.abs(self.mlp_proj_grad[layer_id]["up"]
                                  .mul(mlp_proj_dict[self.model_type]["up"](layer).weight.data))
            down_taylor = torch.abs(self.mlp_proj_grad[layer_id]["down"]
                                    .mul(mlp_proj_dict[self.model_type]["down"](layer).weight.data))

            # sum(1): sum up in_feature, get out_feature values
            mha_taylor = q_taylor.sum(1) + k_taylor.sum(1) + v_taylor.sum(1) \
                + o_taylor.sum(0)
            mlp_taylor = up_taylor.sum(1)+down_taylor.sum(0)
            if "gate" in mlp_proj_dict[self.model_type]:
                mlp_taylor += gate_taylor.sum(1)

            mha_taylor = mha_taylor.unsqueeze(0).detach().cpu()
            mlp_taylor = mlp_taylor.unsqueeze(0).detach().cpu()
            if self.mha_scores is None:
                self.mha_scores = mha_taylor
            else:
                self.mha_scores = torch.cat((self.mha_scores, mha_taylor), dim=0)
            if self.mlp_scores is None:
                self.mlp_scores = mlp_taylor
            else:
                self.mlp_scores = torch.cat((self.mlp_scores, mlp_taylor), dim=0)

    def free_memory(self):
        self.mha_proj_grad = None
        self.mlp_proj_grad = None
        super().free_memory()
