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

import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from pruning.module_dict import *


def pruning_schedule(
    pruned_times, prune_shots,
    current_layer, target_layer,
    current_mha_dim, target_mha_dim,
    current_head_num,
    current_mlp_dim, target_mlp_dim,
    mha_scores, mlp_scores,
    prune_head_num: bool, prune_head_dim: bool,
):
    remaining_shots = prune_shots - pruned_times
    next_layer = current_layer-(current_layer-target_layer)//remaining_shots
    next_mha_dim = current_mha_dim - \
        (current_mha_dim-target_mha_dim)//remaining_shots
    next_mlp_dim = current_mlp_dim - \
        (current_mlp_dim-target_mlp_dim)//remaining_shots
    print(
        f"next_layer: {next_layer}, next_mha_dim: {next_mha_dim}, next_mlp_dim: {next_mlp_dim}")

    # Getting scores
    # print("  Calculating MHA and MLP...")
    all_layer_mha_keep_indices, all_layer_mlp_keep_indices = [], []
    for layer_id in range(next_layer):
        # 1d score tensor(inter_size)
        mha_score = mha_scores[layer_id]
        # 1d score tensor(inter_size)
        mlp_score = mlp_scores[layer_id]

        # Init
        remain_score = mha_score.view(current_head_num, -1)
        remain_head_num, remain_head_dim = remain_score.shape
        next_head_num, next_head_dim = remain_head_num, remain_head_dim

        # check head first, if we can remove some entire heads
        # then check hidden dims in each head
        topk_head_indices = torch.arange(0, next_head_num)
        if prune_head_num:
            next_head_num = math.ceil(next_mha_dim / next_head_dim)

            ALPHA = 10.0/next_head_dim
            inter_head_score = torch.mean(remain_score, dim=1) + \
                ALPHA * torch.max(remain_score, dim=1).values
            _, topk_head_indices = torch.topk(
                inter_head_score.cuda(),
                next_head_num, sorted=False
            )  # non-ascending value
            topk_head_indices, _ = torch.sort(
                topk_head_indices)  # ascending indices
            topk_head_indices = topk_head_indices.cpu()
            remain_score = remain_score[topk_head_indices]

        head_topk_dim_indices_list = [torch.arange(
            0, next_head_dim) for _ in range(next_head_num)]
        # currently prune entire heads and then prune dims
        if prune_head_dim:
            next_head_dim = next_mha_dim // next_head_num
            # HACK: make sure next_head_dim is even
            next_head_dim = next_head_dim if next_head_dim % 2 == 0 else next_head_dim + 1

            for i, intra_head_score in enumerate(remain_score):
                _, head_topk_indices = torch.topk(
                    intra_head_score.cuda(),
                    next_head_dim, sorted=False
                )  # non-ascending value
                head_topk_indices, _ = torch.sort(
                    head_topk_indices)  # ascending indices
                head_topk_indices = head_topk_indices.cpu()
                head_topk_dim_indices_list[i] = head_topk_indices

        mha_topk_indices = []
        for i, head_idx in enumerate(topk_head_indices):
            head_topk_indices = head_topk_dim_indices_list[i] + \
                head_idx*remain_head_dim
            # HACK: offset is remain_head_dim not next_head_dim
            mha_topk_indices += head_topk_indices.tolist()
        # print("MHA: Keep heads of layer %d: (%d * %d)" %
        #             (layer_id, next_head_num, next_head_dim))
        all_layer_mha_keep_indices.append(mha_topk_indices)

        _, mlp_topk_indices = torch.topk(
            mlp_score.cuda(), next_mlp_dim, sorted=False)  # non-ascending value
        mlp_topk_indices, _ = torch.sort(mlp_topk_indices)  # ascending indices
        mlp_topk_indices = mlp_topk_indices.cpu()
        mlp_topk_indices = mlp_topk_indices.tolist()
        # print("MLP: Keep inter dim of layer %d: %s" %
        #             (layer_id, str(next_mlp_dim)))
        all_layer_mlp_keep_indices.append(mlp_topk_indices)

    next_mha_dim = next_head_dim * next_head_num

    pruning_config = {
        "next_layer": next_layer,
        "next_head_num": next_head_num,
        "next_head_dim": next_head_dim,
        "next_mha_dim": next_mha_dim,
        "next_mlp_dim": next_mlp_dim
    }
    return pruning_config, all_layer_mha_keep_indices, all_layer_mlp_keep_indices


def pruning_structured(
    model: Any,
    model_type: str,
    next_layer: int,
    next_head_num: int,
    next_head_dim: int,
    next_mha_dim: int,
    next_mlp_dim: int,
    all_layer_mha_keep_indices,
    all_layer_mlp_keep_indices,
) -> Tuple[Any, int, int]:
    for layer_id in range(next_layer):
        # Set pruning dimension (input or output) of modules
        prune_out = [
            f"{layer_str_dict[model_type]}.{layer_id}.{prune_out_module}"
            for prune_out_module in prune_out_str_dict[model_type]
        ]
        prune_in = [
            f"{layer_str_dict[model_type]}.{layer_id}.{prune_in_module}"
            for prune_in_module in prune_in_str_dict[model_type]
        ]

        layer = layer_dict[model_type](model)[layer_id]
        for name, module in layer.named_modules():
            if mha_str_dict[model_type] == name:  # Attention layer Module
                module.mha_inter_size = next_mha_dim
                module.num_attention_head = next_head_num
                module.num_key_value_heads = next_head_num
                module.head_dim = next_head_dim
                continue
            if mlp_str_dict[model_type] == name:  # MLP layer Module
                module.intermediate_size = next_mlp_dim
                continue
            if not hasattr(module, 'weight'):
                continue

            if mha_str_dict[model_type] in name:
                topk_indices = all_layer_mha_keep_indices[layer_id]
                next_dim = next_mha_dim
            elif mlp_str_dict[model_type] in name:
                topk_indices = all_layer_mlp_keep_indices[layer_id]
                next_dim = next_mlp_dim
            else:
                continue

            full_name = f"{layer_str_dict[model_type]}.{layer_id}.{name}"
            if full_name in prune_out:
                chunks = pack_weights_chunks_dict[model_type].get(name, 1)
                # default 1 chunk
                weight_chunks = torch.chunk(module.weight.data, chunks, dim=0)
                if chunks > 1:
                    print(
                        f"  Dividied {full_name} into {chunks} weight chunks.")
                new_weight = weight_chunks[0][topk_indices, :]
                for i in range(1, chunks):
                    new_weight = torch.cat(
                        (new_weight, weight_chunks[i][topk_indices, :]), dim=0)
                module.weight = nn.Parameter(new_weight)
                # HACK: Linear(bias=False), but hasattr(module, 'bias') is True
                if hasattr(module.bias, 'data'):
                    bias_chunks = torch.chunk(module.bias.data, chunks, dim=0)
                    new_bias = bias_chunks[0][topk_indices]
                    for i in range(1, chunks):
                        new_bias = torch.cat(
                            (new_bias, bias_chunks[i][topk_indices]), dim=0)
                    module.bias = nn.Parameter(new_bias)
                module.out_features = next_dim * chunks
            elif full_name in prune_in:
                module.weight = nn.Parameter(
                    module.weight.data[:, topk_indices])  # remove cols
                module.in_features = next_dim
            # print("  Pruned %s to %s" % (name, str(next_dim)))

    return model
