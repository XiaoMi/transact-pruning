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

### model ###
model_dict = {
    "llama": lambda model: model.model,
    "baichuan": lambda model: model.model,
}
model_str_dict = {
    "llama": "model.model",
    "baichuan": "model.model",
}
### model ###

### layernorm ###
ln_str_dict = {
    "llama": ["input_layernorm", "post_attention_layernorm", "norm"],
    "baichuan": ["input_layernorm", "post_attention_layernorm", "norm"],
}
### layernorm ###

### lm_head ###
clf_dict = {
    "llama": lambda model: model.lm_head,
    "baichuan": lambda model: model.lm_head,
}
clf_str_dict = {
    "llama": "model.lm_head",
    "baichuan": "model.lm_head",
}
### lm_head ###

### embedding ###
embed_dict = {
    "llama": lambda model: model.model.embed_tokens,
    "baichuan": lambda model: model.model.embed_tokens,
}
embed_str_dict = {
    "llama": "model.model.embed_tokens",
    "baichuan": "model.model.embed_tokens",
}
### embedding ###

### model layer ###
layer_dict = {
    "llama": lambda model: model.model.layers,
    "baichuan": lambda model: model.model.layers,
}
layer_str_dict = {
    "llama": "model.model.layers",
    "baichuan": "model.model.layers",
}
### model layer ###

### attn layer ###
mha_dict = {
    "llama": lambda layer: layer.self_attn,
    "baichuan": lambda layer: layer.self_attn,
}
mha_str_dict = {
    "llama": "self_attn",
    "baichuan": "self_attn",
}

### attn projection ###
mha_proj_dict = {
    "llama": {"q": lambda layer: mha_dict["llama"](layer).q_proj,
              "k": lambda layer: mha_dict["llama"](layer).k_proj,
              "v": lambda layer: mha_dict["llama"](layer).v_proj,
              "o": lambda layer: mha_dict["llama"](layer).o_proj},
    "baichuan": {"q": lambda layer: mha_dict["baichuan"](layer).W_pack,
                 "k": "same_as_q",
                 "v": "same_as_q",
                 "o": lambda layer: mha_dict["baichuan"](layer).o_proj},
}
mha_proj_str_dict = {
    "llama": {"q": "q_proj",
              "k": "k_proj",
              "v": "v_proj",
              "o": "o_proj"},
    "baichuan": {"q": "W_pack",
                 "k": "same_as_q",
                 "v": "same_as_q",
                 "o": "o_proj"},
}
### attn projection ###

### mlp layer ###
mlp_dict = {
    "llama": lambda layer: layer.mlp,
    "baichuan": lambda layer: layer.mlp,
}
mlp_str_dict = {
    "llama": "mlp",
    "baichuan": "mlp",
}
### mlp layer ###

### mlp projection ###
mlp_proj_dict = {
    "llama": {"gate": lambda layer: mlp_dict["llama"](layer).gate_proj,
              "up": lambda layer: mlp_dict["llama"](layer).up_proj,
              "down": lambda layer: mlp_dict["llama"](layer).down_proj},
    "baichuan": {"gate": lambda layer: mlp_dict["baichuan"](layer).gate_proj,
                 "up": lambda layer: mlp_dict["baichuan"](layer).up_proj,
                 "down": lambda layer: mlp_dict["baichuan"](layer).down_proj},
}
mlp_proj_str_dict = {
    "llama": {"gate": "gate_proj",
              "up": "up_proj",
              "down": "down_proj"},
    "baichuan": {"gate": "gate_proj",
                 "up": "up_proj",
                 "down": "down_proj"},
}
### mlp projection ###

### projection input to be pruned ###
prune_in_str_dict = {
    "llama": ["self_attn.o_proj", "mlp.down_proj"],
    "baichuan": ["self_attn.o_proj", "mlp.down_proj"],
}
### projection input to be pruned ###

### projection output to be pruned ###
prune_out_str_dict = {
    "llama": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"],
    "baichuan": ["self_attn.W_pack", "mlp.gate_proj", "mlp.up_proj"],
}
### projection output to be pruned ###

### packed weights ###
pack_weights_chunks_dict = {
    "llama": {

    },
    "baichuan": {
        "self_attn.W_pack": 3,
    },
}
### packed weights ###
