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

export HF_DATASETS_CACHE="./hf-cache/datasets"

set -e

help() {
    echo "Usage:"
    echo "  -a A: [A] MHA inter dim. Will override sparsity"
    echo "  -d D: [D] dataset_owner/dataset_name"
    echo "  -f F: [F] MLP inter dim. Will override sparsity"
    echo "  -k K: [K] steps for pruning"
    echo "  -h: display help"
    echo "  -l L: [L] training length"
    echo "  -n N: [N] eval samples"
    echo "  -p P: select mode from [acts|taylor]"
    echo "  -t T: [T]B tokens to use in total"
    echo "  -x X -y Y -z Z: [X]/[Y]/[Z] model_type/model/model_size"
}

eval_model() {
    model_dir=$1
    python -m lm-evaluation-harness.main \
        --model hf-causal-prune \
        --model_args pretrained=${model_dir},model_type=${model_type} \
        --tasks wikitext,lambada_standard,arc_easy,boolq,logiqa,openbookqa,piqa,truthfulqa_mc \
        --device cuda

    python -m lm-evaluation-harness.main \
        --model hf-causal-prune \
        --model_args pretrained=${model_dir},model_type=${model_type} \
        --tasks arc_challenge \
        --num_fewshot 25 \
        --device cuda

    python -m lm-evaluation-harness.main \
        --model hf-causal-prune \
        --model_args pretrained=${model_dir},model_type=${model_type} \
        --tasks hellaswag \
        --num_fewshot 10 \
        --device cuda

    python -m lm-evaluation-harness.main \
        --model hf-causal-prune \
        --model_args pretrained=${model_dir},model_type=${model_type} \
        --tasks winogrande \
        --num_fewshot 5 \
        --device cuda
}

last(){
    res_dir="${output_dir_base}"
    echo "========== ${res_dir} =========="
    eval_model ${res_dir}
}

full() {
    res_dir="${output_dir_base}"
    for ckpt_dir in $(find ${res_dir} -type d -name "checkpoint-*"); do
        echo "========== ${ckpt_dir} =========="
        eval_model ${ckpt_dir}
    done
    last
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    help
    exit 0
fi

# default
prune_mode=acts
prune_shots=8
eval_samples=128
target_mha_dim=2048
target_mlp_dim=3072
train_seqlen=4096
ft_tokens_B=50
model_type="llama"
model="llama2"
model_size="7B"

while getopts a:d:f:k:l:n:p:t:x:y:z: flag; do
    case "${flag}" in
    a) target_mha_dim=${OPTARG} ;;
    d) dataset_name=${OPTARG} ;;
    f) target_mlp_dim=${OPTARG} ;;
    k) prune_shots=${OPTARG} ;;
    l) train_seqlen=${OPTARG} ;;
    n) eval_samples=${OPTARG} ;;
    p) prune_mode=${OPTARG} ;;
    t) ft_tokens_B=${OPTARG} ;;
    x) model_type=${OPTARG} ;;
    y) model=${OPTARG} ;;
    z) model_size=${OPTARG} ;;
    esac
done

dataset_root="./data"
dataset_name="togethercomputer/RedPajama-Data-1T"
dataset_name_or_path="${dataset_root}/${dataset_name}"

output_dir_base="./outputs/${model}-${model_size}_${dataset_name}/${train_seqlen}len_${target_mha_dim}a_${target_mlp_dim}f_${prune_shots}shots_${eval_samples}samples-${prune_mode}"

# full
last
