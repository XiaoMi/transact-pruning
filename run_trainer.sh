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
    echo "  -b B: [B] samples in a micro batch size"
    echo "  -d D: [D] dataset_owner/dataset_name"
    echo "  -f F: [F] MLP inter dim. Will override sparsity"
    echo "  -g G: [G] samples in a global batch"
    echo "  -k K: [K] steps for pruning"
    echo "  -h: display help"
    echo "  -l L: [L] training length"
    echo "  -m M: select mode from [prune|ft|all]"
    echo "  -n N: [N] eval samples"
    echo "  -p P: select mode from [acts|taylor]"
    echo "  -s S: [S]% sparsity"
    echo "  -t T: [T]B tokens to use in total"
    echo "  -x X -y Y -z Z: [X]/[Y]/[Z] model_type/model/model_size"
}

prune() {
    micro_bsz=1
    global_bsz=64
    grad_accu=$(($global_bsz / $micro_bsz / $n_gpu))
    max_steps=$(($1 / (${train_seqlen} * ${micro_bsz} * ${n_gpu} * ${grad_accu})))
    accelerate launch --config_file train_config.yaml \
        --main_process_port 28571 \
        -m training.run_clm_prune \
        --ddp_timeout 3600 \
        --model_name_or_path ${model_name_or_path} \
        --model_type ${model_type} \
        --dataset_name ${dataset_name_or_path} \
        --block_size ${train_seqlen} \
        --dataloader_num_workers 2 \
        --do_train \
        --do_eval \
        --do_prune \
        --max_steps ${max_steps} \
        --max_eval_samples ${eval_samples} \
        --bf16 True \
        --optim adamw_torch_fused \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --learning_rate ${current_lr} \
        --warmup_ratio 0.1 \
        --per_device_train_batch_size ${micro_bsz} \
        --per_device_eval_batch_size ${micro_bsz} \
        --gradient_accumulation_steps ${grad_accu} \
        --prune_mode ${prune_mode} \
        --prune_head_num True \
        --prune_head_dim False \
        --target_mha_dim ${target_mha_dim} \
        --target_mlp_dim ${target_mlp_dim} \
        --prune_shots ${prune_shots} \
        --current_shot ${current_shot} \
        --save_steps 0 \
        --overwrite_output_dir \
        --output_dir ${output_dir}
}

finetune() {
    output_dir="${output_dir_base}/ft_${ft_tokens_B}Btokens"
    save_steps=0.1
    micro_bsz=$ft_micro_batch
    global_bsz=$ft_global_bsz
    grad_accu=$(($global_bsz / $micro_bsz / $n_gpu))
    max_steps=$(($1 / (${train_seqlen} * ${micro_bsz} * ${n_gpu} * ${grad_accu})))
    accelerate launch --config_file train_config.yaml \
        --main_process_port 28571 \
        -m training.run_clm_prune \
        --ddp_timeout 3600 \
        --model_name_or_path ${model_name_or_path} \
        --model_type ${model_type} \
        --dataset_name ${dataset_name_or_path} \
        --block_size ${train_seqlen} \
        --dataloader_num_workers 2 \
        --do_train \
        --max_steps ${max_steps} \
        --bf16 True \
        --optim adamw_torch_fused \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --learning_rate ${current_lr} \
        --warmup_ratio 0.1 \
        --per_device_train_batch_size ${micro_bsz} \
        --gradient_accumulation_steps ${grad_accu} \
        --save_steps ${save_steps} \
        --output_dir ${output_dir}
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    help
    exit 0
fi

# default
target_mha_dim=2048
ft_micro_batch=2
dataset_name=togethercomputer/RedPajama-Data-1T
target_mlp_dim=3072
ft_global_bsz=64
train_seqlen=4096
mode=all
eval_samples=128
prune_mode=acts
prune_shots=8
sparsity=62
ft_tokens_B=50
model_type=llama
model=llama2
model_size=7B

n_gpu=8
lr=5 # e-5
initial_mha_dim=4096
initial_mlp_dim=11008

while getopts a:b:d:f:g:k:l:m:n:p:s:t:x:y:z: flag; do
    case "${flag}" in
    a) target_mha_dim=${OPTARG} ;;
    b) ft_micro_batch=${OPTARG} ;;
    d) dataset_name=${OPTARG} ;;
    f) target_mlp_dim=${OPTARG} ;;
    g) ft_global_bsz=${OPTARG} ;;
    k) prune_shots=${OPTARG} ;;
    l) train_seqlen=${OPTARG} ;;
    m) mode=${OPTARG} ;;
    n) eval_samples=${OPTARG} ;;
    p) prune_mode=${OPTARG} ;;
    s) sparsity=${OPTARG} ;;  # sparsity%
    t) ft_tokens_B=${OPTARG} ;;
    x) model_type=${OPTARG} ;;
    y) model=${OPTARG} ;;
    z) model_size=${OPTARG} ;;
    esac
done

dataset_root="./data"
dataset_name_or_path="${dataset_root}/${dataset_name}"

if [ "${prune_shots}" = 1 ]; then
    prune_tokens=0 # 1-shot pruning without any training
else
    prune_tokens=$((1 * 1000 * 1000 * 1000 / ${prune_shots})) # tokens in a pruning step
fi

ft_tokens=$((${ft_tokens_B} * 1000 * 1000 * 1000)) # tokens in finetuning

if [ -z "$target_mha_dim" ]; then
    target_mha_dim=$((${initial_mha_dim} * (100 - ${sparsity}) / 100))
fi
if [ -z "$target_mlp_dim" ]; then
    target_mlp_dim=$((${initial_mlp_dim} * (100 - ${sparsity}) / 100))
fi

model_name_or_path="./models/${model}/${model_size}"
output_dir_base="./outputs/${model}-${model_size}_${dataset_name}/${train_seqlen}len_${target_mha_dim}a_${target_mlp_dim}f_${prune_shots}shots_${eval_samples}samples-${prune_mode}"

echo "=============================="
echo "Mode: ${mode}"
echo "Model: ${model_name_or_path}"
echo "N_GPU: ${n_gpu}"
echo "Finetune global-batch: ${ft_global_bsz}"
echo "Finetune micro-batch: ${ft_micro_batch}"
echo "=============================="
echo "Length: ${train_seqlen}"
echo "Eval samples: ${eval_samples}"
echo "Prune Mode: ${prune_mode}"
echo "Prune Dataset: ${dataset_name_or_path}"
echo "Prune tokens: 1B"
echo "Finetune Dataset: ${dataset_name_or_path}"
echo "Finetune tokens: ${ft_tokens}"
echo "=============================="
echo "Sparsity: ${sparsity}%"
echo "Target MHA: ${target_mha_dim}"
echo "Target MLP: ${target_mlp_dim}"
echo "=============================="

if [ "${prune_shots}" = "2" ]; then
    output_dir="${output_dir_base}/step1"
    current_shot=1
    current_lr=$(echo "scale=7; ${lr} * 2 * 1 / ${prune_shots} / 100000" | bc)
    prune ${prune_tokens}
    model_name_or_path=$output_dir

    output_dir="${output_dir_base}/step2"
    current_shot=1
    current_lr=0
    prune 0
    model_name_or_path=$output_dir
else
    for ((i = 1; i <= ${prune_shots}; i++)); do
        output_dir="${output_dir_base}/step${i}"
        current_shot=$i
        current_lr=$(echo "scale=7; ${lr} * 2 * ${i} / ${prune_shots} / 100000" | bc)
        if [ "$mode" = "prune" ] || [ "$mode" = "all" ]; then
            prune $((${prune_tokens} / 2 + ${prune_tokens} * ($i - 1) / (${prune_shots} - 1)))
        fi
        model_name_or_path=$output_dir
    done
fi

model_name_or_path="${output_dir_base}/step${prune_shots}"
if [ "$mode" = "ft" ] || [ "$mode" = "all" ]; then
    current_lr=$(echo "scale=7; ${lr} / 100000" | bc)
    finetune ${ft_tokens}
fi
