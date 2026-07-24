#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G
export PYTHONNOUSERSITE=1

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
async_scene_context_steps=${7:-0}
checkpoint_id=${8:-30000}
test_num=${9:-100}
policy_server_host=${10:-127.0.0.1}
policy_server_port=${11:-8000}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

policy_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${policy_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${policy_dir}/../.."

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --async_scene_context_steps ${async_scene_context_steps} \
    --checkpoint_id ${checkpoint_id} \
    --test_num ${test_num} \
    --policy_server_host ${policy_server_host} \
    --policy_server_port ${policy_server_port} \
    --policy_name ${policy_name} 
