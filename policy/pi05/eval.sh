#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

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
sync_action_chunk_steps=${10:-10}
phase_prompt_conditioning=${11:-1}
boundary_context_steps=${12:-20}
remote_policy_host=${13:-127.0.0.1}
remote_policy_port=${14:-0}
runtime_ld_path=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# source .venv/bin/activate
cd ../.. # move to root

LD_LIBRARY_PATH=${runtime_ld_path} PYTHONWARNINGS=ignore::UserWarning \
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
    --sync_action_chunk_steps ${sync_action_chunk_steps} \
    --phase_prompt_conditioning ${phase_prompt_conditioning} \
    --boundary_context_steps ${boundary_context_steps} \
    --remote_policy_host ${remote_policy_host} \
    --remote_policy_port ${remote_policy_port} \
    --policy_name ${policy_name} 
