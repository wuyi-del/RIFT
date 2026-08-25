#!/usr/bin/env bash
set -euo pipefail

NGPU=${NGPU:-8}
GRAD=${GRAD:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-5e-6}
PORT=${PORT:-12964}
DATA=${DATA:-data/openthoughts_math_30k}
MAX_LENGTH=${MAX_LENGTH:-1024}
MAX_STEPS=${MAX_STEPS:-50}
SEED=${SEED:-42}
RUN_TAG=${RUN_TAG:-formal50_len1024_seed42}

accelerate launch \
  --config_file accelerate.yaml \
  --num_processes "$NGPU" \
  --gradient_accumulation_steps "$GRAD" \
  --main_process_port "$PORT" \
  opsd_train.py \
  --model_name_or_path /models/Qwen3-4B \
  --dataset_path "$DATA" \
  --task_type math \
  --learning_rate "$LR" \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_checkpointing \
  --gradient_accumulation_steps "$GRAD" \
  --output_dir results/matched_opsd \
  --run_config "matched_opsd_qwen3_4b_${RUN_TAG}" \
  --max_steps "$MAX_STEPS" \
  --max_completion_length "$MAX_LENGTH" \
  --save_steps 25 \
  --logging_steps 1 \
  --attn_implementation sdpa \
  --torch_dtype bfloat16 \
  --max_length 20000 \
  --beta 0 \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.35 \
  --vllm_tensor_parallel_size 1 \
  --use_peft \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature 1.1 \
  --top_p 0.95 \
  --top_k 20 \
  --lmbda 1 \
  --fixed_teacher \
  --jsd_token_clip 0.05 \
  --wandb_project OPSD-RIFT \
  --kd_type MATCHED-OPSD \
  --seed "$SEED"
