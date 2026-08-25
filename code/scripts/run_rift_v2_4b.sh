#!/usr/bin/env bash
set -euo pipefail

NGPU=${NGPU:-8}
GRAD=${GRAD:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-5e-6}
PORT=${PORT:-12967}
DATA=${DATA:-data/openthoughts_math_30k}
MAX_LENGTH=${MAX_LENGTH:-1024}
MAX_STEPS=${MAX_STEPS:-50}
SEED=${SEED:-42}
SIGN_MARGIN=${SIGN_MARGIN:-0.05}
ENTROPY_QUANTILE=${ENTROPY_QUANTILE:-0.75}
ROUTE_WEIGHT=${ROUTE_WEIGHT:-1.0}
RECOVERY_WINDOW=${RECOVERY_WINDOW:-32}
RECOVERY_MARGIN=${RECOVERY_MARGIN:-0.005}
RECOVERY_QUANTILE=${RECOVERY_QUANTILE:--1}
EXACT_RANK=${EXACT_RANK:-0}
ROUTING_SCORE=${ROUTING_SCORE:-future_recovery}
REQUIRE_FULL_WINDOW=${REQUIRE_FULL_WINDOW:-0}
HARD_ENTROPY_QUANTILE=${HARD_ENTROPY_QUANTILE:--1}
HARD_RECOVERY_QUANTILE=${HARD_RECOVERY_QUANTILE:--1}
GROUPWISE_RECOVERY_QUANTILES=${GROUPWISE_RECOVERY_QUANTILES:-0}
FORK_ONSET_ROUTING=${FORK_ONSET_ROUTING:-0}
FORK_ONSET_GAP=${FORK_ONSET_GAP:-4}
REFLECTION_SAFE_WEIGHTING=${REFLECTION_SAFE_WEIGHTING:-0}
REFLECTION_PROTECTION_WEIGHT=${REFLECTION_PROTECTION_WEIGHT:-0.25}
ASYMMETRIC_SOFT_CLAMP=${ASYMMETRIC_SOFT_CLAMP:-0}
SOFT_CLAMP_MULTIPLIER=${SOFT_CLAMP_MULTIPLIER:-3.0}
ASYMMETRIC_LOG_COMPRESSION=${ASYMMETRIC_LOG_COMPRESSION:-0}
BASE_PERSISTENCE_ROUTING=${BASE_PERSISTENCE_ROUTING:-0}
BASE_PERSISTENCE_WINDOW=${BASE_PERSISTENCE_WINDOW:-4}
BASE_PERSISTENCE_MIN_GAIN=${BASE_PERSISTENCE_MIN_GAIN:-0}
OUTPUT_DIR=${OUTPUT_DIR:-results/rift_v2}
KD_TYPE=${KD_TYPE:-RIFT-v2-RECOVERY}
RUN_TAG=${RUN_TAG:-recovery_w${RECOVERY_WINDOW}_m${RECOVERY_MARGIN}_$(date +%Y%m%d_%H%M%S)}

RIFT_EXTRA_ARGS=()
if [[ "$EXACT_RANK" == 1 ]]; then
  RIFT_EXTRA_ARGS+=(--rift_exact_rank)
fi
if [[ "$REQUIRE_FULL_WINDOW" == 1 ]]; then
  RIFT_EXTRA_ARGS+=(--rift_require_full_window)
fi
if [[ "$GROUPWISE_RECOVERY_QUANTILES" == 1 ]]; then
  RIFT_EXTRA_ARGS+=(--rift_groupwise_recovery_quantiles)
fi
if [[ "$FORK_ONSET_ROUTING" == 1 ]]; then
  RIFT_EXTRA_ARGS+=(--rift_fork_onset_routing --rift_fork_onset_gap "$FORK_ONSET_GAP")
fi
if [[ "$REFLECTION_SAFE_WEIGHTING" == 1 ]]; then
  RIFT_EXTRA_ARGS+=(
    --rift_reflection_safe_weighting
    --rift_reflection_protection_weight "$REFLECTION_PROTECTION_WEIGHT"
    --rift_fork_onset_gap "$FORK_ONSET_GAP"
  )
fi
if [[ "$ASYMMETRIC_SOFT_CLAMP" == 1 ]]; then
  RIFT_EXTRA_ARGS+=(
    --rift_asymmetric_soft_clamp
    --rift_soft_clamp_multiplier "$SOFT_CLAMP_MULTIPLIER"
    --rift_fork_onset_gap "$FORK_ONSET_GAP"
  )
fi
if [[ "$ASYMMETRIC_LOG_COMPRESSION" == 1 ]]; then
  RIFT_EXTRA_ARGS+=(
    --rift_asymmetric_log_compression
    --rift_fork_onset_gap "$FORK_ONSET_GAP"
  )
fi
if [[ "$BASE_PERSISTENCE_ROUTING" == 1 ]]; then
  RIFT_EXTRA_ARGS+=(
    --rift_base_persistence_routing
    --rift_base_persistence_window "$BASE_PERSISTENCE_WINDOW"
    --rift_base_persistence_min_gain "$BASE_PERSISTENCE_MIN_GAIN"
  )
fi

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
  --output_dir "$OUTPUT_DIR" \
  --run_config "rift_v2_qwen3_4b_${RUN_TAG}" \
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
  --use_rift_routing \
  --rift_sign_margin "$SIGN_MARGIN" \
  --rift_entropy_quantile "$ENTROPY_QUANTILE" \
  --rift_route_weight "$ROUTE_WEIGHT" \
  --rift_recovery_window "$RECOVERY_WINDOW" \
  --rift_recovery_margin "$RECOVERY_MARGIN" \
  --rift_recovery_quantile "$RECOVERY_QUANTILE" \
  --rift_routing_score "$ROUTING_SCORE" \
  --rift_hard_entropy_quantile "$HARD_ENTROPY_QUANTILE" \
  --rift_hard_recovery_quantile "$HARD_RECOVERY_QUANTILE" \
  --jsd_token_clip 0.05 \
  --wandb_project OPSD-RIFT \
  --kd_type "$KD_TYPE" \
  "${RIFT_EXTRA_ARGS[@]}" \
  --seed "$SEED"
