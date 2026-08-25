GRAD=${GRAD:-8}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-5e-6}
PORT=${PORT:-19346}
NGPU=${NGPU:-4}
BETA=${BETA:-0}
TASK=${TASK:-"math"}
DATA=${DATA:-"data/openthoughts_math_30k"}
CUVIS=${CUVIS:-0,1,2,3}
TIMEOUT=${TIMEOUT:-10}
MODEL=${MODEL:-"PATH_TO_BASE_MODEL"}
WANDB_PRO=${WANDB_PRO:-"GRPO"}

CUDA_VISIBLE_DEVICES="$CUVIS" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "$NGPU" \
    --gradient_accumulation_steps "$GRAD" \
    --main_process_port "$PORT" \
    grpo_train.py \
    --learning_rate "$LR" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD" \
    --model_name_or_path "$MODEL" \
    --output_dir results/grpo/ \
    --run_config qwen8b-2epoch \
    --num_train_epochs 2 \
    --num_iterations 2 \
    --gradient_checkpointing \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_prompt_length 2048 \
    --max_completion_length 16000 \
    --num_generations 8 \
    --temperature 1.2 \
    --use_vllm \
    --use_peft \
    --vllm_mode colocate \
    --logging_steps 10 \
    --save_steps 20 \
    --beta 0.0 \
    --loss_type grpo \
    --scale_rewards group \
    --wandb_project "$WANDB_PRO" \
    --max_steps 301 \
    --task_type "$TASK" \
    --dataset_path "$DATA" \
    --code_exec_timeout "$TIMEOUT"
