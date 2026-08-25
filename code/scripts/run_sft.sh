accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 4 \
    --main_process_port 19346 \
    sft_train.py \
    --model_name_or_path PATH_TO_BASE_MODEL \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --output_dir results/sft/qwen31b/openthoughts \
    --num_train_epochs 4 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_length 16000 \
    --logging_steps 5 \
    --save_steps 20
