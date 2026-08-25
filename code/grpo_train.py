import os
import wandb
import re
import json

from math_verify import parse, verify

from transformers import AutoTokenizer

from trl import (
    GRPOTrainer,
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.extras.profiling import profiling_decorator
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with GRPO-specific options."""

    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    wandb_entity: str = field(
        default=None,
        metadata={"help": "WandB entity (username or team name) to log runs under."},
    )
    wandb_project: str = field(
        default="grpo-training",
        metadata={"help": "WandB project name to log runs under."},
    )
    kd_type: str = field(
        default="SFT",
        metadata={
            "help": "model training type: SFT, Distillm, KD, GKD, OPSD, etc."
        }
    )
    dataset_path: str = field(
        default="data/openthoughts_math_30k",
        metadata={
            "help": "Path to local dataset. Can be a JSON file, JSONL file, or directory containing Arrow/Parquet files. "
        },
    )
    task_type: str = field(
        default="math",
        metadata={"help": "Task type: math (math_verify reward) or coding (code execution reward)."},
    )
    code_exec_timeout: int = field(
        default=10,
        metadata={"help": "Timeout in seconds for each code execution test case (coding task only)."},
    )


def extract_boxed_answer(text):
    """
    Extract the answer from \\boxed{} format.
    For thinking models, only searches after </think> to avoid picking up
    intermediate answers from the thinking block.
    Handles nested braces correctly (e.g. \\boxed{\\frac{1}{2}}).
    """
    # For thinking models (e.g. Qwen3), only look after </think>
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>") :] if think_end != -1 else text

    idx = search_text.find(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(search_text) and depth > 0:
        if search_text[i] == "{":
            depth += 1
        elif search_text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return search_text[start : i - 1].strip()
    return None


def _preprocess_for_parse(answer):
    """Convert ratio notation a:b → \\frac{a}{b} so math_verify can parse it."""
    if answer is None:
        return None
    ratio_match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer)
    if ratio_match:
        return rf"\frac{{{ratio_match.group(1)}}}{{{ratio_match.group(2)}}}"
    return answer


def reward_correctness(completions, Answer, **kwargs):
    rewards = []
    for i, (completion, ground_truth) in enumerate(zip(completions, Answer)):
        pred_answer = extract_boxed_answer(completion)

        reward = 0.0

        # Try math_verify for mathematical equivalence (handles fractions, algebra, etc.)
        # Only use it when both sides actually parse to something (avoids silent None returns
        # for MCQ answers like "E" which parse() returns None for)
        gold_parsed = parse(ground_truth)
        pred_parsed = parse(_preprocess_for_parse(pred_answer))
        if gold_parsed is not None and pred_parsed is not None:
            try:
                reward = 1.0 if verify(gold_parsed, pred_parsed) else 0.0
            except Exception:
                pass

        # Fallback: whitespace-stripped string match (handles MCQ like "E", "A", etc.)
        if reward == 0.0:
            pred_norm = re.sub(r"\s+", "", pred_answer or "").lower()
            gt_norm = re.sub(r"\s+", "", ground_truth or "").lower()
            if pred_norm and pred_norm == gt_norm:
                reward = 1.0

        rewards.append(reward)

    return rewards


def _strip_thinking(text: str) -> str:
    """Strip thinking/reasoning content from model output (Qwen3 style)."""
    # Strip Qwen3 thinking tags
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL)
    # Strip alternative thought tags
    text = re.sub(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_code(text: str) -> str | None:
    """Extract the first ```python ... ``` block from completion text."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _run_code_against_test_cases(code: str, test_cases_str: str, timeout: int = 10, max_test_cases: int = 5) -> float:
    """
    Execute code against test cases and return the pass rate.
    
    test_cases_str is a JSON string: {"inputs": ["...", ...], "outputs": ["...", ...]}
    Returns fraction of passed test cases (0.0 to 1.0).
    
    Limits to max_test_cases to keep reward computation fast and balanced across ranks.
    Launches all test cases in parallel via Popen for speed.
    """
    import subprocess
    
    try:
        tc = json.loads(test_cases_str)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    inputs = tc.get("inputs", [])
    expected_outputs = tc.get("outputs", [])
    if not inputs or not expected_outputs:
        return 0.0

    # Limit number of test cases for speed and rank balance
    inputs = inputs[:max_test_cases]
    expected_outputs = expected_outputs[:max_test_cases]

    # Launch all test cases in parallel
    procs = []
    for inp in inputs:
        try:
            p = subprocess.Popen(
                ["python3", "-c", code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            p._input_data = inp
            procs.append(p)
        except Exception:
            procs.append(None)

    # Collect results with early exit on consecutive failures
    passed = 0
    consecutive_fails = 0
    for p, expected in zip(procs, expected_outputs):
        if p is None:
            consecutive_fails += 1
        else:
            try:
                stdout, _ = p.communicate(input=p._input_data, timeout=timeout)
                if stdout == expected:
                    passed += 1
                    consecutive_fails = 0
                else:
                    consecutive_fails += 1
            except (subprocess.TimeoutExpired, Exception):
                p.kill()
                p.wait()
                consecutive_fails += 1

        if consecutive_fails >= 3:
            # Kill remaining processes
            for remaining in procs:
                if remaining is not None and remaining.poll() is None:
                    remaining.kill()
                    remaining.wait()
            break

    return passed / len(inputs)


def reward_code_correctness(completions, test_cases, **kwargs):
    """Reward function for coding tasks: execute generated code against test cases.
    
    Called by GRPOTrainer as: reward_func(prompts=..., completions=..., completion_ids=..., test_cases=..., trainer_state=...)
    """
    rewards = []
    for completion, tc_str in zip(completions, test_cases):
        text = _strip_thinking(completion)
        code = _extract_code(text)
        if code is None:
            rewards.append(0.0)
            continue
        rewards.append(_run_code_against_test_cases(code, tc_str, timeout=10))
    return rewards


def make_format_prompt(tokenizer, task_type: str = "math"):
    """
    Returns a formatting function that applies the tokenizer's chat template.
    """

    if task_type == "coding":
        def format_prompt(example):
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Problem: {example['problem']}\n"
                        "Write Python code to solve the problem. "
                        "Present the code in ```python\nYour code\n``` at the end. "
                        "You need to think first then write the Python code."
                    ),
                }
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return {"prompt": prompt, "test_cases": example["test_cases"]}

        return format_prompt

    # math (default)
    def format_prompt(example):
        messages = [
            {
                "role": "user",
                "content": f"Problem: {example['Question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "Answer": example["Answer"]}

    return format_prompt


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-5 -> "2e-5")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_name = (
            f"GRPO_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_"
            f"temp{training_args.temperature}"
        )

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_name}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Num Generations: {training_args.num_generations}")
    print(f"Temperature: {training_args.temperature}")
    print(f"Max Prompt Length: {training_args.max_prompt_length}")
    print(f"Max Completion Length: {training_args.max_completion_length}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "num_generations": training_args.num_generations,
                "max_prompt_length": training_args.max_prompt_length,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "loss_type": training_args.loss_type,
                "scale_rewards": training_args.scale_rewards,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    if script_args.dataset_path:
        print(f"\n{'='*80}")
        print(f"Loading dataset from local path: {script_args.dataset_path}")
        print(f"{'='*80}\n")
        if script_args.dataset_path.endswith(('.json', '.jsonl')):
            dataset = load_dataset(
                "json",
                data_files=script_args.dataset_path,
                split="train"
            )
        else:
            import pandas as pd
            
            if os.path.isdir(script_args.dataset_path):
                parquet_files = [os.path.join(script_args.dataset_path, f) 
                                for f in os.listdir(script_args.dataset_path) 
                                if f.endswith('.parquet')]
                print(f"Found {len(parquet_files)} parquet files")
                dfs = [pd.read_parquet(f) for f in parquet_files]
                df = pd.concat(dfs, ignore_index=True)
            else:
                df = pd.read_parquet(script_args.dataset_path)
            
            dataset = Dataset.from_pandas(df)
            print(f"Loaded {len(dataset)} examples from parquet files")
        
    else:
        print(f"\n{'='*80}")
        print(f"{'='*80}\n")
        dataset = load_dataset("PATH_TO_DATASET")
        dataset = dataset["train"]

    train_dataset = dataset if isinstance(dataset, Dataset) else dataset["train"]


    # Apply the format_prompt function to create the expected structure
    task_type = script_args.task_type
    format_prompt = make_format_prompt(tokenizer, task_type=task_type)
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    split_dataset = train_dataset.train_test_split(test_size=0.007, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Select reward function based on task_type
    if task_type == "coding":
        reward_func = reward_code_correctness
    else:
        reward_func = reward_correctness

    ################
    # Training
    ################
    # GRPOTrainer.training_step = profiling_decorator(GRPOTrainer.training_step)

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Auto-resume from latest checkpoint if one exists
    resume_from_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoints = sorted(
            [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if checkpoints:
            resume_from_checkpoint = os.path.join(training_args.output_dir, checkpoints[-1])
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save model
    trainer.save_model(training_args.output_dir)
