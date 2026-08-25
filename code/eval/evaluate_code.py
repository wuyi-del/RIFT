import torch
import argparse
import json
import re
import os
import numpy as np
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter

from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from evalplus.sanitize import sanitize as evalplus_sanitize


def strip_thinking_content(text: str) -> str:
    """Strip thinking/reasoning content from model output."""
    # Strip <think>...</think> tags (Qwen3 style)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip <|begin_of_thought|>...<|end_of_thought|> tags
    text = re.sub(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", "", text, flags=re.DOTALL)
    return text.strip()


def build_prompt(problem: dict, enable_thinking: bool, tokenizer) -> str:
    """Build chat prompt for HumanEval+ or MBPP+ problems."""
    prompt_text = problem["prompt"]
    user_message = (
        f"{prompt_text}\n Write Python code to solve the problem. Present the code in ```python Your code ```  at the end.\n You need to think first then write the Python code"
    )
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


def postprocess_completion(generated_text: str, problem: dict, dataset_name: str) -> str:
    """
    Post-process generated text using evalplus.sanitize for robust code extraction.

    Uses tree-sitter based sanitization to extract clean, runnable code
    from model output (handles markdown blocks, thinking content, etc.).

    Returns complete solution code for both HumanEval+ and MBPP+.
    """
    # Strip thinking content first
    text = strip_thinking_content(generated_text)

    entry_point = problem["entry_point"]

    if dataset_name == "humaneval":
        # In chat/instruction mode the model typically generates the full function
        # definition. Only prepend the prompt (function signature + docstring) when
        # the generated text does NOT already contain the target function, i.e. the
        # model produced just the function body (completion-style output).
        if f"def {entry_point}" in text:
            sanitized = evalplus_sanitize(text, entry_point)
        else:
            full_code = problem["prompt"] + text
            sanitized = evalplus_sanitize(full_code, entry_point)
        return sanitized
    else:  # mbpp
        sanitized = evalplus_sanitize(text, entry_point)
        return sanitized


def load_vllm_model(
    base_model_path: str,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int = None,
    enable_thinking: bool = True,
):
    """Load a model using vLLM for fast inference."""
    print(f"Loading model with vLLM from: {base_model_path}")

    if max_model_len is None:
        max_model_len = 40960 if enable_thinking else 32768
        print(f"Auto-setting max_model_len to {max_model_len} for {'thinking' if enable_thinking else 'non-thinking'} mode")

    llm_config = {
        "model": base_model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "distributed_executor_backend": "mp",
        "enforce_eager": True,
    }

    if lora_adapter_path is not None:
        print(f"LoRA adapter path provided: {lora_adapter_path}")
        adapter_path = Path(lora_adapter_path) / "adapter_model.safetensors"
        if not adapter_path.exists():
            adapter_path = Path(lora_adapter_path) / "adapter_model.bin"

        if adapter_path.exists():
            print("LoRA weights found. Enabling LoRA support...")
            llm_config["enable_lora"] = True
            llm_config["max_lora_rank"] = 64
            llm_config["max_loras"] = 1
            llm_config["max_cpu_loras"] = 1
        else:
            print(f"Warning: No LoRA weights found at {lora_adapter_path}")
            print("Continuing with base model only...")
            lora_adapter_path = None

    llm = LLM(**llm_config)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print("\n" + "=" * 70)
    print("MODEL DTYPE INFORMATION")
    print("=" * 70)
    print(f"vLLM Model Config dtype: {llm.llm_engine.model_config.dtype}")
    print(f"vLLM Model quantization: {llm.llm_engine.model_config.quantization}")
    print(f"KV cache dtype: {llm.llm_engine.cache_config.cache_dtype}")
    print("=" * 70 + "\n")

    print("vLLM model loaded successfully!")
    return llm, tokenizer


def run_evalplus_evaluation(samples_path: str, dataset_name: str) -> dict:
    """
    Run evalplus evaluation on generated samples and return parsed results.

    Returns dict with:
        - base_results: {task_id: [bool, ...]} pass/fail per sample (base tests)
        - plus_results: {task_id: [bool, ...]} pass/fail per sample (base+plus tests)
        - eval_results: raw eval results dict
    """
    from evalplus.evaluate import evaluate as evalplus_evaluate

    # evalplus.evaluate writes results to a file next to the samples
    evalplus_evaluate(
        dataset=dataset_name,
        samples=samples_path,
        parallel=max(1, os.cpu_count() // 2),
        i_just_wanna_run=True,
        test_details=True,
    )

    # Load the results file
    result_path = samples_path.replace(".jsonl", "_eval_results.json")
    with open(result_path, "r") as f:
        eval_results = json.load(f)

    return eval_results


def evaluate_code(
    llm,
    tokenizer,
    max_new_tokens: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = -1,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    num_samples: int = None,
    output_file: str = None,
    lora_request=None,
    dataset_name: str = "humaneval",
    base_model_name: str = None,
    enable_thinking: bool = True,
    val_n: int = 1,
    checkpoint_dir: str = None,
):
    """
    Evaluate model on HumanEval+ or MBPP+ using evalplus.

    Args:
        llm: The vLLM LLM instance
        tokenizer: The tokenizer for chat template
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        min_p: Minimum probability threshold
        presence_penalty: Presence penalty
        num_samples: Number of problems to evaluate (None = all)
        output_file: Path to save detailed results JSON
        lora_request: Optional LoRA request for inference
        dataset_name: "humaneval" or "mbpp"
        base_model_name: Base model name for logging
        enable_thinking: Whether to use thinking mode
        val_n: Number of solutions per problem
        checkpoint_dir: Checkpoint directory path for logging
    """
    print(f"\n{'='*70}")
    print(f"CODE EVALUATION CONFIGURATION")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name.upper()}+")
    print(f"Thinking Mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Temperature: {temperature}")
    print(f"Top-P: {top_p}")
    print(f"Top-K: {top_k}")
    print(f"Min-P: {min_p}")
    print(f"Presence Penalty: {presence_penalty}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Val-N (solutions per problem): {val_n}")
    print(f"{'='*70}\n")

    # Load dataset
    if dataset_name == "humaneval":
        problems = get_human_eval_plus()
    elif dataset_name == "mbpp":
        problems = get_mbpp_plus()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'humaneval' or 'mbpp'")

    task_ids = list(problems.keys())
    print(f"Loaded {len(task_ids)} problems from {dataset_name.upper()}+")

    if num_samples:
        task_ids = task_ids[:min(num_samples, len(task_ids))]

    num_problems = len(task_ids)

    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_new_tokens,
        presence_penalty=presence_penalty,
        n=val_n,
    )

    # Build prompts
    all_prompts = []
    all_task_ids = []
    for task_id in task_ids:
        prompt = build_prompt(problems[task_id], enable_thinking, tokenizer)
        all_prompts.append(prompt)
        all_task_ids.append(task_id)

    # Generate
    print(f"\nRunning vLLM batch inference on {len(all_prompts)} problems...")
    print(f"Using LoRA: {lora_request is not None}")

    if lora_request is not None:
        outputs = llm.generate(all_prompts, sampling_params, lora_request=lora_request, use_tqdm=True)
    else:
        outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

    # Process generations and build evalplus samples
    print("\nProcessing generations...")
    evalplus_samples = []
    results = []
    total_length = 0
    formatted_count = 0
    total = 0

    for idx, (output, task_id) in enumerate(zip(outputs, all_task_ids)):
        problem = problems[task_id]
        generations = []
        completions = []

        for i in range(len(output.outputs)):
            generated_text = output.outputs[i].text
            generations.append(generated_text)

            # Post-process to get code completion
            completion = postprocess_completion(generated_text, problem, dataset_name)
            completions.append(completion)

            # Track token lengths
            gen_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
            total_length += len(gen_tokens)
            total += 1

            # Check if code was properly extracted (has some substance)
            has_code = len(completion.strip()) > 0 and any(
                kw in completion for kw in ["return", "def ", "for ", "if ", "while ", "=", "print"]
            )
            if has_code:
                formatted_count += 1

            # Write to evalplus format: one entry per (task_id, sample)
            # Use "solution" (complete runnable code) since evalplus_sanitize
            # returns the full function definition for both datasets
            evalplus_samples.append({
                "task_id": task_id,
                "solution": completion,
            })

        results.append({
            "problem_id": task_id,
            "prompt": problem["prompt"],
            "entry_point": problem["entry_point"],
            "val_n": val_n,
            "generations": [
                {"completion": comp, "full_generation": gen}
                for comp, gen in zip(completions, generations)
            ],
        })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{num_problems} problems")

    # Write evalplus samples to temp JSONL
    output_dir = Path(output_file).parent if output_file else Path("eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = str(output_dir / f"evalplus_samples_{dataset_name}.jsonl")
    write_jsonl(samples_path, evalplus_samples)
    print(f"\nWrote {len(evalplus_samples)} samples to {samples_path}")

    # Run evalplus evaluation
    print(f"\nRunning evalplus evaluation on {dataset_name}+...")
    eval_results = run_evalplus_evaluation(samples_path, dataset_name)

    # Parse evalplus results to compute metrics matching math eval format
    PASS = "pass"
    eval_data = eval_results.get("eval", {})

    # Per-problem pass/fail analysis
    pass_at_n = 0  # problems with at least one correct solution
    total_correct_per_problem = 0  # total correct solutions across all problems
    majority_vote_correct_count = 0

    for task_id in task_ids:
        task_results = eval_data.get(task_id, [])
        # A solution passes if it passes both base and plus tests
        correct_list = [
            r.get("base_status", "") == PASS and r.get("plus_status", "") == PASS
            for r in task_results
        ]

        num_correct = sum(correct_list)
        has_correct = any(correct_list)

        if has_correct:
            pass_at_n += 1
        total_correct_per_problem += num_correct

        # Majority vote: for code, we count the most common pass/fail status
        if len(correct_list) > 0:
            if Counter(correct_list).most_common(1)[0][0]:
                majority_vote_correct_count += 1

        # Enrich results with correctness info
        for result in results:
            if result["problem_id"] == task_id:
                result["num_correct"] = num_correct
                result["pass_at_n"] = has_correct
                result["majority_vote_correct"] = (
                    Counter(correct_list).most_common(1)[0][0] if correct_list else False
                )
                for i, gen in enumerate(result["generations"]):
                    gen["correct"] = correct_list[i] if i < len(correct_list) else False
                    # Also store base-only result
                    if i < len(task_results):
                        gen["base_pass"] = task_results[i].get("base_status", "") == PASS
                        gen["plus_pass"] = task_results[i].get("plus_status", "") == PASS
                break

    # Compute final metrics
    total_solutions = num_problems * val_n
    format_rate = formatted_count / total * 100 if total > 0 else 0.0
    avg_length = total_length / total if total > 0 else 0.0
    pass_at_n_pct = pass_at_n / num_problems * 100 if num_problems > 0 else 0.0
    average_at_n_pct = total_correct_per_problem / total_solutions * 100 if total_solutions > 0 else 0.0
    majority_vote_at_n_pct = majority_vote_correct_count / num_problems * 100 if num_problems > 0 else 0.0

    # Also compute base-only metrics
    base_pass_at_n = 0
    base_total_correct = 0
    for task_id in task_ids:
        task_results = eval_data.get(task_id, [])
        base_correct_list = [r.get("base_status", "") == PASS for r in task_results]
        if any(base_correct_list):
            base_pass_at_n += 1
        base_total_correct += sum(base_correct_list)

    base_pass_at_n_pct = base_pass_at_n / num_problems * 100 if num_problems > 0 else 0.0
    base_average_at_n_pct = base_total_correct / total_solutions * 100 if total_solutions > 0 else 0.0

    # Print results
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS")
    print("=" * 70)
    print(f"Base Model: {base_model_name}")
    print(f"Checkpoint: {checkpoint_dir if checkpoint_dir else 'None (base model only)'}")
    print(f"Dataset: {dataset_name.upper()}+")
    print(f"Thinking Mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Total problems: {num_problems}")
    print(f"Solutions per problem: {val_n}")
    print(f"Total solutions: {total_solutions}")
    print(f"\n{dataset_name.upper()}+ Metrics (base + extra tests):")
    print(f"  Pass@{val_n}: {pass_at_n_pct:.2f}% ({pass_at_n}/{num_problems})")
    print(f"  Average@{val_n}: {average_at_n_pct:.2f}% ({total_correct_per_problem}/{total_solutions})")
    print(f"  Majority Vote@{val_n}: {majority_vote_at_n_pct:.2f}% ({majority_vote_correct_count}/{num_problems})")
    print(f"\n{dataset_name.upper()} Metrics (base tests only):")
    print(f"  Pass@{val_n}: {base_pass_at_n_pct:.2f}% ({base_pass_at_n}/{num_problems})")
    print(f"  Average@{val_n}: {base_average_at_n_pct:.2f}% ({base_total_correct}/{total_solutions})")
    print(f"\nFormatting:")
    print(f"  Valid code extractions: {formatted_count}/{total}")
    print(f"  Format rate: {format_rate:.2f}%")
    print(f"\nGeneration Length:")
    print(f"  Total tokens generated: {total_length}")
    print(f"  Average generation length: {avg_length:.2f} tokens")
    print("=" * 70)

    # Save detailed results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "base_model": base_model_name,
            "checkpoint_dir": checkpoint_dir,
            "dataset": dataset_name,
            "enable_thinking": enable_thinking,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "max_new_tokens": max_new_tokens,
            "val_n": val_n,
            "num_problems": num_problems,
            "total_solutions": total_solutions,
            "pass_at_n": pass_at_n,
            "pass_at_n_pct": pass_at_n_pct,
            "average_at_n": total_correct_per_problem,
            "average_at_n_pct": average_at_n_pct,
            "majority_vote_at_n": majority_vote_correct_count,
            "majority_vote_at_n_pct": majority_vote_at_n_pct,
            "formatted_count": formatted_count,
            "format_rate": format_rate,
            "avg_generation_length": avg_length,
            "total_tokens_generated": total_length,
            "base_pass_at_n": base_pass_at_n,
            "base_pass_at_n_pct": base_pass_at_n_pct,
            "base_average_at_n": base_total_correct,
            "base_average_at_n_pct": base_average_at_n_pct,
            "results": results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {output_file}")

    return average_at_n_pct, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on HumanEval+/MBPP+ coding benchmarks")
    parser.add_argument(
        "--base_model", type=str, required=True, help="Path to base model",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Path to checkpoint directory with LoRA adapters. If not provided, uses base model only.",
    )
    parser.add_argument(
        "--dataset", type=str, default="humaneval", choices=["humaneval", "mbpp"],
        help="Dataset to use: humaneval (HumanEval+) or mbpp (MBPP+) (default: humaneval)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=16384,
        help="Maximum tokens to generate (default: 16384)",
    )
    parser.add_argument(
        "--enable_thinking", action="store_true", default=True,
        help="Enable thinking mode (default: True)",
    )
    parser.add_argument(
        "--no_thinking", dest="enable_thinking", action="store_false",
        help="Disable thinking mode",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k sampling parameter (default: -1, disabled)")
    parser.add_argument("--min_p", type=float, default=0.0, help="Minimum probability threshold (default: 0.0)")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty (default: 0.0)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of problems to evaluate (None = all)")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save detailed results JSON")
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="GPU memory utilization for vLLM (0.0 to 1.0, default: 0.9)",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument("--max_model_len", type=int, default=None, help="Maximum model context length")
    parser.add_argument("--val_n", type=int, default=1, help="Number of solutions per problem (default: 1)")

    args = parser.parse_args()

    # Validate checkpoint directory
    if args.checkpoint_dir is not None:
        checkpoint_path = Path(args.checkpoint_dir)
        if not checkpoint_path.exists():
            print(f"\nERROR: Checkpoint directory does not exist: {args.checkpoint_dir}")
            exit(1)

    if args.top_p is None:
        args.top_p = 0.95 if args.enable_thinking else 0.8
        print(f"Auto-setting top_p to {args.top_p} for {'thinking' if args.enable_thinking else 'non-thinking'} mode")

    # Auto-generate output file
    if args.output_file is None:

        if args.checkpoint_dir:
            args.output_file = str(
                Path("eval_results") /
                args.dataset /
                Path(args.base_model).name /
                checkpoint_path.parent.parent.parent.name /
                checkpoint_path.parent.parent.name /
                checkpoint_path.parent.name /
                checkpoint_path.name /
                f"{'thinking' if args.enable_thinking else 'nonthinking'}_valn{args.val_n}.json"
            )
        else:
            args.output_file = str(
                Path("eval_results") /
                args.dataset /
                Path(args.base_model).name /
                f"{'thinking' if args.enable_thinking else 'nonthinking'}_valn{args.val_n}.json"
            )
    
    

    print(f"Results will be saved to: {args.output_file}")

    print("\n" + "=" * 70)
    print("CODE EVALUATION WITH EVALPLUS")
    print("=" * 70)
    print(f"Dataset: {args.dataset.upper()}+")
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint_dir or 'None (base model only)'}")
    print(f"Thinking Mode: {'ENABLED' if args.enable_thinking else 'DISABLED'}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Min-p: {args.min_p}")
    print(f"Presence penalty: {args.presence_penalty}")
    print(f"Val-N: {args.val_n}")
    print(f"Output file: {args.output_file}")
    print("=" * 70 + "\n")

    # Load model
    llm, tokenizer = load_vllm_model(
        args.base_model,
        args.checkpoint_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )

    # Setup LoRA request
    lora_request = None
    if args.checkpoint_dir is not None:
        try:
            from vllm.lora.request import LoRARequest

            adapter_safetensors = Path(args.checkpoint_dir) / "adapter_model.safetensors"
            adapter_bin = Path(args.checkpoint_dir) / "adapter_model.bin"

            if adapter_safetensors.exists() or adapter_bin.exists():
                lora_request = LoRARequest("checkpoint_lora", 1, args.checkpoint_dir)
                print(f"Successfully created LoRA request for: {args.checkpoint_dir}")
            else:
                print(f"Warning: No LoRA adapter weights found at {args.checkpoint_dir}")
                print("Continuing with base model only...")
        except ImportError:
            print("Warning: Could not import LoRARequest. Running without LoRA.")
        except Exception as e:
            print(f"Warning: Could not create LoRA request: {e}")
            print("Continuing without LoRA.")

    # Run evaluation
    average_at_n_pct, results = evaluate_code(
        llm,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        num_samples=args.num_samples,
        output_file=args.output_file,
        lora_request=lora_request,
        dataset_name=args.dataset,
        base_model_name=args.base_model,
        enable_thinking=args.enable_thinking,
        val_n=args.val_n,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"Final Average@{args.val_n}: {average_at_n_pct:.2f}%")
    print(f"Results saved to: {args.output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
