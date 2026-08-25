import torch


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: sees only the problem (with chat template)
    Teacher: sees problem + solution + transition prompt (with chat template)

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    # Task-specific prompt components
    TASK_PROMPTS = {
        "math": {
            "student_instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
            "teacher_instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
            "reason_first_prompt": (
                "\n\nThe reference reasoning above arrives at the correct answer. "
                "Please analyze this solution and explain the key reasoning steps and problem-solving strategies employed. "
                "Do NOT use <think> tags. Do NOT derive your own solution. "
                "Simply analyze and explain the reference solution provided above.\n"
            ),
            "transition_prompt": (
                "\n\nAfter reading the reference solution above, make sure you truly understand "
                "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
                "own words and independent reasoning, derive the same final answer to the problem above. "
                "Think step by step, explore different approaches, and don't be afraid to backtrack "
                "or reconsider if something doesn't work out:\n"
            ),
        },
        "coding": {
            "student_instruction": (
                "Write Python code to solve the problem. "
                "Present the code in ```python\nYour code\n``` at the end. "
                "You need to think first then write the Python code."
            ),
            "teacher_instruction": (
                "Write Python code to solve the problem. "
                "Present the code in ```python\nYour code\n``` at the end. "
                "You need to think first then write the Python code."
            ),
            "reason_first_prompt": (
                "\n\nThe reference solution above solves the problem correctly. "
                "Please analyze this solution and explain the key algorithmic ideas, data structures, "
                "and implementation strategies employed. "
                "Do NOT use <think> tags. Do NOT write your own solution. "
                "Simply analyze and explain the reference solution provided above.\n"
            ),
            "transition_prompt": (
                "\n\nAfter reading the reference solution above, make sure you truly understand "
                "the algorithm and implementation — do not copy or paraphrase it. Now, using your "
                "own words and independent reasoning, solve the same problem from scratch. "
                "Think step by step, consider edge cases.\n"
            ),
        },
    }

    def __init__(self, tokenizer, max_length=2048, reason_first=True, task_type="math"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.task_type = task_type

        if task_type not in self.TASK_PROMPTS:
            raise ValueError(f"Unknown task_type: {task_type}. Supported: {list(self.TASK_PROMPTS.keys())}")

        prompts = self.TASK_PROMPTS[task_type]
        self.student_instruction = prompts["student_instruction"]
        self.teacher_instruction = prompts["teacher_instruction"]
        self.reason_first_prompt = prompts["reason_first_prompt"]
        self.transition_prompt = prompts["transition_prompt"]

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
        print(f"[DataCollator] Task type: {self.task_type}")

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # NEW: for reason_first mode

        for feature in features:
            # Extract problem and solution from dataset
            # Handle different possible column names
            problem = feature["problem"]
            solution = feature["solution"]

            # Student prompt: problem with task-specific instruction
            student_user_message = f"Problem: {problem}\n\n{self.student_instruction}"
            student_messages = [{"role": "user", "content": student_user_message}]

            # Apply chat template for student (matching evaluation)
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            student_prompts.append(student_prompt)

            if self.reason_first:
                # Reasoning prompt: ask teacher to analyze the solution
                reasoning_user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a correct reasoning to this problem:"
                    f"=== Reference Reasoning Start ===\n"
                    f"{solution}\n"
                    f"=== Reference Reasoning End ===\n\n"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages, tokenize=False, add_generation_prompt=True
                )
                teacher_reasoning_prompts.append(reasoning_prompt)

                # Teacher prompt will be constructed during training after reasoning
                # For now, create placeholder (will be replaced in training_step)
                teacher_prompts.append("")  # Placeholder
            else:
                # Original teacher prompt (unchanged)
                teacher_user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a reference solution to this problem:\n"
                    f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                    f"{self.transition_prompt}\n"
                    f"{self.teacher_instruction}"
                )
                teacher_messages = [{"role": "user", "content": teacher_user_message}]

                # Apply chat template for teacher
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
                teacher_prompts.append(teacher_prompt)

        # Tokenize WITHOUT padding first to get true lengths
        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]

        # Find max lengths in this batch
        max_student_prompt_len = max(student_prompt_lengths)

        # Tokenize WITH padding to max length in batch
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,  # Single value for batch!
            # Keep individual lengths for proper masking
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }

        if self.reason_first:
            # Tokenize reasoning prompts
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            # Tokenize transition prompt (this will be appended after reasoning)
            # Don't use chat template here - just the raw text
            transition_text = f"\n{self.transition_prompt}\n{self.teacher_instruction}"
            transition_encoded = self.tokenizer(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            # Normal mode: tokenize teacher prompts
            teacher_encoded_no_pad = self.tokenizer(
                teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
            max_teacher_prompt_len = max(teacher_prompt_lengths)

            teacher_encoded = self.tokenizer(
                teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_teacher_prompt_len,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                }
            )

        # Pass ground truth answers for signal analysis (correctness evaluation)
        result["answer_gt"] = [feature.get("Answer", None) for feature in features]

        return result


class GKDDataCollator:
    """
    Data collator for GKD with mixed on-policy SGO and off-policy SFT-CoT training.

    Expected dataset columns:
    - problem: math problem / question
    - COT_Reason: reference chain-of-thought completion

    Student and teacher both only see the prompt. The teacher prompt can use a
    matching chat-template mode, and neither receives privileged ground-truth
    reasoning as context.
    """

    def __init__(self, tokenizer, max_length=2048, max_completion_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_completion_length = max_completion_length

        print(f"[GKDDataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[GKDDataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[GKDDataCollator] Max prompt length: {self.max_length}")
        print(f"[GKDDataCollator] Max completion length: {self.max_completion_length}")

    def _build_student_prompt(self, problem: str) -> str:
        user_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        messages = [{"role": "user", "content": user_message}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

    def _build_teacher_prompt(self, problem: str) -> str:
        user_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        messages = [{"role": "user", "content": user_message}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

    def __call__(self, features):
        student_prompts = []
        teacher_prompts = []
        completion_token_seqs = []

        for feature in features:
            problem = feature["problem"]
            cot = feature["COT_Reason"]

            student_prompts.append(self._build_student_prompt(problem))
            teacher_prompts.append(self._build_teacher_prompt(problem))
            completion_ids = self.tokenizer(
                cot,
                padding=False,
                truncation=True,
                max_length=self.max_completion_length,
                add_special_tokens=False,
            )["input_ids"]
            if (
                self.tokenizer.eos_token_id is not None
                and len(completion_ids) < self.max_completion_length
                and (len(completion_ids) == 0 or completion_ids[-1] != self.tokenizer.eos_token_id)
            ):
                completion_ids = completion_ids + [self.tokenizer.eos_token_id]
            completion_token_seqs.append(completion_ids)

        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]
        max_student_prompt_len = max(student_prompt_lengths)
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        teacher_encoded_no_pad = self.tokenizer(
            teacher_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
        max_teacher_prompt_len = max(teacher_prompt_lengths)
        teacher_encoded = self.tokenizer(
            teacher_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_teacher_prompt_len,
            return_tensors="pt",
        )

        completion_lengths = [len(ids) for ids in completion_token_seqs]
        max_completion_len = max(completion_lengths) if completion_lengths else 0
        padded_completion_ids = []
        padded_completion_attention_mask = []
        target_completion_len = max_completion_len if max_completion_len > 0 else 1
        for ids in completion_token_seqs:
            pad_len = target_completion_len - len(ids)
            padded_completion_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_completion_attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
            "teacher_prompts": teacher_encoded["input_ids"],
            "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
            "teacher_prompt_length": max_teacher_prompt_len,
            "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
            "completion_ids": torch.tensor(padded_completion_ids, dtype=torch.long),
            "completion_attention_mask": torch.tensor(padded_completion_attention_mask, dtype=torch.long),
            "completion_lengths_per_example": torch.tensor(completion_lengths),
        }
