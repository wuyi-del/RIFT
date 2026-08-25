import os
import subprocess
import wandb

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, GenerationConfig, set_seed

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from opsd_trainer import OPSDTrainer
from dataclasses import dataclass, field

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    teacher_path: str = field(
        default=None,
        metadata={
            "help": "Path or name of the teacher model. If specified, will use a separate teacher model "
            "instead of self-distillation. The teacher model is kept frozen during training."
        },
    )
    teacher_lora_path: str = field(
        default=None,
        metadata={
            "help": "Path to a LoRA adapter for the teacher model. When specified, loads the base model "
            "from teacher_path and applies the LoRA adapter on top. The merged model is then frozen. "
            "This is more memory-efficient than loading a full separate teacher model. "
            "Requires teacher_path to be set (as the base model path)."
        },
    )
    kd_type: str = field(
        default="SFT",
        metadata={
            "help": "model training type: SFT, Distillm, KD, GKD, OPSD, etc."
        }
    )
    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates. "
            "Not compatible with teacher_path."
        },
    )
    use_rift_routing: bool = field(
        default=False,
        metadata={
            "help": "Enable RIFT sign-consistent privilege routing. At high-entropy tokens where the "
            "privileged fixed teacher suppresses the sampled token but the unprivileged fixed teacher does not, "
            "route the dense JSD target to the unprivileged teacher. Requires fixed_teacher=True and full-vocab JSD."
        },
    )
    rift_sign_margin: float = field(
        default=0.05,
        metadata={
            "help": "Log-probability tolerance for RIFT sign-conflict detection. The privileged advantage must be "
            "below -margin while the unprivileged advantage is at least -margin."
        },
    )
    rift_entropy_quantile: float = field(
        default=0.75,
        metadata={
            "help": "Per-trajectory student-entropy quantile above which RIFT may route teacher supervision."
        },
    )
    rift_route_weight: float = field(
        default=1.0,
        metadata={
            "help": "Interpolation weight from privileged JSD to unprivileged JSD at RIFT-routed positions."
        },
    )
    rift_recovery_window: int = field(
        default=0,
        metadata={
            "help": "RIFT-v2 future-realignment window in tokens. Values > 0 route only candidate forks whose "
            "privileged JSD drops by rift_recovery_margin within this window; 0 preserves RIFT-v1 behavior."
        },
    )
    rift_recovery_margin: float = field(
        default=0.0,
        metadata={
            "help": "Minimum privileged-JSD reduction required inside rift_recovery_window for RIFT-v2 routing."
        },
    )
    rift_recovery_quantile: float = field(
        default=-1.0,
        metadata={
            "help": "Per-trajectory recovery-score quantile for adaptive RIFT routing. Values in [0, 1] route "
            "candidates at or above that quantile; -1 disables adaptive routing and uses rift_recovery_margin."
        },
    )
    rift_exact_rank: bool = field(
        default=False,
        metadata={
            "help": "Use deterministic per-trajectory exact-rank routing instead of quantile-threshold routing. "
            "For quantile q and n eligible candidates, routes exactly ceil((1-q)*n) candidates."
        },
    )
    rift_routing_score: str = field(
        default="future_recovery",
        metadata={
            "help": "Score used to rank the fixed RIFT candidate set: future_recovery or ad_risk. "
            "ad_risk is P_student(y)*(logP_student(y)-logP_privileged(y)) clipped at zero."
        },
    )
    rift_require_full_window: bool = field(
        default=False,
        metadata={
            "help": "Restrict adaptive candidates to positions with the complete future recovery window. "
            "Incomplete-window candidates are retained only for the near-EOS audit."
        },
    )
    rift_hard_entropy_quantile: float = field(
        default=-1.0,
        metadata={
            "help": "Entropy quantile defining a separate uncertainty band. When enabled, candidate forks in "
            "that band use rift_hard_recovery_quantile while other candidates use rift_recovery_quantile. "
            "-1 disables uncertainty-banded routing."
        },
    )
    rift_hard_recovery_quantile: float = field(
        default=-1.0,
        metadata={
            "help": "Recovery-score quantile used in the separate uncertainty band. Must be enabled together "
            "with rift_hard_entropy_quantile; either stricter or looser than rift_recovery_quantile is valid."
        },
    )
    rift_groupwise_recovery_quantiles: bool = field(
        default=False,
        metadata={
            "help": "Compute regular and hard recovery quantiles within their own candidate bands instead of "
            "from the mixed candidate distribution. Requires uncertainty-banded routing."
        },
    )
    rift_fork_onset_routing: bool = field(
        default=False,
        metadata={
            "help": "Keep the adaptive recovery route count fixed but move routed tokens toward the start "
            "of local candidate-fork episodes. This is the budget-matched RIFT-FO ablation."
        },
    )
    rift_fork_onset_gap: int = field(
        default=4,
        metadata={
            "help": "Maximum candidate-token gap bridged inside one RIFT-FO fork episode."
        },
    )
    rift_reflection_safe_weighting: bool = field(
        default=False,
        metadata={
            "help": "Keep adaptive recovery routing unchanged and add a fractional unprivileged-teacher "
            "gate only at reflection markers or local candidate-fork onsets."
        },
    )
    rift_reflection_protection_weight: float = field(
        default=0.25,
        metadata={
            "help": "Fractional unprivileged-teacher interpolation at RIFT-RS protected tokens."
        },
    )
    rift_asymmetric_soft_clamp: bool = field(
        default=False,
        metadata={
            "help": "Keep recovery-q25 routing and targets unchanged, but softly clamp only extreme "
            "privileged JSD at unrouted reflection markers or candidate-fork onsets where the "
            "privileged teacher suppresses the sampled token."
        },
    )
    rift_soft_clamp_multiplier: float = field(
        default=3.0,
        metadata={
            "help": "Detached batch-mean multiplier for RIFT-ASC. The adaptive threshold is also "
            "bounded above by jsd_token_clip when a legacy hard clip is configured."
        },
    )
    rift_asymmetric_log_compression: bool = field(
        default=False,
        metadata={
            "help": "Keep recovery-q25 routing and targets unchanged, but apply detached log-scale "
            "compression to the legacy privileged JSD only at unrouted reflection/fork boundaries "
            "where the privileged teacher suppresses the sampled token."
        },
    )
    rift_base_persistence_routing: bool = field(
        default=False,
        metadata={
            "help": "Keep the q25 route count fixed but select branches where the unprivileged teacher "
            "continues to support the sampled trajectory over a local future window."
        },
    )
    rift_base_persistence_window: int = field(
        default=4,
        metadata={
            "help": "Number of future sampled tokens used to measure unprivileged-teacher support in RIFT-BSP."
        },
    )
    rift_base_persistence_min_gain: int = field(
        default=0,
        metadata={
            "help": "Minimum incoming-minus-outgoing continuation-support count required for an exact-budget "
            "RIFT-ASG swap. Zero preserves full-pool RIFT-BSP reranking."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )

    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    dataset_path: str = field(
        default="data/openthoughts_math_30k",
        metadata={
            "help": "Path to local dataset. Can be a JSON file, JSONL file, or directory containing Arrow/Parquet files. "
        },
    )
    task_type: str = field(
        default="math",
        metadata={
            "help": "Task type for prompt template selection. "
            "Options: 'math' (math reasoning with boxed answers), 'coding' (code generation). "
            "Default: 'math'"
        },
    )
    use_renio: bool = field(
        default=False,
        metadata={
            "help": "Enable ReNIO sample weighting (fixed-threshold S/T log-ratio filtering). "
            "When disabled, uniform sample weights are used."
        },
    )
    imp_token_threshold: float = field(
        default=0.2,
        metadata={
            "help": "Important token threshold (percentage). "
            "Used to select top-k% tokens for computing sample weights. "
            "Default: 0.3 (top 30% tokens)"
        },
    )
    kd_clamp: float = field(
        default=1.0,
        metadata={
            "help": "Clamp value for log-ratio. "
            "Log-ratio values will be clamped to this maximum value. "
            "Default: 2.0"
        },
    )
    weight_norm_type: str = field(
        default="batch_mean",
        metadata={
            "help": "Weight normalization type. "
            "Options: 'batch_mean' (normalize per batch), "
            "'ema' (normalize using global EMA statistics), "
            "'none' (no normalization), "
            "'clamp' (clamp then normalize). "
            "Default: 'batch_mean'"
        },
    )
    kd_sgo_tem: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for SGO weight computation. "
            "Used to soften the log-ratio before exp. "
            "Default: 1.0"
        },
    )
    use_entropy_gating: bool = field(
        default=False,
        metadata={
            "help": "Enable EAMS-OPSD dynamic token weighting from teacher predictive entropy. "
            "In inverse mode, low-entropy teacher positions receive larger distillation weights."
        },
    )
    entropy_gate_mode: str = field(
        default="inverse",
        metadata={
            "help": "Entropy gate mode. 'inverse' upweights low teacher entropy milestones; "
            "'direct' upweights high teacher entropy positions for ablations."
        },
    )
    entropy_gate_min: float = field(
        default=0.25,
        metadata={
            "help": "Minimum token-level entropy gate value before/after normalization."
        },
    )
    entropy_gate_max: float = field(
        default=2.0,
        metadata={
            "help": "Maximum token-level entropy gate value before/after normalization."
        },
    )
    entropy_gate_power: float = field(
        default=1.0,
        metadata={
            "help": "Power applied to normalized entropy salience before mapping to the gate range."
        },
    )
    entropy_gate_normalize: bool = field(
        default=True,
        metadata={
            "help": "Normalize entropy gates to mean 1 over valid tokens in the batch to preserve loss scale."
        },
    )
    entropy_gate_schedule: str = field(
        default="constant",
        metadata={
            "help": "Schedule for blending entropy gate back to vanilla OPSD. "
            "Options: constant, linear_decay, cosine_decay, phase_off."
        },
    )
    entropy_gate_schedule_start: float = field(
        default=0.0,
        metadata={
            "help": "Training progress ratio where entropy-gate decay starts. 0.0 means decay starts immediately."
        },
    )
    entropy_gate_schedule_end: float = field(
        default=1.0,
        metadata={
            "help": "Training progress ratio where entropy-gate decay reaches vanilla OPSD."
        },
    )
    use_repr_aux: bool = field(
        default=False,
        metadata={
            "help": "Enable low-weight hidden transition representation auxiliary on top of full OPSD."
        },
    )
    repr_aux_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight for hidden transition representation auxiliary. Recommended smoke values: 0.01 or 0.03."
        },
    )
    repr_aux_position_count: int = field(
        default=128,
        metadata={
            "help": "Maximum uniformly sampled completion transition positions per sample for representation auxiliary."
        },
    )
    repr_aux_layer_fraction: float = field(
        default=0.25,
        metadata={
            "help": "Fraction of final transformer layers used for representation auxiliary."
        },
    )
    repr_aux_eps: float = field(
        default=1e-6,
        metadata={
            "help": "Epsilon for cosine similarity in representation auxiliary."
        },
    )
    trajectory_selection_rollouts: int = field(
        default=1,
        metadata={
            "help": "Number of on-policy student rollouts sampled per prompt before trajectory selection. "
            "Use 1 to preserve vanilla OPSD generation."
        },
    )
    trajectory_selection_mode: str = field(
        default="none",
        metadata={
            "help": "Trajectory selector. 'gold_consensus' ranks candidates by exact final-answer correctness, "
            "then answer consensus. Requires trajectory_selection_rollouts > 1."
        },
    )
    use_regap: bool = field(
        default=False,
        metadata={
            "help": "Enable ReGap-OPSD Lite branch-value distillation over TopK student/teacher candidate tokens."
        },
    )
    regap_mode: str = field(
        default="replace",
        metadata={
            "help": "ReGap loss mode: 'replace' uses sparse ReGap as the objective; "
            "'additive' keeps full-token OPSD and adds ReGap terms."
        },
    )
    regap_branch_weight: float = field(
        default=1.0,
        metadata={
            "help": "Loss weight on ReGap branch CE. In additive mode, use 0.1-0.3 for a light regularizer."
        },
    )
    regap_top_k: int = field(
        default=2,
        metadata={
            "help": "Number of student TopK and teacher TopK candidates used at each ReGap decision point."
        },
    )
    regap_tau: float = field(
        default=0.5,
        metadata={
            "help": "Temperature for q(a)=softmax((V_T-V_S)/tau) in ReGap-Lite."
        },
    )
    regap_lambda_pi: float = field(
        default=1.0,
        metadata={
            "help": "Weight on prediction-interval KL/JSD gated by teacher-value agreement."
        },
    )
    regap_eta_dead: float = field(
        default=0.05,
        metadata={
            "help": "Weight on dead-branch unlikelihood loss."
        },
    )
    regap_dead_teacher_threshold: float = field(
        default=0.10,
        metadata={
            "help": "Teacher branch-value threshold below which a branch is treated as dead."
        },
    )
    regap_dead_student_threshold: float = field(
        default=0.30,
        metadata={
            "help": "Student branch-value threshold above which a dead branch is actively suppressed."
        },
    )
    regap_decision_ratio: float = field(
        default=0.25,
        metadata={
            "help": "Fraction of valid generated tokens kept as ReGap decision points per sample."
        },
    )
    regap_min_decisions: int = field(
        default=4,
        metadata={
            "help": "Minimum number of ReGap decision points per sample when enough valid tokens exist."
        },
    )
    regap_gap_weight: float = field(
        default=1.0,
        metadata={
            "help": "Decision-point score weight for positive rescue gap max_a max(V_T-V_S, 0)."
        },
    )
    regap_disagreement_weight: float = field(
        default=1.0,
        metadata={
            "help": "Decision-point score weight for candidate-set teacher/student disagreement."
        },
    )
    regap_student_entropy_weight: float = field(
        default=0.0,
        metadata={
            "help": "Optional decision-point score weight for normalized student entropy. "
            "Default 0 avoids an extra full-vocab entropy pass."
        },
    )
    regap_weight_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Sample-weight coefficient for positive rescue score in regap_mode='weighted'."
        },
    )
    regap_weight_beta: float = field(
        default=1.0,
        metadata={
            "help": "Sample-weight coefficient for suspicious/negative rescue score in regap_mode='weighted'."
        },
    )
    regap_weight_min: float = field(
        default=0.7,
        metadata={
            "help": "Minimum clipped sample weight in regap_mode='weighted'."
        },
    )
    regap_weight_max: float = field(
        default=1.3,
        metadata={
            "help": "Maximum clipped sample weight in regap_mode='weighted'."
        },
    )


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    
    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )
    
    
    model_name = model_args.model_name_or_path.split("/")[-1]

    # Create concise run name
    full_wandb_run_config = (
        script_args.dataset_path.split("/")[-1] + "_"
        f"{model_name}_"
        f"renio{script_args.use_renio}"
        f"eams{script_args.use_entropy_gating}_"
        f"rift{script_args.use_rift_routing}_"
        f"regap{script_args.use_regap}_"
        f"type{script_args.kd_type}_"
        f"lr{lr_str}_"
        f"bs{effective_batch_size}_"
        f"tok{training_args.max_completion_length}"
        f"clip{script_args.kd_clamp}_"
        f"tem{script_args.kd_sgo_tem}_"
        f"thresh{script_args.imp_token_threshold}"
        f"egate{script_args.entropy_gate_mode}"
        f"emin{script_args.entropy_gate_min}"
        f"emax{script_args.entropy_gate_max}"
        f"esch{script_args.entropy_gate_schedule}"
        f"es{script_args.entropy_gate_schedule_start}"
        f"ee{script_args.entropy_gate_schedule_end}"
        f"repr{script_args.use_repr_aux}"
        f"rw{script_args.repr_aux_weight}"
        f"rpos{script_args.repr_aux_position_count}"
        f"rlfrac{script_args.repr_aux_layer_fraction}"
        f"rmode{script_args.regap_mode}"
        f"rbranchw{script_args.regap_branch_weight}"
        f"rtop{script_args.regap_top_k}"
        f"rtau{script_args.regap_tau}"
        f"rdec{script_args.regap_decision_ratio}"
        f"rgapw{script_args.regap_gap_weight}"
        f"rdisw{script_args.regap_disagreement_weight}"
        f"rentw{script_args.regap_student_entropy_weight}"
        f"rwalpha{script_args.regap_weight_alpha}"
        f"rwbeta{script_args.regap_weight_beta}"
        f"rwmin{script_args.regap_weight_min}"
        f"rwmax{script_args.regap_weight_max}"
        f"beta{training_args.beta}"
    )

    # Add fixed_teacher to wandb name if enabled
    if script_args.fixed_teacher:
        full_wandb_run_config += "_fixteach"
    
    from pathlib import Path

    output_suffix_parts = [
        f"renio{script_args.use_renio}",
        f"eams{script_args.use_entropy_gating}",
        f"rift{script_args.use_rift_routing}",
        f"regap{script_args.use_regap}",
        f"repr{script_args.use_repr_aux}",
        f"jclip{script_args.jsd_token_clip}",
    ]
    if script_args.use_entropy_gating:
        output_suffix_parts.extend(
            [
                f"eg{script_args.entropy_gate_mode}",
                f"emin{script_args.entropy_gate_min}",
                f"emax{script_args.entropy_gate_max}",
                f"esch{script_args.entropy_gate_schedule}",
            ]
        )
    if script_args.use_rift_routing:
        output_suffix_parts.extend(
            [
                f"pm{script_args.rift_sign_margin}",
                f"pq{script_args.rift_entropy_quantile}",
                f"pw{script_args.rift_route_weight}",
            ]
        )
        if script_args.rift_recovery_window > 0:
            output_suffix_parts.append(f"rcw{script_args.rift_recovery_window}")
            if script_args.rift_recovery_quantile >= 0:
                output_suffix_parts.append(f"rcq{script_args.rift_recovery_quantile}")
                if script_args.rift_exact_rank:
                    output_suffix_parts.append("exactrank")
                if script_args.rift_require_full_window:
                    output_suffix_parts.append("fullwindow")
                if script_args.rift_routing_score != "future_recovery":
                    output_suffix_parts.append(
                        f"rscore{script_args.rift_routing_score}"
                    )
                if script_args.rift_hard_entropy_quantile >= 0:
                    output_suffix_parts.extend(
                        [
                            f"heq{script_args.rift_hard_entropy_quantile}",
                            f"hrq{script_args.rift_hard_recovery_quantile}",
                        ]
                    )
                    if script_args.rift_groupwise_recovery_quantiles:
                        output_suffix_parts.append("grq")
            else:
                output_suffix_parts.append(f"rcm{script_args.rift_recovery_margin}")
        if script_args.rift_fork_onset_routing:
            output_suffix_parts.append(f"fog{script_args.rift_fork_onset_gap}")
        if script_args.rift_reflection_safe_weighting:
            output_suffix_parts.extend(
                [
                    f"rsw{script_args.rift_reflection_protection_weight}",
                    f"rsg{script_args.rift_fork_onset_gap}",
                ]
            )
        if script_args.rift_asymmetric_soft_clamp:
            output_suffix_parts.extend(
                [
                    f"asc{script_args.rift_soft_clamp_multiplier}",
                    f"ascg{script_args.rift_fork_onset_gap}",
                ]
            )
        if script_args.rift_asymmetric_log_compression:
            output_suffix_parts.extend(["alc", f"alcg{script_args.rift_fork_onset_gap}"])
        if script_args.rift_base_persistence_routing:
            output_suffix_parts.append(f"bpsw{script_args.rift_base_persistence_window}")
            if script_args.rift_base_persistence_min_gain > 0:
                output_suffix_parts.append(f"asgg{script_args.rift_base_persistence_min_gain}")
    if script_args.use_repr_aux:
        output_suffix_parts.extend(
            [
                f"rw{script_args.repr_aux_weight}",
                f"rp{script_args.repr_aux_position_count}",
                f"rlf{script_args.repr_aux_layer_fraction}",
            ]
        )
    if script_args.trajectory_selection_rollouts > 1:
        output_suffix_parts.extend(
            [
                f"tsn{script_args.trajectory_selection_rollouts}",
                f"tsm{script_args.trajectory_selection_mode}",
            ]
        )
    if script_args.use_regap:
        output_suffix_parts.extend(
            [
                f"rmode{script_args.regap_mode}",
                f"rbw{script_args.regap_branch_weight}",
                f"rtop{script_args.regap_top_k}",
                f"rdec{script_args.regap_decision_ratio}",
            ]
        )

    training_args.output_dir = str(
        Path(training_args.output_dir)
        / model_name
        / script_args.dataset_path.split("/")[-1]
        / script_args.kd_type
        / script_args.run_config
        / "_".join(output_suffix_parts)
    )


    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Validate fixed_teacher and teacher_path are mutually exclusive
    if script_args.fixed_teacher and script_args.teacher_path is not None:
        raise ValueError(
            "fixed_teacher=True and teacher_path are mutually exclusive. "
            "Use either a separate teacher model or fixed teacher (self-distillation), not both."
        )

    # Validate fixed_teacher argument
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )

    if script_args.use_rift_routing:
        if not script_args.fixed_teacher:
            raise ValueError("use_rift_routing=True requires fixed_teacher=True.")
        if script_args.use_tinker_loss:
            raise ValueError("use_rift_routing=True requires the full-vocab JSD loss (use_tinker_loss=False).")
        if script_args.top_k_loss > 0:
            raise ValueError("use_rift_routing=True requires top_k_loss=0.")
        if script_args.use_entropy_gating or script_args.use_regap:
            raise ValueError("RIFT routing must be run as an isolated ablation without entropy gating or ReGap.")
        if script_args.rift_recovery_window < 0:
            raise ValueError("rift_recovery_window must be non-negative.")
        if script_args.rift_recovery_margin < 0:
            raise ValueError("rift_recovery_margin must be non-negative.")
        if script_args.rift_recovery_quantile != -1 and not 0 <= script_args.rift_recovery_quantile <= 1:
            raise ValueError("rift_recovery_quantile must be -1 or in [0, 1].")
        if script_args.rift_routing_score not in {"future_recovery", "ad_risk"}:
            raise ValueError(
                "rift_routing_score must be future_recovery or ad_risk."
            )
        if script_args.rift_exact_rank:
            if script_args.rift_recovery_quantile < 0:
                raise ValueError("rift_exact_rank requires rift_recovery_quantile in [0, 1].")
            if script_args.rift_recovery_window <= 0:
                raise ValueError("rift_exact_rank requires rift_recovery_window > 0.")
            if (
                script_args.rift_hard_entropy_quantile >= 0
                or script_args.rift_groupwise_recovery_quantiles
            ):
                raise ValueError(
                    "rift_exact_rank cannot be combined with uncertainty-banded routing."
                )
        if script_args.rift_routing_score == "ad_risk" and not script_args.rift_exact_rank:
            raise ValueError("ad_risk routing requires rift_exact_rank.")
        if script_args.rift_require_full_window and script_args.rift_recovery_window <= 0:
            raise ValueError(
                "rift_require_full_window requires rift_recovery_window > 0."
            )
        hard_dq_values = (
            script_args.rift_hard_entropy_quantile,
            script_args.rift_hard_recovery_quantile,
        )
        if (hard_dq_values[0] == -1) != (hard_dq_values[1] == -1):
            raise ValueError("RIFT-DQ hard entropy and recovery quantiles must be enabled together.")
        if hard_dq_values[0] != -1:
            if not all(0 <= value <= 1 for value in hard_dq_values):
                raise ValueError("RIFT-DQ hard entropy and recovery quantiles must be in [0, 1].")
            if script_args.rift_recovery_quantile < 0:
                raise ValueError("RIFT-DQ requires rift_recovery_quantile in [0, 1].")
            if script_args.rift_recovery_window <= 0:
                raise ValueError("RIFT-DQ requires rift_recovery_window > 0.")
            if script_args.rift_hard_entropy_quantile < script_args.rift_entropy_quantile:
                raise ValueError("rift_hard_entropy_quantile must be >= rift_entropy_quantile.")
        elif script_args.rift_groupwise_recovery_quantiles:
            raise ValueError("rift_groupwise_recovery_quantiles requires uncertainty-banded routing.")
        if script_args.rift_fork_onset_gap < 1:
            raise ValueError("rift_fork_onset_gap must be positive.")
        if script_args.rift_fork_onset_routing:
            if script_args.rift_recovery_window <= 0 or script_args.rift_recovery_quantile < 0:
                raise ValueError(
                    "RIFT-FO requires adaptive recovery routing with a positive future window."
                )
            if script_args.rift_hard_entropy_quantile >= 0 or script_args.rift_groupwise_recovery_quantiles:
                raise ValueError("RIFT-FO cannot be combined with uncertainty-banded routing.")
        if script_args.rift_reflection_safe_weighting and not 0 < script_args.rift_reflection_protection_weight < 1:
            raise ValueError("rift_reflection_protection_weight must be in (0, 1).")
        if script_args.rift_reflection_safe_weighting:
            if script_args.rift_recovery_window <= 0 or script_args.rift_recovery_quantile < 0:
                raise ValueError(
                    "RIFT-RS requires adaptive recovery routing with a positive future window."
                )
            if script_args.rift_fork_onset_routing:
                raise ValueError("RIFT-RS cannot be combined with RIFT-FO.")
            if script_args.rift_hard_entropy_quantile >= 0 or script_args.rift_groupwise_recovery_quantiles:
                raise ValueError("RIFT-RS cannot be combined with uncertainty-banded routing.")
        if script_args.rift_soft_clamp_multiplier <= 0:
            raise ValueError("rift_soft_clamp_multiplier must be positive.")
        if script_args.rift_asymmetric_soft_clamp:
            if script_args.rift_recovery_window <= 0 or script_args.rift_recovery_quantile < 0:
                raise ValueError(
                    "RIFT-ASC requires adaptive recovery routing with a positive future window."
                )
            if script_args.rift_reflection_safe_weighting or script_args.rift_fork_onset_routing:
                raise ValueError("RIFT-ASC cannot be combined with RIFT-RS or RIFT-FO.")
            if script_args.rift_hard_entropy_quantile >= 0 or script_args.rift_groupwise_recovery_quantiles:
                raise ValueError("RIFT-ASC cannot be combined with uncertainty-banded routing.")
        if script_args.rift_asymmetric_log_compression:
            if script_args.rift_recovery_window <= 0 or script_args.rift_recovery_quantile < 0:
                raise ValueError(
                    "RIFT-ALC requires adaptive recovery routing with a positive future window."
                )
            if (
                script_args.rift_reflection_safe_weighting
                or script_args.rift_fork_onset_routing
                or script_args.rift_asymmetric_soft_clamp
            ):
                raise ValueError("RIFT-ALC cannot be combined with RIFT-RS, RIFT-FO, or RIFT-ASC.")
            if script_args.rift_hard_entropy_quantile >= 0 or script_args.rift_groupwise_recovery_quantiles:
                raise ValueError("RIFT-ALC cannot be combined with uncertainty-banded routing.")
        if script_args.rift_base_persistence_window < 1:
            raise ValueError("rift_base_persistence_window must be positive.")
        if script_args.rift_base_persistence_min_gain < 0:
            raise ValueError("rift_base_persistence_min_gain must be non-negative.")
        if script_args.rift_base_persistence_min_gain > script_args.rift_base_persistence_window:
            raise ValueError("rift_base_persistence_min_gain cannot exceed the support window.")
        if script_args.rift_base_persistence_min_gain > 0 and not script_args.rift_base_persistence_routing:
            raise ValueError("RIFT-ASG minimum support gain requires base-persistence routing.")
        if script_args.rift_base_persistence_routing:
            if script_args.rift_recovery_window <= 0 or script_args.rift_recovery_quantile < 0:
                raise ValueError(
                    "RIFT-BSP requires adaptive recovery routing with a positive future window."
                )
            if script_args.rift_route_weight != 1.0:
                raise ValueError("RIFT-BSP requires a hard route weight of 1.0.")
            if script_args.rift_hard_entropy_quantile >= 0 or script_args.rift_groupwise_recovery_quantiles:
                raise ValueError("RIFT-BSP cannot be combined with uncertainty-banded routing.")
            if (
                script_args.rift_fork_onset_routing
                or script_args.rift_reflection_safe_weighting
                or script_args.rift_asymmetric_soft_clamp
                or script_args.rift_asymmetric_log_compression
            ):
                raise ValueError("RIFT-BSP replaces q25 selection and cannot combine with FO, RS, or ASC.")

    # Validate teacher_lora_path requires teacher_path
    if script_args.teacher_lora_path is not None and script_args.teacher_path is None:
        raise ValueError(
            "teacher_lora_path requires teacher_path to specify the base model."
        )

    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "student_model_name": model_args.model_name_or_path,
                "teacher_model_name": script_args.teacher_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "use_tinker_loss": script_args.use_tinker_loss,
                "fixed_teacher": script_args.fixed_teacher,
                "use_rift_routing": script_args.use_rift_routing,
                "rift_sign_margin": script_args.rift_sign_margin,
                "rift_entropy_quantile": script_args.rift_entropy_quantile,
                "rift_route_weight": script_args.rift_route_weight,
                "rift_recovery_window": script_args.rift_recovery_window,
                "rift_recovery_margin": script_args.rift_recovery_margin,
                "rift_recovery_quantile": script_args.rift_recovery_quantile,
                "rift_exact_rank": script_args.rift_exact_rank,
                "rift_routing_score": script_args.rift_routing_score,
                "rift_require_full_window": script_args.rift_require_full_window,
                "rift_hard_entropy_quantile": script_args.rift_hard_entropy_quantile,
                "rift_hard_recovery_quantile": script_args.rift_hard_recovery_quantile,
                "rift_groupwise_recovery_quantiles": script_args.rift_groupwise_recovery_quantiles,
                "rift_fork_onset_routing": script_args.rift_fork_onset_routing,
                "rift_fork_onset_gap": script_args.rift_fork_onset_gap,
                "rift_reflection_safe_weighting": script_args.rift_reflection_safe_weighting,
                "rift_reflection_protection_weight": script_args.rift_reflection_protection_weight,
                "rift_asymmetric_soft_clamp": script_args.rift_asymmetric_soft_clamp,
                "rift_soft_clamp_multiplier": script_args.rift_soft_clamp_multiplier,
                "rift_asymmetric_log_compression": script_args.rift_asymmetric_log_compression,
                "rift_base_persistence_routing": script_args.rift_base_persistence_routing,
                "rift_base_persistence_window": script_args.rift_base_persistence_window,
                "rift_base_persistence_min_gain": script_args.rift_base_persistence_min_gain,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
                "use_renio": script_args.use_renio,
                "imp_token_threshold": script_args.imp_token_threshold,
                "kd_clamp": script_args.kd_clamp,
                "weight_norm_type": script_args.weight_norm_type,
                "kd_sgo_tem": script_args.kd_sgo_tem,
                "use_entropy_gating": script_args.use_entropy_gating,
                "entropy_gate_mode": script_args.entropy_gate_mode,
                "entropy_gate_min": script_args.entropy_gate_min,
                "entropy_gate_max": script_args.entropy_gate_max,
                "entropy_gate_power": script_args.entropy_gate_power,
                "entropy_gate_normalize": script_args.entropy_gate_normalize,
                "entropy_gate_schedule": script_args.entropy_gate_schedule,
                "entropy_gate_schedule_start": script_args.entropy_gate_schedule_start,
                "entropy_gate_schedule_end": script_args.entropy_gate_schedule_end,
                "use_repr_aux": script_args.use_repr_aux,
                "repr_aux_weight": script_args.repr_aux_weight,
                "repr_aux_position_count": script_args.repr_aux_position_count,
                "repr_aux_layer_fraction": script_args.repr_aux_layer_fraction,
                "repr_aux_eps": script_args.repr_aux_eps,
                "trajectory_selection_rollouts": script_args.trajectory_selection_rollouts,
                "trajectory_selection_mode": script_args.trajectory_selection_mode,
                "use_regap": script_args.use_regap,
                "regap_mode": script_args.regap_mode,
                "regap_branch_weight": script_args.regap_branch_weight,
                "regap_top_k": script_args.regap_top_k,
                "regap_tau": script_args.regap_tau,
                "regap_lambda_pi": script_args.regap_lambda_pi,
                "regap_eta_dead": script_args.regap_eta_dead,
                "regap_dead_teacher_threshold": script_args.regap_dead_teacher_threshold,
                "regap_dead_student_threshold": script_args.regap_dead_student_threshold,
                "regap_decision_ratio": script_args.regap_decision_ratio,
                "regap_min_decisions": script_args.regap_min_decisions,
                "regap_gap_weight": script_args.regap_gap_weight,
                "regap_disagreement_weight": script_args.regap_disagreement_weight,
                "regap_student_entropy_weight": script_args.regap_student_entropy_weight,
                "regap_weight_alpha": script_args.regap_weight_alpha,
                "regap_weight_beta": script_args.regap_weight_beta,
                "regap_weight_min": script_args.regap_weight_min,
                "regap_weight_max": script_args.regap_weight_max,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        model_kwargs["device_map"] = None
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        if world_size > 1:
            print("Disabling k-bit quantization for multi-process training; loading bf16 model per rank.")
        else:
            # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
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
    # Load Teacher Model (if specified)
    ################
    teacher_model = None
    if script_args.teacher_path is not None:
        print(f"\n{'='*80}")
        print(f"Loading teacher model from: {script_args.teacher_path}")
        if script_args.teacher_lora_path:
            print(f"  with LoRA adapter: {script_args.teacher_lora_path}")
        print(f"{'='*80}\n")

        from transformers import AutoModelForCausalLM

        # Use same dtype as student model
        teacher_dtype = model_dtype

        # Prepare teacher model kwargs
        teacher_model_kwargs = dict(
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation or "flash_attention_2",
            torch_dtype=teacher_dtype,
            use_cache=False,  # Disable cache for teacher during training
        )

        # Load base model
        teacher_model = AutoModelForCausalLM.from_pretrained(
            script_args.teacher_path,
            **teacher_model_kwargs,
        )

        # Apply LoRA adapter if specified
        if script_args.teacher_lora_path is not None:
            from peft import PeftModel

            print(f"Applying LoRA adapter from: {script_args.teacher_lora_path}")
            teacher_model = PeftModel.from_pretrained(
                teacher_model, script_args.teacher_lora_path
            )
            # Merge LoRA into base weights to avoid adapter overhead during forward
            teacher_model = teacher_model.merge_and_unload()
            print(f"LoRA adapter merged into base model")

        # Disable dropout in teacher model
        if training_args.disable_dropout:
            from trl.trainer.utils import disable_dropout_in_model
            disable_dropout_in_model(teacher_model)

        print(f"\n{'='*80}")
        print(f"Teacher model loaded successfully")
        print(f"  Base model: {script_args.teacher_path}")
        if script_args.teacher_lora_path:
            print(f"  LoRA adapter: {script_args.teacher_lora_path} (merged)")
        print(f"  Parameters: {teacher_model.num_parameters():,}")
        print(f"  Dtype: {teacher_dtype}")
        print(f"{'='*80}\n")

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    training_args.presence_penalty = script_args.presence_penalty

    # Skip SFTTrainer's default dataset tokenization — OPSD uses its own data collator
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset from local path or HuggingFace hub
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
        dataset = load_dataset("PATH_TO_DATASET_URL")
        dataset = dataset["train"]

    train_dataset = dataset if isinstance(dataset, Dataset) else dataset["train"]

    # SFTTrainer applies PEFT before the base Trainer has guaranteed that every
    # RNG is reset. Seed explicitly here so LoRA initialization is identical
    # across matched-OPSD and RIFT-disabled processes.
    set_seed(training_args.seed)
    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        use_rift_routing=script_args.use_rift_routing,
        rift_sign_margin=script_args.rift_sign_margin,
        rift_entropy_quantile=script_args.rift_entropy_quantile,
        rift_route_weight=script_args.rift_route_weight,
        rift_recovery_window=script_args.rift_recovery_window,
        rift_recovery_margin=script_args.rift_recovery_margin,
        rift_recovery_quantile=script_args.rift_recovery_quantile,
        rift_exact_rank=script_args.rift_exact_rank,
        rift_routing_score=script_args.rift_routing_score,
        rift_require_full_window=script_args.rift_require_full_window,
        rift_hard_entropy_quantile=script_args.rift_hard_entropy_quantile,
        rift_hard_recovery_quantile=script_args.rift_hard_recovery_quantile,
        rift_groupwise_recovery_quantiles=script_args.rift_groupwise_recovery_quantiles,
        rift_fork_onset_routing=script_args.rift_fork_onset_routing,
        rift_fork_onset_gap=script_args.rift_fork_onset_gap,
        rift_reflection_safe_weighting=script_args.rift_reflection_safe_weighting,
        rift_reflection_protection_weight=script_args.rift_reflection_protection_weight,
        rift_asymmetric_soft_clamp=script_args.rift_asymmetric_soft_clamp,
        rift_soft_clamp_multiplier=script_args.rift_soft_clamp_multiplier,
        rift_asymmetric_log_compression=script_args.rift_asymmetric_log_compression,
        rift_base_persistence_routing=script_args.rift_base_persistence_routing,
        rift_base_persistence_window=script_args.rift_base_persistence_window,
        rift_base_persistence_min_gain=script_args.rift_base_persistence_min_gain,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        teacher_model=teacher_model,
        use_renio=script_args.use_renio,
        imp_token_threshold=script_args.imp_token_threshold,
        kd_clamp=script_args.kd_clamp,
        weight_norm_type=script_args.weight_norm_type,
        kd_sgo_tem=script_args.kd_sgo_tem,
        use_entropy_gating=script_args.use_entropy_gating,
        entropy_gate_mode=script_args.entropy_gate_mode,
        entropy_gate_min=script_args.entropy_gate_min,
        entropy_gate_max=script_args.entropy_gate_max,
        entropy_gate_power=script_args.entropy_gate_power,
        entropy_gate_normalize=script_args.entropy_gate_normalize,
        entropy_gate_schedule=script_args.entropy_gate_schedule,
        entropy_gate_schedule_start=script_args.entropy_gate_schedule_start,
        entropy_gate_schedule_end=script_args.entropy_gate_schedule_end,
        use_repr_aux=script_args.use_repr_aux,
        repr_aux_weight=script_args.repr_aux_weight,
        repr_aux_position_count=script_args.repr_aux_position_count,
        repr_aux_layer_fraction=script_args.repr_aux_layer_fraction,
        repr_aux_eps=script_args.repr_aux_eps,
        trajectory_selection_rollouts=script_args.trajectory_selection_rollouts,
        trajectory_selection_mode=script_args.trajectory_selection_mode,
        use_regap=script_args.use_regap,
        regap_mode=script_args.regap_mode,
        regap_branch_weight=script_args.regap_branch_weight,
        regap_top_k=script_args.regap_top_k,
        regap_tau=script_args.regap_tau,
        regap_lambda_pi=script_args.regap_lambda_pi,
        regap_eta_dead=script_args.regap_eta_dead,
        regap_dead_teacher_threshold=script_args.regap_dead_teacher_threshold,
        regap_dead_student_threshold=script_args.regap_dead_student_threshold,
        regap_decision_ratio=script_args.regap_decision_ratio,
        regap_min_decisions=script_args.regap_min_decisions,
        regap_gap_weight=script_args.regap_gap_weight,
        regap_disagreement_weight=script_args.regap_disagreement_weight,
        regap_student_entropy_weight=script_args.regap_student_entropy_weight,
        regap_weight_alpha=script_args.regap_weight_alpha,
        regap_weight_beta=script_args.regap_weight_beta,
        regap_weight_min=script_args.regap_weight_min,
        regap_weight_max=script_args.regap_weight_max,
        task_type=script_args.task_type,
        # dataset_kwargs={"skip_prepare_dataset": True},  # Skip dataset preparation
    )

    trainer.train(
        resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None)
    )

    trainer.save_model(training_args.output_dir)
