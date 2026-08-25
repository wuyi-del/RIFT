# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import random
import re
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from accelerate import PartialState
from accelerate.utils import DistributedType, broadcast_object_list, gather_object, is_peft_model
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.utils import (
    is_flash_attn_2_available,
    is_liger_kernel_available,
    is_peft_available,
    is_rich_available,
)

from trl.data_utils import is_conversational, maybe_convert_to_chatml, pack_dataset, truncate_dataset
from trl.extras.profiling import profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available
from trl.models import prepare_deepspeed
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import (
    DataCollatorForChatML,
    disable_dropout_in_model,
    empty_cache,
    ensure_master_addr_port,
    pad,
)
from trl.experimental.gold.gold_config import GOLDConfig
from data_collator import SelfDistillationDataCollator
from rift_p0_routing import ad_risk_score, mask_checksum53, routing_audit


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


class EMAUpdateCallback(TrainerCallback):
    """Update EMA teacher weights after each optimizer step."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Only update when the optimizer actually stepped (end of a gradient accumulation cycle)
        if self.trainer.use_ema_teacher and self.trainer.accelerator.sync_gradients:
            self.trainer._update_ema()


class GOLDVLLMSyncCallback(TrainerCallback):
    """Sync the model weights to vLLM after training steps when it's safe to do so."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Sync weights after training step when DeepSpeed is stable."""
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
        ):
            # Check if this is a step where gradients are synchronized
            # This happens at the end of gradient accumulation cycles
            if (
                hasattr(self.trainer.accelerator, "sync_gradients")
                and self.trainer.accelerator.sync_gradients
            ):
                self.trainer._move_model_to_vllm()
                self.trainer._last_vllm_sync_step = state.global_step


class OPSDTrainer(SFTTrainer):
    _tag_names = ["trl", "opsd"]
    _name = "OPSD"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
        use_thinking_machines_loss: bool = False,
        fixed_teacher: bool = False,
        use_rift_routing: bool = False,
        rift_sign_margin: float = 0.05,
        rift_entropy_quantile: float = 0.75,
        rift_route_weight: float = 1.0,
        rift_recovery_window: int = 0,
        rift_recovery_margin: float = 0.0,
        rift_recovery_quantile: float = -1.0,
        rift_exact_rank: bool = False,
        rift_routing_score: str = "future_recovery",
        rift_require_full_window: bool = False,
        rift_hard_entropy_quantile: float = -1.0,
        rift_hard_recovery_quantile: float = -1.0,
        rift_groupwise_recovery_quantiles: bool = False,
        rift_fork_onset_routing: bool = False,
        rift_fork_onset_gap: int = 4,
        rift_reflection_safe_weighting: bool = False,
        rift_reflection_protection_weight: float = 0.25,
        rift_asymmetric_soft_clamp: bool = False,
        rift_soft_clamp_multiplier: float = 3.0,
        rift_asymmetric_log_compression: bool = False,
        rift_base_persistence_routing: bool = False,
        rift_base_persistence_window: int = 4,
        rift_base_persistence_min_gain: int = 0,
        reason_first: bool = False,
        top_k_loss: int | None = None,
        jsd_token_clip: float | None = None,
        use_ema_teacher: bool = False,
        ema_decay: float = 0.999,
        use_renio: bool = False,
        imp_token_threshold: float = 0.3,
        kd_clamp: float = 2.0,
        weight_norm_type: str = "batch_mean",
        kd_sgo_tem: float = 1.0,
        use_entropy_gating: bool = False,
        entropy_gate_mode: str = "inverse",
        entropy_gate_min: float = 0.25,
        entropy_gate_max: float = 2.0,
        entropy_gate_power: float = 1.0,
        entropy_gate_normalize: bool = True,
        entropy_gate_schedule: str = "constant",
        entropy_gate_schedule_start: float = 0.0,
        entropy_gate_schedule_end: float = 1.0,
        use_repr_aux: bool = False,
        repr_aux_weight: float = 0.0,
        repr_aux_position_count: int = 128,
        repr_aux_layer_fraction: float = 0.25,
        repr_aux_eps: float = 1e-6,
        trajectory_selection_rollouts: int = 1,
        trajectory_selection_mode: str = "none",
        use_regap: bool = False,
        regap_mode: str = "replace",
        regap_branch_weight: float = 1.0,
        regap_top_k: int = 2,
        regap_tau: float = 0.5,
        regap_lambda_pi: float = 1.0,
        regap_eta_dead: float = 0.05,
        regap_dead_teacher_threshold: float = 0.10,
        regap_dead_student_threshold: float = 0.30,
        regap_decision_ratio: float = 0.25,
        regap_min_decisions: int = 4,
        regap_gap_weight: float = 1.0,
        regap_disagreement_weight: float = 1.0,
        regap_student_entropy_weight: float = 0.0,
        regap_weight_alpha: float = 1.0,
        regap_weight_beta: float = 1.0,
        regap_weight_min: float = 0.7,
        regap_weight_max: float = 1.3,
        teacher_model: PreTrainedModel | nn.Module | None = None,
        task_type: str = "math",
    ):
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = getattr(args, "student_model_revision", None)
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        # Custom data collator for self-distillation
        if data_collator is None:
            data_collator = SelfDistillationDataCollator(
                tokenizer=processing_class, max_length=args.max_length, reason_first=reason_first,
                task_type=task_type,
            )

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        self._cast_trainable_parameters_to_fp32()

        self.student_vocab_size = self.model.config.vocab_size
        self.model_safe_pad_token_id = self._resolve_model_safe_pad_token_id()

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd
        self.use_thinking_machines_loss = use_thinking_machines_loss
        self.fixed_teacher = fixed_teacher
        self.use_rift_routing = use_rift_routing
        self.rift_sign_margin = rift_sign_margin
        self.rift_entropy_quantile = rift_entropy_quantile
        self.rift_route_weight = rift_route_weight
        self.rift_recovery_window = rift_recovery_window
        self.rift_recovery_margin = rift_recovery_margin
        self.rift_recovery_quantile = rift_recovery_quantile
        self.rift_exact_rank = rift_exact_rank
        self.rift_routing_score = rift_routing_score
        self.rift_require_full_window = rift_require_full_window
        self.rift_hard_entropy_quantile = rift_hard_entropy_quantile
        self.rift_hard_recovery_quantile = rift_hard_recovery_quantile
        self.rift_groupwise_recovery_quantiles = rift_groupwise_recovery_quantiles
        self.rift_fork_onset_routing = rift_fork_onset_routing
        self.rift_fork_onset_gap = rift_fork_onset_gap
        self.rift_reflection_safe_weighting = rift_reflection_safe_weighting
        self.rift_reflection_protection_weight = rift_reflection_protection_weight
        self.rift_asymmetric_soft_clamp = rift_asymmetric_soft_clamp
        self.rift_soft_clamp_multiplier = rift_soft_clamp_multiplier
        self.rift_asymmetric_log_compression = rift_asymmetric_log_compression
        self.rift_base_persistence_routing = rift_base_persistence_routing
        self.rift_base_persistence_window = rift_base_persistence_window
        self.rift_base_persistence_min_gain = rift_base_persistence_min_gain
        self.rift_reflection_markers = (
            "wait",
            "hold on",
            "let me",
            "recalculate",
            "re-calculate",
            "recheck",
            "re-check",
            "reconsider",
            "rethink",
            "check again",
            "verify again",
            "another approach",
            "try again",
        )
        self.rift_reflection_token_sequences = []
        if (
            self.rift_reflection_safe_weighting
            or self.rift_asymmetric_soft_clamp
            or self.rift_asymmetric_log_compression
        ):
            if processing_class is None or not hasattr(processing_class, "encode"):
                raise ValueError("RIFT-RS/ASC requires a tokenizer with an encode method.")
            encoded_markers = set()
            for marker in self.rift_reflection_markers:
                for text in (marker, marker.capitalize(), f" {marker}", f" {marker.capitalize()}"):
                    token_ids = tuple(processing_class.encode(text, add_special_tokens=False))
                    if token_ids:
                        encoded_markers.add(token_ids)
            self.rift_reflection_token_sequences = sorted(
                encoded_markers, key=lambda item: (len(item), item)
            )
        self.reason_first = reason_first
        self.task_type = task_type
        self.top_k_loss = top_k_loss
        self.jsd_token_clip = jsd_token_clip
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self._ema_params = None  # lazily initialized on first optimizer step

        # Sample weighting parameters
        self.use_renio = use_renio
        self.imp_token_threshold = imp_token_threshold
        self.kd_clamp = kd_clamp
        self.weight_norm_type = weight_norm_type
        self.kd_sgo_tem = kd_sgo_tem
        self.ratio_type = "renio" if self.use_renio else "uniform"

        # EAMS-OPSD: Entropy-Aware Milestone Selection.
        # Low teacher entropy means the privileged teacher has a sharp next-token belief,
        # so that position is treated as a reasoning milestone and receives larger KL weight.
        self.use_entropy_gating = use_entropy_gating
        self.entropy_gate_mode = entropy_gate_mode
        self.entropy_gate_min = entropy_gate_min
        self.entropy_gate_max = entropy_gate_max
        self.entropy_gate_power = entropy_gate_power
        self.entropy_gate_normalize = entropy_gate_normalize
        self.entropy_gate_schedule = entropy_gate_schedule
        self.entropy_gate_schedule_start = entropy_gate_schedule_start
        self.entropy_gate_schedule_end = entropy_gate_schedule_end
        self.entropy_gate_eps = 1e-6
        self._last_entropy_gate_mix = 1.0

        # Representation auxiliary: low-interference hidden transition matching.
        # This keeps full-token OPSD untouched and adds a small cosine loss on
        # hidden deltas h[t+1] - h[t] for selected late layers and positions.
        self.use_repr_aux = use_repr_aux
        self.repr_aux_weight = repr_aux_weight
        self.repr_aux_position_count = repr_aux_position_count
        self.repr_aux_layer_fraction = repr_aux_layer_fraction
        self.repr_aux_eps = repr_aux_eps

        # Trajectory selection leaves the OPSD loss untouched. It samples several
        # student completions, then selects one complete trajectory before either
        # the student or privileged teacher forward pass.
        self.trajectory_selection_rollouts = trajectory_selection_rollouts
        self.trajectory_selection_mode = trajectory_selection_mode

        if self.entropy_gate_mode not in {"inverse", "direct"}:
            raise ValueError("entropy_gate_mode must be either 'inverse' or 'direct'.")
        if self.entropy_gate_schedule not in {"constant", "linear_decay", "cosine_decay", "phase_off"}:
            raise ValueError(
                "entropy_gate_schedule must be one of: constant, linear_decay, cosine_decay, phase_off."
            )
        if not 0 <= self.entropy_gate_schedule_start <= 1:
            raise ValueError("entropy_gate_schedule_start must be in [0, 1].")
        if not 0 <= self.entropy_gate_schedule_end <= 1:
            raise ValueError("entropy_gate_schedule_end must be in [0, 1].")
        if self.entropy_gate_schedule_start > self.entropy_gate_schedule_end:
            raise ValueError("entropy_gate_schedule_start cannot exceed entropy_gate_schedule_end.")
        if self.entropy_gate_min < 0 or self.entropy_gate_max <= 0:
            raise ValueError("entropy_gate_min must be >= 0 and entropy_gate_max must be > 0.")
        if self.entropy_gate_min > self.entropy_gate_max:
            raise ValueError("entropy_gate_min cannot exceed entropy_gate_max.")
        if self.entropy_gate_power <= 0:
            raise ValueError("entropy_gate_power must be positive.")
        if self.repr_aux_weight < 0:
            raise ValueError("repr_aux_weight must be non-negative.")
        if self.repr_aux_position_count <= 0:
            raise ValueError("repr_aux_position_count must be positive.")
        if not 0 < self.repr_aux_layer_fraction <= 1:
            raise ValueError("repr_aux_layer_fraction must be in (0, 1].")
        if self.repr_aux_eps <= 0:
            raise ValueError("repr_aux_eps must be positive.")
        if self.trajectory_selection_rollouts <= 0:
            raise ValueError("trajectory_selection_rollouts must be positive.")
        if self.trajectory_selection_mode not in {"none", "gold_consensus"}:
            raise ValueError("trajectory_selection_mode must be 'none' or 'gold_consensus'.")
        if self.trajectory_selection_rollouts == 1 and self.trajectory_selection_mode != "none":
            raise ValueError("trajectory_selection_mode requires trajectory_selection_rollouts > 1.")
        if self.trajectory_selection_rollouts > 1 and self.trajectory_selection_mode == "none":
            raise ValueError("trajectory_selection_rollouts > 1 requires a selection mode.")
        if self.trajectory_selection_rollouts > 1 and self.task_type != "math":
            raise ValueError("trajectory selection currently supports task_type='math' only.")

        # ReGap-OPSD Lite: Counterfactual Rescue-Gap branch supervision.
        # "replace" is the original ablation: sparse branch CE + decision PI-KL
        # replaces full-token OPSD. "additive" keeps the full OPSD trunk and adds
        # ReGap as a small branch-level calibration term. "weighted" keeps full
        # OPSD and uses rescue/suspicious gaps only as detached sample weights.
        self.use_regap = use_regap
        self.regap_mode = regap_mode
        self.regap_branch_weight = regap_branch_weight
        self.regap_top_k = regap_top_k
        self.regap_tau = regap_tau
        self.regap_lambda_pi = regap_lambda_pi
        self.regap_eta_dead = regap_eta_dead
        self.regap_dead_teacher_threshold = regap_dead_teacher_threshold
        self.regap_dead_student_threshold = regap_dead_student_threshold
        self.regap_decision_ratio = regap_decision_ratio
        self.regap_min_decisions = regap_min_decisions
        self.regap_gap_weight = regap_gap_weight
        self.regap_disagreement_weight = regap_disagreement_weight
        self.regap_student_entropy_weight = regap_student_entropy_weight
        self.regap_weight_alpha = regap_weight_alpha
        self.regap_weight_beta = regap_weight_beta
        self.regap_weight_min = regap_weight_min
        self.regap_weight_max = regap_weight_max
        self.regap_eps = 1e-6

        if self.use_regap and self.use_thinking_machines_loss:
            raise ValueError("use_regap=True currently supports the JSD/OPSD loss path, not use_tinker_loss.")
        if self.use_regap and self.top_k_loss is not None and self.top_k_loss > 0:
            raise ValueError("use_regap=True currently requires top_k_loss=0 so candidate branches use full-vocab log-probs.")
        if self.regap_mode not in {"replace", "additive", "weighted"}:
            raise ValueError("regap_mode must be 'replace', 'additive', or 'weighted'.")
        if self.regap_branch_weight < 0:
            raise ValueError("regap_branch_weight must be non-negative.")
        if self.regap_top_k <= 0:
            raise ValueError("regap_top_k must be positive.")
        if self.regap_tau <= 0:
            raise ValueError("regap_tau must be positive.")
        if self.regap_lambda_pi < 0 or self.regap_eta_dead < 0:
            raise ValueError("regap_lambda_pi and regap_eta_dead must be non-negative.")
        if not 0 < self.regap_decision_ratio <= 1:
            raise ValueError("regap_decision_ratio must be in (0, 1].")
        if self.regap_min_decisions <= 0:
            raise ValueError("regap_min_decisions must be positive.")
        if self.regap_weight_alpha < 0 or self.regap_weight_beta < 0:
            raise ValueError("regap_weight_alpha and regap_weight_beta must be non-negative.")
        if self.regap_weight_min <= 0 or self.regap_weight_max <= 0:
            raise ValueError("regap_weight_min and regap_weight_max must be positive.")
        if self.regap_weight_min > self.regap_weight_max:
            raise ValueError("regap_weight_min cannot exceed regap_weight_max.")
        if (
            self.regap_gap_weight < 0
            or self.regap_disagreement_weight < 0
            or self.regap_student_entropy_weight < 0
        ):
            raise ValueError(
                "regap_gap_weight, regap_disagreement_weight, and "
                "regap_student_entropy_weight must be non-negative."
            )
        if (
            self.regap_gap_weight == 0
            and self.regap_disagreement_weight == 0
            and self.regap_student_entropy_weight == 0
        ):
            raise ValueError("At least one ReGap decision-score weight must be positive.")

        # Global EMA statistics for weight normalization
        self._global_weight_mean = 1.0
        self._global_weight_count = 0

        # Validate fixed_teacher option
        if self.fixed_teacher and peft_config is None:
            raise ValueError(
                "fixed_teacher=True requires a PEFT config (use_peft=True). "
                "The fixed teacher is implemented by disabling LoRA adapters during teacher forward passes."
            )

        if self.use_ema_teacher and self.fixed_teacher:
            raise ValueError(
                "use_ema_teacher=True and fixed_teacher=True are mutually exclusive teacher strategies."
            )
        if self.use_rift_routing and not self.fixed_teacher:
            raise ValueError("use_rift_routing=True requires fixed_teacher=True.")
        if self.use_rift_routing and self.use_thinking_machines_loss:
            raise ValueError("RIFT routing requires the full-vocabulary JSD loss.")
        if self.use_rift_routing and self.top_k_loss is not None and self.top_k_loss > 0:
            raise ValueError("RIFT routing requires top_k_loss=0.")
        if self.use_rift_routing and self.use_entropy_gating:
            raise ValueError("RIFT routing and entropy gating are separate ablations and cannot be enabled together.")
        if self.use_rift_routing and self.use_regap:
            raise ValueError("RIFT routing and ReGap are separate ablations and cannot be enabled together.")
        if self.rift_sign_margin < 0:
            raise ValueError("rift_sign_margin must be non-negative.")
        if not 0 <= self.rift_entropy_quantile <= 1:
            raise ValueError("rift_entropy_quantile must be in [0, 1].")
        if not 0 <= self.rift_route_weight <= 1:
            raise ValueError("rift_route_weight must be in [0, 1].")
        if self.rift_recovery_window < 0:
            raise ValueError("rift_recovery_window must be non-negative.")
        if self.rift_recovery_margin < 0:
            raise ValueError("rift_recovery_margin must be non-negative.")
        if self.rift_recovery_quantile != -1 and not 0 <= self.rift_recovery_quantile <= 1:
            raise ValueError("rift_recovery_quantile must be -1 or in [0, 1].")
        if self.rift_routing_score not in {"future_recovery", "ad_risk"}:
            raise ValueError(
                "rift_routing_score must be future_recovery or ad_risk."
            )
        if self.rift_exact_rank:
            if self.rift_recovery_quantile < 0 or self.rift_recovery_window <= 0:
                raise ValueError(
                    "rift_exact_rank requires adaptive routing with a positive future window."
                )
            if (
                self.rift_hard_entropy_quantile >= 0
                or self.rift_groupwise_recovery_quantiles
            ):
                raise ValueError(
                    "rift_exact_rank cannot be combined with uncertainty-banded routing."
                )
        if self.rift_routing_score == "ad_risk" and not self.rift_exact_rank:
            raise ValueError("ad_risk routing requires rift_exact_rank.")
        if self.rift_require_full_window and self.rift_recovery_window <= 0:
            raise ValueError(
                "rift_require_full_window requires rift_recovery_window > 0."
            )
        hard_dq_values = (self.rift_hard_entropy_quantile, self.rift_hard_recovery_quantile)
        if (hard_dq_values[0] == -1) != (hard_dq_values[1] == -1):
            raise ValueError("RIFT-DQ hard entropy and recovery quantiles must be enabled together.")
        if hard_dq_values[0] != -1:
            if not all(0 <= value <= 1 for value in hard_dq_values):
                raise ValueError("RIFT-DQ hard entropy and recovery quantiles must be in [0, 1].")
            if self.rift_recovery_quantile < 0 or self.rift_recovery_window <= 0:
                raise ValueError("RIFT-DQ requires adaptive recovery routing with a positive future window.")
            if self.rift_hard_entropy_quantile < self.rift_entropy_quantile:
                raise ValueError("rift_hard_entropy_quantile must be >= rift_entropy_quantile.")
        elif self.rift_groupwise_recovery_quantiles:
            raise ValueError("rift_groupwise_recovery_quantiles requires uncertainty-banded routing.")
        if self.rift_fork_onset_gap < 1:
            raise ValueError("rift_fork_onset_gap must be positive.")
        if self.rift_fork_onset_routing:
            if self.rift_recovery_window <= 0 or self.rift_recovery_quantile < 0:
                raise ValueError(
                    "RIFT-FO requires adaptive recovery routing with a positive future window."
                )
            if self.rift_hard_entropy_quantile >= 0 or self.rift_groupwise_recovery_quantiles:
                raise ValueError("RIFT-FO cannot be combined with uncertainty-banded routing.")
        if self.rift_reflection_safe_weighting and not 0 < self.rift_reflection_protection_weight < 1:
            raise ValueError("rift_reflection_protection_weight must be in (0, 1).")
        if self.rift_reflection_safe_weighting:
            if self.rift_recovery_window <= 0 or self.rift_recovery_quantile < 0:
                raise ValueError(
                    "RIFT-RS requires adaptive recovery routing with a positive future window."
                )
            if self.rift_fork_onset_routing:
                raise ValueError("RIFT-RS keeps q25 routing unchanged and cannot enable RIFT-FO.")
            if self.rift_hard_entropy_quantile >= 0 or self.rift_groupwise_recovery_quantiles:
                raise ValueError("RIFT-RS cannot be combined with uncertainty-banded routing.")
        if self.rift_soft_clamp_multiplier <= 0:
            raise ValueError("rift_soft_clamp_multiplier must be positive.")
        if self.rift_asymmetric_soft_clamp:
            if self.rift_recovery_window <= 0 or self.rift_recovery_quantile < 0:
                raise ValueError(
                    "RIFT-ASC requires adaptive recovery routing with a positive future window."
                )
            if self.rift_reflection_safe_weighting or self.rift_fork_onset_routing:
                raise ValueError("RIFT-ASC cannot be combined with RIFT-RS or RIFT-FO.")
            if self.rift_hard_entropy_quantile >= 0 or self.rift_groupwise_recovery_quantiles:
                raise ValueError("RIFT-ASC cannot be combined with uncertainty-banded routing.")
        if self.rift_asymmetric_log_compression:
            if self.rift_recovery_window <= 0 or self.rift_recovery_quantile < 0:
                raise ValueError(
                    "RIFT-ALC requires adaptive recovery routing with a positive future window."
                )
            if (
                self.rift_reflection_safe_weighting
                or self.rift_fork_onset_routing
                or self.rift_asymmetric_soft_clamp
            ):
                raise ValueError("RIFT-ALC cannot combine with RIFT-RS, RIFT-FO, or RIFT-ASC.")
            if self.rift_hard_entropy_quantile >= 0 or self.rift_groupwise_recovery_quantiles:
                raise ValueError("RIFT-ALC cannot be combined with uncertainty-banded routing.")
        if self.rift_base_persistence_window < 1:
            raise ValueError("rift_base_persistence_window must be positive.")
        if self.rift_base_persistence_min_gain < 0:
            raise ValueError("rift_base_persistence_min_gain must be non-negative.")
        if self.rift_base_persistence_min_gain > self.rift_base_persistence_window:
            raise ValueError("rift_base_persistence_min_gain cannot exceed the support window.")
        if self.rift_base_persistence_min_gain > 0 and not self.rift_base_persistence_routing:
            raise ValueError("RIFT-ASG minimum support gain requires base-persistence routing.")
        if self.rift_base_persistence_routing:
            if self.rift_recovery_window <= 0 or self.rift_recovery_quantile < 0:
                raise ValueError(
                    "RIFT-BSP requires adaptive recovery routing with a positive future window."
                )
            if self.rift_route_weight != 1.0:
                raise ValueError("RIFT-BSP requires a hard route weight of 1.0.")
            if self.rift_hard_entropy_quantile >= 0 or self.rift_groupwise_recovery_quantiles:
                raise ValueError("RIFT-BSP cannot be combined with uncertainty-banded routing.")
            if (
                self.rift_fork_onset_routing
                or self.rift_reflection_safe_weighting
                or self.rift_asymmetric_soft_clamp
                or self.rift_asymmetric_log_compression
            ):
                raise ValueError("RIFT-BSP replaces q25 selection and cannot combine with FO, RS, or ASC.")

        # Handle teacher model
        self.teacher_model = teacher_model
        self.use_separate_teacher = teacher_model is not None

        if self.use_separate_teacher:
            print(f"\n{'='*80}")
            print("SEPARATE TEACHER MODEL MODE ENABLED")
            print(f"Using separate teacher model: {teacher_model.__class__.__name__}")
            print(f"Teacher parameters: {teacher_model.num_parameters():,}")
            print("Teacher model is kept frozen during training")
            print(f"{'='*80}\n")

            # Disable EMA and fixed_teacher when using separate teacher
            if self.use_ema_teacher:
                print("Warning: use_ema_teacher is disabled when using separate teacher model")
                self.use_ema_teacher = False
            if self.fixed_teacher:
                print("Warning: fixed_teacher is disabled when using separate teacher model")
                self.fixed_teacher = False

            # Put teacher model in eval mode and disable gradients
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False

            # Move teacher model to the correct device
            teacher_device = self.accelerator.device
            teacher_model.to(teacher_device)
            print(f"Teacher model moved to device: {teacher_device}")
            print(f"{'='*80}\n")

        if self.use_ema_teacher:
            self.add_callback(EMAUpdateCallback(self))
            print(f"\n{'='*80}")
            print("EMA TEACHER MODE ENABLED")
            print(f"EMA decay: {self.ema_decay}")
            print("Teacher is an exponential moving average of the student weights.")
            print("EMA parameters are initialized on the first optimizer step.")
            print(f"{'='*80}\n")

        if self.fixed_teacher:
            print(f"\n{'='*80}")
            print("FIXED TEACHER MODE ENABLED")
            print("Teacher will use the initial policy (base model without LoRA adapters)")
            print("Student will update with LoRA adapters")
            print(f"{'='*80}\n")

        if self.use_rift_routing:
            print(f"\n{'='*80}")
            print("RIFT SIGN-CONSISTENT PRIVILEGE ROUTING ENABLED")
            print(f"Sign margin: {self.rift_sign_margin}")
            print(f"Student entropy quantile: {self.rift_entropy_quantile}")
            print(f"Route weight: {self.rift_route_weight}")
            print(f"Recovery window: {self.rift_recovery_window}")
            print(f"Recovery margin: {self.rift_recovery_margin}")
            print(f"Recovery quantile: {self.rift_recovery_quantile}")
            print(f"Exact-rank routing: {self.rift_exact_rank}")
            print(f"Routing score: {self.rift_routing_score}")
            print(f"Require full future window: {self.rift_require_full_window}")
            print(f"Hard-fork entropy quantile: {self.rift_hard_entropy_quantile}")
            print(f"Hard-fork recovery quantile: {self.rift_hard_recovery_quantile}")
            print(f"Groupwise recovery quantiles: {self.rift_groupwise_recovery_quantiles}")
            print(f"Fork-onset routing: {self.rift_fork_onset_routing}")
            print(f"Fork-onset gap: {self.rift_fork_onset_gap}")
            print(f"Reflection-safe weighting: {self.rift_reflection_safe_weighting}")
            print(f"Reflection protection weight: {self.rift_reflection_protection_weight}")
            print(f"Asymmetric soft clamp: {self.rift_asymmetric_soft_clamp}")
            print(f"Soft-clamp multiplier: {self.rift_soft_clamp_multiplier}")
            print(f"Asymmetric log compression: {self.rift_asymmetric_log_compression}")
            print(f"Base-support persistence routing: {self.rift_base_persistence_routing}")
            print(f"Base-support persistence window: {self.rift_base_persistence_window}")
            print(f"Base-support minimum swap gain: {self.rift_base_persistence_min_gain}")
            if (
                self.rift_reflection_safe_weighting
                or self.rift_asymmetric_soft_clamp
                or self.rift_asymmetric_log_compression
            ):
                print(f"Reflection marker token sequences: {len(self.rift_reflection_token_sequences)}")
            print(f"{'='*80}\n")

        if self.reason_first:
            print(f"\n{'='*80}")
            print("REASON FIRST MODE ENABLED")
            print("Teacher will first reason about the privileged solution, then evaluate student's response")
            print(f"{'='*80}\n")

        # Track per-step loss statistics for on/off-policy batches (used in logging)
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0
        self._trajectory_selection_sums = defaultdict(float)

        self.use_transformers_paged = args.use_transformers_paged or False
        
        # Track generation outputs for saving
        self._generation_outputs_buffer = []
        self._generation_save_frequency = 5  # Save every 5 steps

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
            remove_invalid_values=True,
            renormalize_logits=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Generation config for reasoning phase (when reason_first=True)
        max_reasoning_length = getattr(args, "max_reasoning_length", 4096)
        self.reasoning_generation_config = GenerationConfig(
            max_new_tokens=max_reasoning_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
            remove_invalid_values=True,
            renormalize_logits=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.reasoning_generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_mode = args.vllm_mode
            self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
            self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
            self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    self.vllm_client = VLLMClient(
                        host=args.vllm_server_host,
                        server_port=args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout,
                    )
                    self.vllm_client.init_communicator()
            elif self.vllm_mode == "colocate":
                student_model_name_or_path = self.model_name_or_path

                # Make sure tensor_parallel_size divides world size evenly
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP
                    self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()

                vllm_enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "0").lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if vllm_enforce_eager:
                    print("VLLM_ENFORCE_EAGER=1: disabling vLLM CUDA graph capture for stability.")

                self.vllm_engine = LLM(
                    model=student_model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps,
                    max_model_len=args.max_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=self.vllm_enable_sleep_mode,
                    enforce_eager=vllm_enforce_eager,
                )

                if self.vllm_enable_sleep_mode:
                    self.vllm_engine.sleep(level=2)

                # When using vLLM, the main process is responsible for loading the model weights. This can cause process
                # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
                # synchronize all processes after vLLM has been fully initialized.
                self.accelerator.wait_for_everyone()
            else:
                raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")
            self.vllm_guided_decoding_regex = args.vllm_guided_decoding_regex
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1

            self.add_callback(GOLDVLLMSyncCallback(self))

    def _cast_trainable_parameters_to_fp32(self):
        """Keep PEFT/LoRA trainable weights stable under bf16 DeepSpeed CPUAdam."""
        converted_params = 0
        converted_elements = 0
        for _, param in self.model.named_parameters():
            if param.requires_grad and param.dtype != torch.float32:
                converted_params += 1
                converted_elements += param.numel()
                param.data = param.data.float()

        if converted_params:
            print(
                f"Converted {converted_params} trainable parameter tensors "
                f"({converted_elements:,} elements) to fp32 for optimizer stability."
            )

    def _resolve_model_safe_pad_token_id(self):
        """Return a token id that is safe to feed to the model embedding."""
        for token_id in (
            self.processing_class.pad_token_id,
            self.processing_class.eos_token_id,
        ):
            if token_id is not None and 0 <= int(token_id) < self.student_vocab_size:
                return int(token_id)
        return 0

    def _sanitize_token_ids_for_model(self, token_ids, context):
        """Replace invalid token ids before they reach the HF model embedding."""
        if token_ids is None:
            return token_ids

        invalid_mask = (token_ids < 0) | (token_ids >= self.student_vocab_size)
        if not invalid_mask.any():
            return token_ids

        invalid_count = int(invalid_mask.sum().item())
        min_token = int(token_ids.min().item())
        max_token = int(token_ids.max().item())
        rank = getattr(self.accelerator, "process_index", 0)
        print(
            f"[token_sanitize][rank={rank}] {context}: replaced {invalid_count} invalid token ids "
            f"(min={min_token}, max={max_token}, vocab={self.student_vocab_size}) "
            f"with {self.model_safe_pad_token_id}."
        )

        token_ids = token_ids.clone()
        token_ids[invalid_mask] = self.model_safe_pad_token_id
        return token_ids

    def _sanitize_log_probs_for_loss(self, log_probs, context):
        """Keep occasional invalid logits from poisoning the whole ReGap batch."""
        if log_probs is None:
            return log_probs

        finite_mask = torch.isfinite(log_probs)
        if finite_mask.all():
            return log_probs

        invalid_count = int((~finite_mask).sum().item())
        rank = getattr(self.accelerator, "process_index", 0)
        print(
            f"[logprob_sanitize][rank={rank}] {context}: replaced {invalid_count} "
            "non-finite log-prob values."
        )

        neg_large = torch.finfo(log_probs.dtype).min
        return torch.nan_to_num(
            log_probs,
            nan=neg_large,
            posinf=0.0,
            neginf=neg_large,
        ).clamp(max=0.0)

    @staticmethod
    def _extract_trajectory_boxed_answer(text):
        """Return the final balanced ``\\boxed{...}`` answer from a completion."""
        think_end = text.rfind("</think>")
        search_text = text[think_end + len("</think>") :] if think_end >= 0 else text
        start = search_text.rfind(r"\boxed{")
        if start < 0:
            return None

        start += len(r"\boxed{")
        depth = 1
        index = start
        while index < len(search_text) and depth > 0:
            if search_text[index] == "{":
                depth += 1
            elif search_text[index] == "}":
                depth -= 1
            index += 1

        return search_text[start : index - 1].strip() if depth == 0 else None

    @staticmethod
    def _normalize_trajectory_answer(answer):
        """Use a cheap deterministic equivalence check in the training hot path."""
        if answer is None:
            return ""
        normalized = str(answer).strip()
        normalized = normalized.replace("$", "")
        normalized = normalized.replace(r"\left", "").replace(r"\right", "")
        normalized = normalized.replace(r"\displaystyle", "")
        return re.sub(r"\s+", "", normalized).lower()

    @staticmethod
    def _rift_fork_onset_mask(candidate_mask, onset_gap):
        """Return the first candidate token in each locally connected fork episode."""
        if onset_gap < 1:
            raise ValueError("RIFT fork onset_gap must be positive.")
        sequence_length = candidate_mask.shape[1]
        recent_candidate = torch.zeros_like(candidate_mask)
        max_offset = min(onset_gap, sequence_length)
        for offset in range(1, max_offset + 1):
            recent_candidate[:, offset:] |= candidate_mask[:, :-offset]
        return candidate_mask & ~recent_candidate

    @staticmethod
    def _match_rift_reflection_tokens(token_ids, valid_mask, token_sequences):
        """Mark complete occurrences of configured reflection-marker token sequences."""
        if token_ids.shape != valid_mask.shape:
            raise ValueError("RIFT-RS token ids and valid mask must have identical shapes.")
        matched_tokens = torch.zeros_like(valid_mask)
        sequence_length = token_ids.shape[1]
        for sequence in token_sequences:
            marker_length = len(sequence)
            if marker_length == 0 or marker_length > sequence_length:
                continue
            window_count = sequence_length - marker_length + 1
            matched_starts = torch.ones(
                (token_ids.shape[0], window_count),
                dtype=torch.bool,
                device=token_ids.device,
            )
            for offset, marker_token_id in enumerate(sequence):
                matched_starts &= token_ids[:, offset : offset + window_count].eq(
                    marker_token_id
                )
                matched_starts &= valid_mask[:, offset : offset + window_count]
            for offset in range(marker_length):
                matched_tokens[:, offset : offset + window_count] |= matched_starts
        return matched_tokens & valid_mask

    @staticmethod
    def _apply_rift_reflection_protection(
        route_mask,
        protected_mask,
        route_weight,
        protection_weight,
        dtype,
    ):
        """Keep hard q25 routing intact and add a fractional gate only at protected tokens."""
        if route_mask.shape != protected_mask.shape:
            raise ValueError("RIFT-RS route and protection masks must have identical shapes.")
        hard_gate = route_mask.to(dtype) * route_weight
        soft_gate = protected_mask.to(dtype) * protection_weight
        return torch.maximum(hard_gate, soft_gate)

    @staticmethod
    def _apply_rift_asymmetric_soft_clamp(
        legacy_loss,
        raw_loss,
        privileged_advantage,
        protected_mask,
        route_mask,
        valid_mask,
        multiplier,
        sign_margin,
        hard_cap=None,
    ):
        """Softly clamp only suppressed, unrouted reflection-boundary tokens.

        The detached cap follows Soft Clamp, but the intervention is asymmetric:
        it is restricted to positions where the privileged teacher assigns the
        sampled token a sufficiently lower log-probability than the student.
        Recovery routing and the legacy loss remain unchanged everywhere else.
        """
        expected_shape = legacy_loss.shape
        tensors = (
            raw_loss,
            privileged_advantage,
            protected_mask,
            route_mask,
            valid_mask,
        )
        if any(tensor.shape != expected_shape for tensor in tensors):
            raise ValueError("RIFT-ASC tensors must have identical token shapes.")
        if multiplier <= 0:
            raise ValueError("RIFT-ASC multiplier must be positive.")

        valid_float = valid_mask.to(raw_loss.dtype)
        cap = multiplier * (
            (raw_loss.detach() * valid_float).sum()
            / valid_float.sum().clamp(min=1.0)
        )
        if hard_cap is not None and hard_cap > 0:
            cap = torch.minimum(cap, cap.new_tensor(float(hard_cap)))
        cap = cap.detach().clamp_min(torch.finfo(raw_loss.dtype).tiny)

        protected_suppression = (
            protected_mask
            & ~route_mask
            & valid_mask
            & (privileged_advantage <= -sign_margin)
        )
        active_mask = (
            protected_suppression
            & (raw_loss.detach() > cap)
            & (legacy_loss.detach() > cap)
        )
        detached_scale = cap / raw_loss.detach().clamp_min(
            torch.finfo(raw_loss.dtype).tiny
        )
        token_scale = torch.where(
            active_mask,
            detached_scale.clamp(max=1.0),
            torch.ones_like(raw_loss),
        ).detach()
        soft_clamped = raw_loss * token_scale
        calibrated_loss = torch.where(active_mask, soft_clamped, legacy_loss)
        return calibrated_loss, active_mask, protected_suppression, cap, token_scale

    @staticmethod
    def _apply_rift_asymmetric_log_compression(
        legacy_loss,
        privileged_advantage,
        protected_mask,
        route_mask,
        valid_mask,
        sign_margin,
    ):
        """Compress privileged suppression at unrouted reflection boundaries."""
        expected_shape = legacy_loss.shape
        tensors = (privileged_advantage, protected_mask, route_mask, valid_mask)
        if any(tensor.shape != expected_shape for tensor in tensors):
            raise ValueError("RIFT-ALC tensors must have identical token shapes.")

        active_mask = (
            protected_mask
            & ~route_mask
            & valid_mask
            & (privileged_advantage <= -sign_margin)
        )
        magnitude = (-privileged_advantage).clamp_min(
            torch.finfo(legacy_loss.dtype).tiny
        )
        log_scale = (torch.log1p(magnitude) / magnitude).to(legacy_loss.dtype)
        token_scale = torch.where(
            active_mask,
            log_scale.clamp(min=torch.finfo(legacy_loss.dtype).tiny, max=1.0),
            torch.ones_like(legacy_loss),
        ).detach()
        calibrated_loss = legacy_loss * token_scale
        return calibrated_loss, active_mask, token_scale

    @staticmethod
    def _frontload_rift_fork_budget(
        candidate_mask,
        pool_mask,
        reference_route_mask,
        recovery_score,
        onset_gap,
    ):
        """Move an exact reference route budget toward the start of candidate episodes.

        Candidate tokens separated by at most ``onset_gap`` positions belong to the
        same fork episode. Selection is lexicographic: smaller distance from the
        episode onset first, then larger recovery score, then earlier position.
        The selected count is exactly the q25 reference count for each trajectory.
        """
        if candidate_mask.shape != pool_mask.shape or candidate_mask.shape != reference_route_mask.shape:
            raise ValueError("RIFT-FO masks must have identical shapes.")
        if candidate_mask.shape != recovery_score.shape:
            raise ValueError("RIFT-FO recovery scores must match mask shape.")
        if onset_gap < 1:
            raise ValueError("RIFT-FO onset_gap must be positive.")

        batch_size, sequence_length = candidate_mask.shape
        onset_mask = OPSDTrainer._rift_fork_onset_mask(candidate_mask, onset_gap)

        positions = torch.arange(sequence_length, device=candidate_mask.device).unsqueeze(0)
        start_positions = torch.where(onset_mask, positions, positions.new_full((), -1))
        last_start = torch.cummax(start_positions, dim=1).values
        fork_age = torch.where(
            candidate_mask,
            positions - last_start,
            positions.new_full((batch_size, sequence_length), sequence_length),
        )

        selected = torch.zeros_like(candidate_mask)
        for batch_idx in range(batch_size):
            pool_indices = torch.nonzero(pool_mask[batch_idx], as_tuple=False).squeeze(-1)
            target_count = min(
                int(reference_route_mask[batch_idx].sum().item()),
                int(pool_indices.numel()),
            )
            if target_count == 0:
                continue
            pool_recovery = recovery_score[batch_idx, pool_indices].double()
            recovery_min = pool_recovery.min()
            recovery_range = pool_recovery.max() - recovery_min
            normalized_recovery = torch.where(
                recovery_range > 0,
                (pool_recovery - recovery_min) / recovery_range,
                torch.zeros_like(pool_recovery),
            )
            priority = (
                -fork_age[batch_idx, pool_indices].double()
                + 0.25 * normalized_recovery
                - 1e-12 * pool_indices.double()
            )
            chosen = pool_indices[torch.topk(priority, target_count, sorted=False).indices]
            selected[batch_idx, chosen] = True
        return selected, onset_mask, fork_age

    @staticmethod
    def _select_rift_base_persistent_budget(
        candidate_mask,
        pool_mask,
        reference_route_mask,
        recovery_score,
        base_advantage,
        valid_mask,
        support_window,
        sign_margin,
        min_support_gain=0,
    ):
        """Keep q25's exact route count while favoring locally supported base branches.

        A route is more trustworthy when the unprivileged teacher continues to
        support the sampled continuation, not only the initial fork token.
        Selection is lexicographic: greater future support count, then greater
        recovery score, then earlier position.
        """
        masks = (candidate_mask, pool_mask, reference_route_mask, valid_mask)
        if any(mask.shape != candidate_mask.shape for mask in masks):
            raise ValueError("RIFT-BSP masks must have identical shapes.")
        if recovery_score.shape != candidate_mask.shape:
            raise ValueError("RIFT-BSP recovery scores must match mask shape.")
        if base_advantage.shape != candidate_mask.shape:
            raise ValueError("RIFT-BSP advantages must match mask shape.")
        if support_window < 1:
            raise ValueError("RIFT-BSP support_window must be positive.")
        if min_support_gain < 0:
            raise ValueError("RIFT-ASG min_support_gain must be non-negative.")

        batch_size, sequence_length = candidate_mask.shape
        support_count = torch.zeros_like(recovery_score, dtype=torch.long)
        max_offset = min(support_window, max(sequence_length - 1, 0))
        for offset in range(1, max_offset + 1):
            invalid_tail = torch.zeros_like(valid_mask[:, :offset])
            future_valid = torch.cat([valid_mask[:, offset:], invalid_tail], dim=1)
            future_supported = torch.cat(
                [base_advantage[:, offset:] >= -sign_margin, invalid_tail], dim=1
            )
            support_count += (future_valid & future_supported).long()

        if min_support_gain > 0:
            selected = reference_route_mask.clone()
            for batch_idx in range(batch_size):
                pool_indices = torch.nonzero(
                    pool_mask[batch_idx], as_tuple=False
                ).squeeze(-1)
                if pool_indices.numel() < 2:
                    continue
                incoming_indices = torch.nonzero(
                    pool_mask[batch_idx] & ~reference_route_mask[batch_idx],
                    as_tuple=False,
                ).squeeze(-1)
                outgoing_indices = torch.nonzero(
                    pool_mask[batch_idx] & reference_route_mask[batch_idx],
                    as_tuple=False,
                ).squeeze(-1)
                pair_count = min(incoming_indices.numel(), outgoing_indices.numel())
                if pair_count == 0:
                    continue

                pool_recovery = recovery_score[batch_idx, pool_indices].double()
                recovery_min = pool_recovery.min()
                recovery_range = pool_recovery.max() - recovery_min
                normalized_recovery = torch.zeros_like(
                    recovery_score[batch_idx], dtype=torch.double
                )
                normalized_recovery[pool_indices] = torch.where(
                    recovery_range > 0,
                    (pool_recovery - recovery_min) / recovery_range,
                    torch.zeros_like(pool_recovery),
                )
                priority = (
                    2.0 * support_count[batch_idx].double()
                    + normalized_recovery
                    - 1e-12
                    * torch.arange(
                        sequence_length,
                        device=pool_indices.device,
                        dtype=torch.double,
                    )
                )
                incoming_order = incoming_indices[
                    torch.argsort(
                        priority[incoming_indices], descending=True, stable=True
                    )
                ]
                outgoing_order = outgoing_indices[
                    torch.argsort(
                        priority[outgoing_indices], descending=False, stable=True
                    )
                ]
                for pair_idx in range(pair_count):
                    incoming_idx = incoming_order[pair_idx]
                    outgoing_idx = outgoing_order[pair_idx]
                    support_gain = (
                        support_count[batch_idx, incoming_idx]
                        - support_count[batch_idx, outgoing_idx]
                    )
                    if int(support_gain.item()) < min_support_gain:
                        break
                    selected[batch_idx, outgoing_idx] = False
                    selected[batch_idx, incoming_idx] = True
            return selected, support_count

        selected = torch.zeros_like(candidate_mask)
        for batch_idx in range(batch_size):
            pool_indices = torch.nonzero(pool_mask[batch_idx], as_tuple=False).squeeze(-1)
            target_count = min(
                int(reference_route_mask[batch_idx].sum().item()),
                int(pool_indices.numel()),
            )
            if target_count == 0:
                continue
            pool_recovery = recovery_score[batch_idx, pool_indices].double()
            recovery_min = pool_recovery.min()
            recovery_range = pool_recovery.max() - recovery_min
            normalized_recovery = torch.where(
                recovery_range > 0,
                (pool_recovery - recovery_min) / recovery_range,
                torch.zeros_like(pool_recovery),
            )
            priority = (
                2.0 * support_count[batch_idx, pool_indices].double()
                + normalized_recovery
                - 1e-12 * pool_indices.double()
            )
            chosen = pool_indices[torch.topk(priority, target_count, sorted=False).indices]
            selected[batch_idx, chosen] = True
        return selected, support_count

    def _select_trajectory_candidates(self, completion_id_groups, answer_gt):
        """Select one rollout per prompt without changing the OPSD objective.

        The rank order is correctness against the dataset answer, then agreement
        among candidate final answers, then the original sampling order. Exact
        normalized answer matching is intentional here: it is deterministic and
        cheap enough to remain outside the training critical path.
        """
        if len(completion_id_groups) != len(answer_gt):
            raise ValueError(
                "Trajectory selection received mismatched prompt and answer counts: "
                f"{len(completion_id_groups)} candidates groups vs {len(answer_gt)} answers."
            )

        selected_ids = []
        stats = defaultdict(float)
        for candidate_ids, gold_answer in zip(completion_id_groups, answer_gt):
            if not candidate_ids:
                raise ValueError("Trajectory selection received an empty candidate group.")

            candidate_texts = [
                self.processing_class.decode(ids, skip_special_tokens=False) for ids in candidate_ids
            ]
            answers = [self._extract_trajectory_boxed_answer(text) for text in candidate_texts]
            normalized_answers = [self._normalize_trajectory_answer(answer) for answer in answers]
            normalized_gold = self._normalize_trajectory_answer(gold_answer)
            correctness = [
                bool(normalized_gold and answer and answer == normalized_gold)
                for answer in normalized_answers
            ]

            answer_counts = defaultdict(int)
            for answer in normalized_answers:
                if answer:
                    answer_counts[answer] += 1

            def rank(index):
                answer = normalized_answers[index]
                return (
                    int(correctness[index]),
                    answer_counts.get(answer, 0),
                    -index,
                )

            selected_index = max(range(len(candidate_ids)), key=rank)
            selected_answer = normalized_answers[selected_index]
            selected_ids.append(candidate_ids[selected_index])

            stats["samples"] += 1
            stats["candidate_count"] += len(candidate_ids)
            stats["candidate_correct"] += sum(correctness)
            stats["oracle_correct"] += int(any(correctness))
            stats["selected_correct"] += int(correctness[selected_index])
            stats["formatted_candidates"] += sum(bool(answer) for answer in normalized_answers)
            stats["selected_consensus"] += answer_counts.get(selected_answer, 0) / len(candidate_ids)

        for key, value in stats.items():
            self._trajectory_selection_sums[key] += value
        return selected_ids

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = [
            "problem",
            "solution",
            "Answer",
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    def _compute_jsd_per_token(self, student_logits, teacher_logits):
        """
        Compute per-token JSD between student and teacher logits.

        Applies temperature scaling, optional top-k restriction, and token clipping.

        Args:
            student_logits: Raw logits (before temperature), shape (batch_size, seq_len, vocab_size)
            teacher_logits: Raw logits (before temperature), shape (batch_size, seq_len, vocab_size)

        Returns:
            per_token_jsd: shape (batch_size, seq_len), JSD summed over vocab dimension
        """
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature

        if self.top_k_loss is not None and self.top_k_loss > 0:
            _, top_k_indices = torch.topk(teacher_logits, k=self.top_k_loss, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        beta = self.beta
        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        if self.jsd_token_clip is not None and self.jsd_token_clip > 0:
            jsd = jsd.clamp(min=0, max=self.jsd_token_clip)

        return jsd.sum(dim=-1)

    def _compute_jsd_from_log_probs(self, student_log_probs, teacher_log_probs, apply_clip=True):
        """
        Compute per-token JSD from pre-computed log-probs (already temperature-scaled).

        Same as _compute_jsd_per_token but accepts log_softmax outputs directly,
        avoiding redundant log_softmax computation when log_probs are already available.

        Args:
            student_log_probs: log_softmax(logits / temperature), shape (batch_size, seq_len, vocab_size)
            teacher_log_probs: log_softmax(logits / temperature), shape (batch_size, seq_len, vocab_size)
            apply_clip: if True, clamp per-token JSD to [0, jsd_token_clip]

        Returns:
            per_token_jsd: shape (batch_size, seq_len), JSD summed over vocab dimension
        """
        beta = self.beta
        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

        if apply_clip and self.jsd_token_clip is not None and self.jsd_token_clip > 0:
            jsd = jsd.clamp(min=0, max=self.jsd_token_clip)

        jsd = torch.nan_to_num(jsd, nan=0.0, posinf=0.0, neginf=0.0)
        return jsd.sum(dim=-1)

    def _compute_raw_and_legacy_jsd_from_log_probs(
        self, student_log_probs, teacher_log_probs
    ):
        """Return raw and legacy JSD without changing legacy clip semantics.

        The repository's historical ``jsd_token_clip`` is applied to each
        vocabulary contribution before summation. RIFT-ASC also needs the raw
        summed divergence to set its detached batch threshold, so both token
        tensors are derived from the same contribution tensor here.
        """
        beta = self.beta
        if beta == 0:
            jsd = F.kl_div(
                student_log_probs, teacher_log_probs,
                reduction="none", log_target=True,
            )
        elif beta == 1:
            jsd = F.kl_div(
                teacher_log_probs, student_log_probs,
                reduction="none", log_target=True,
            )
        else:
            beta_t = torch.tensor(
                beta, dtype=student_log_probs.dtype,
                device=student_log_probs.device,
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        student_log_probs + torch.log1p(-beta_t),
                        teacher_log_probs + torch.log(beta_t),
                    ]
                ),
                dim=0,
            )
            kl_teacher = F.kl_div(
                mixture_log_probs, teacher_log_probs,
                reduction="none", log_target=True,
            )
            kl_student = F.kl_div(
                mixture_log_probs, student_log_probs,
                reduction="none", log_target=True,
            )
            jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

        jsd = torch.nan_to_num(jsd, nan=0.0, posinf=0.0, neginf=0.0)
        raw_token_jsd = jsd.sum(dim=-1)
        if self.jsd_token_clip is not None and self.jsd_token_clip > 0:
            legacy_token_jsd = jsd.clamp(
                min=0, max=self.jsd_token_clip
            ).sum(dim=-1)
        else:
            legacy_token_jsd = raw_token_jsd
        return raw_token_jsd, legacy_token_jsd

    def _entropy_gate_schedule_mix(self):
        """Return how much of the entropy gate to apply at the current training step."""
        if not self.use_entropy_gating or self.entropy_gate_schedule == "constant":
            return 1.0

        max_steps = max(int(getattr(self.args, "max_steps", 0) or 0), 1)
        global_step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        progress = min(max(global_step / max_steps, 0.0), 1.0)

        start = self.entropy_gate_schedule_start
        end = self.entropy_gate_schedule_end
        if progress <= start:
            return 1.0
        if progress >= end:
            return 0.0
        if end <= start:
            return 0.0

        phase = (progress - start) / (end - start)
        if self.entropy_gate_schedule == "phase_off":
            return 1.0 if progress < end else 0.0
        if self.entropy_gate_schedule == "cosine_decay":
            return 0.5 * (1.0 + math.cos(math.pi * phase))
        return 1.0 - phase

    def _compute_entropy_gate_from_log_probs(self, teacher_log_probs, mask):
        """
        Compute EAMS token weights from the privileged teacher's predictive entropy.

        The default inverse mode implements dynamic milestone selection:
        low entropy -> high confidence -> larger distillation weight.
        The gate is detached and can be combined with ReNIO sample weights.
        """
        if not self.use_entropy_gating:
            self._last_entropy_gate_mix = 0.0
            return torch.ones(mask.shape, device=mask.device, dtype=teacher_log_probs.dtype)

        with torch.no_grad():
            schedule_mix = float(self._entropy_gate_schedule_mix())
            self._last_entropy_gate_mix = schedule_mix
            if schedule_mix <= 0:
                return torch.ones(mask.shape, device=mask.device, dtype=teacher_log_probs.dtype)

            teacher_log_probs_fp32 = teacher_log_probs.float()
            teacher_probs = teacher_log_probs_fp32.exp()
            entropy = -(teacher_probs * teacher_log_probs_fp32).sum(dim=-1)
            del teacher_probs

            max_entropy = math.log(max(teacher_log_probs.shape[-1], 2))
            norm_entropy = (entropy / max_entropy).clamp(min=0.0, max=1.0)

            if self.entropy_gate_mode == "inverse":
                salience = 1.0 - norm_entropy
            else:
                salience = norm_entropy

            salience = salience.clamp(min=0.0, max=1.0).pow(self.entropy_gate_power)
            gate = self.entropy_gate_min + (self.entropy_gate_max - self.entropy_gate_min) * salience
            gate = gate.masked_fill(~mask, 0.0)

            if self.entropy_gate_normalize:
                valid_count = mask.sum().clamp(min=1).float()
                valid_mean = (gate.sum() / valid_count).clamp_min(self.entropy_gate_eps)
                gate = gate / valid_mean
                gate = gate.clamp(min=self.entropy_gate_min, max=self.entropy_gate_max)
                gate = gate.masked_fill(~mask, 0.0)

            if schedule_mix < 1.0:
                gate = 1.0 + schedule_mix * (gate - 1.0)
                gate = gate.masked_fill(~mask, 0.0)

            return gate.to(dtype=teacher_log_probs.dtype)

    def _compute_entropy_gate_from_logits(self, teacher_logits, mask):
        """Compute EAMS token weights when only raw teacher logits are available."""
        if not self.use_entropy_gating:
            return torch.ones(mask.shape, device=mask.device, dtype=teacher_logits.dtype)

        with torch.no_grad():
            logits = teacher_logits / self.temperature
            if self.top_k_loss is not None and self.top_k_loss > 0:
                k = min(self.top_k_loss, logits.shape[-1])
                logits, _ = torch.topk(logits, k=k, dim=-1)
            teacher_log_probs = F.log_softmax(logits, dim=-1)
            gate = self._compute_entropy_gate_from_log_probs(teacher_log_probs, mask)
            del teacher_log_probs
            return gate

    def _select_repr_aux_layer_indices(self, hidden_states):
        """Select the last fraction of transformer layers, excluding embeddings."""
        if hidden_states is None or len(hidden_states) <= 1:
            return []

        layer_indices = list(range(1, len(hidden_states)))
        keep_count = max(1, int(math.ceil(len(layer_indices) * self.repr_aux_layer_fraction)))
        return layer_indices[-keep_count:]

    def _compute_hidden_transition_repr_loss(
        self,
        student_hidden_states,
        teacher_hidden_states,
        student_prompt_len,
        teacher_prompt_len,
        mask,
    ):
        """
        Match hidden transition directions without touching the token JSD trunk.

        The auxiliary is:

            mean_l,t 1 - cos(h_s[l,t+1] - h_s[l,t], h_t[l,t+1] - h_t[l,t])

        Only completion positions are used, teacher states are detached, and a
        small uniform position subset is selected per sample to keep memory and
        compute bounded.
        """
        if not self.use_repr_aux or self.repr_aux_weight <= 0:
            zero = torch.zeros((), device=mask.device, dtype=torch.float32)
            return zero, 0, 0
        if student_hidden_states is None or teacher_hidden_states is None:
            zero = torch.zeros((), device=mask.device, dtype=torch.float32)
            return zero, 0, 0

        selected_layers = self._select_repr_aux_layer_indices(student_hidden_states)
        if not selected_layers:
            zero = torch.zeros((), device=mask.device, dtype=torch.float32)
            return zero, 0, 0

        transition_mask = mask[:, 1:] & mask[:, :-1]
        total_loss = None
        total_terms = 0
        selected_positions_total = 0

        for layer_idx in selected_layers:
            student_layer = student_hidden_states[layer_idx][:, student_prompt_len:, :]
            teacher_layer = teacher_hidden_states[layer_idx][:, teacher_prompt_len:, :].detach()

            seq_len = min(student_layer.shape[1], teacher_layer.shape[1], mask.shape[1])
            if seq_len < 2:
                continue

            student_layer = student_layer[:, :seq_len, :]
            teacher_layer = teacher_layer[:, :seq_len, :].to(dtype=student_layer.dtype)
            layer_transition_mask = transition_mask[:, : seq_len - 1]

            student_delta = student_layer[:, 1:, :] - student_layer[:, :-1, :]
            teacher_delta = teacher_layer[:, 1:, :] - teacher_layer[:, :-1, :]

            for batch_idx in range(student_delta.shape[0]):
                valid_positions = torch.nonzero(layer_transition_mask[batch_idx], as_tuple=False).flatten()
                if valid_positions.numel() == 0:
                    continue

                keep_count = min(self.repr_aux_position_count, int(valid_positions.numel()))
                if keep_count < valid_positions.numel():
                    pick = torch.linspace(
                        0,
                        valid_positions.numel() - 1,
                        steps=keep_count,
                        device=valid_positions.device,
                    ).long()
                    valid_positions = valid_positions[pick]

                s_delta = student_delta[batch_idx, valid_positions, :]
                t_delta = teacher_delta[batch_idx, valid_positions, :]
                cosine = F.cosine_similarity(s_delta, t_delta, dim=-1, eps=self.repr_aux_eps)
                layer_loss = (1.0 - cosine).mean()

                total_loss = layer_loss if total_loss is None else total_loss + layer_loss
                total_terms += 1
                selected_positions_total += int(valid_positions.numel())

        if total_loss is None or total_terms == 0:
            zero = torch.zeros((), device=mask.device, dtype=torch.float32)
            return zero, len(selected_layers), 0

        return total_loss / total_terms, len(selected_layers), selected_positions_total

    def _select_regap_decision_mask(self, scores, mask):
        """
        Select a small set of ReGap decision points per sample.

        Full counterfactual probing on every token is too expensive. ReGap-Lite
        uses a cheap branch-value trigger and keeps only a fixed fraction of
        valid generated tokens per sample.
        """
        with torch.no_grad():
            decision_mask = torch.zeros_like(mask, dtype=torch.bool)
            scores = scores.detach().float().masked_fill(~mask, float("-inf"))
            valid_counts = mask.sum(dim=-1)

            for batch_idx in range(mask.shape[0]):
                valid_count = int(valid_counts[batch_idx].item())
                if valid_count <= 0:
                    continue
                keep_count = int(math.ceil(valid_count * self.regap_decision_ratio))
                keep_count = min(valid_count, max(self.regap_min_decisions, keep_count))
                top_indices = torch.topk(scores[batch_idx], k=keep_count, dim=-1).indices
                decision_mask[batch_idx, top_indices] = True

            return decision_mask & mask

    def _compute_regap_lite_components(self, student_log_probs, teacher_log_probs, mask):
        """
        Compute ReGap-Lite branch losses from TopK student/teacher candidates.

        ReGap's full objective needs counterfactual continuations to estimate
        V_T(s, a) and V_S(s, a). This Lite implementation is the low-cost training
        hook: it approximates teacher rescueability and student competence with
        their normalized branch mass over the candidate set, then distills the
        rescue-gap distribution q(a) = softmax((V_T - V_S) / tau).

        Decision points are selected with a configurable proxy score:

            gap_weight * max_a positive(V_T - V_S)
          + disagreement_weight * TV(V_T, V_S)
          + student_entropy_weight * H_S

        This keeps entropy/disagreement as cheap probe triggers rather than the
        final branch-value target, matching the ReGap-Lite MVP design.
        """
        vocab_size = student_log_probs.shape[-1]
        top_k = min(self.regap_top_k, vocab_size)
        neg_large = torch.finfo(student_log_probs.dtype).min

        with torch.no_grad():
            student_top_ids = torch.topk(student_log_probs.detach(), k=top_k, dim=-1).indices
            teacher_top_ids = torch.topk(teacher_log_probs, k=top_k, dim=-1).indices
            candidate_ids = torch.cat([student_top_ids, teacher_top_ids], dim=-1)

            candidate_count = candidate_ids.shape[-1]
            candidate_pos = torch.arange(candidate_count, device=candidate_ids.device)
            same_candidate = candidate_ids.unsqueeze(-1) == candidate_ids.unsqueeze(-2)
            earlier_candidate = candidate_pos.view(1, 1, 1, -1) < candidate_pos.view(1, 1, -1, 1)
            duplicate_candidate = (same_candidate & earlier_candidate).any(dim=-1)
            candidate_mask = (~duplicate_candidate) & mask.unsqueeze(-1)

        student_candidate_log_probs = torch.gather(student_log_probs, dim=-1, index=candidate_ids)
        teacher_candidate_log_probs = torch.gather(teacher_log_probs, dim=-1, index=candidate_ids)
        student_candidate_log_probs = torch.nan_to_num(
            student_candidate_log_probs,
            nan=neg_large,
            posinf=0.0,
            neginf=neg_large,
        ).clamp(max=0.0)
        teacher_candidate_log_probs = torch.nan_to_num(
            teacher_candidate_log_probs,
            nan=neg_large,
            posinf=0.0,
            neginf=neg_large,
        ).clamp(max=0.0)

        student_candidate_masked = student_candidate_log_probs.masked_fill(~candidate_mask, neg_large)
        teacher_candidate_masked = teacher_candidate_log_probs.masked_fill(~candidate_mask, neg_large)

        student_candidate_log_norm = student_candidate_masked - torch.logsumexp(
            student_candidate_masked, dim=-1, keepdim=True
        )
        student_candidate_log_norm = torch.nan_to_num(
            student_candidate_log_norm,
            nan=0.0,
            posinf=0.0,
            neginf=neg_large,
        )

        with torch.no_grad():
            student_candidate_log_norm_detached = student_candidate_masked.detach() - torch.logsumexp(
                student_candidate_masked.detach(), dim=-1, keepdim=True
            )
            teacher_candidate_log_norm = teacher_candidate_masked - torch.logsumexp(
                teacher_candidate_masked, dim=-1, keepdim=True
            )
            student_candidate_log_norm_detached = torch.nan_to_num(
                student_candidate_log_norm_detached,
                nan=0.0,
                posinf=0.0,
                neginf=neg_large,
            )
            teacher_candidate_log_norm = torch.nan_to_num(
                teacher_candidate_log_norm,
                nan=0.0,
                posinf=0.0,
                neginf=neg_large,
            )

            student_value = student_candidate_log_norm_detached.exp().masked_fill(~candidate_mask, 0.0)
            teacher_value = teacher_candidate_log_norm.exp().masked_fill(~candidate_mask, 0.0)
            student_value = torch.nan_to_num(student_value, nan=0.0, posinf=0.0, neginf=0.0)
            teacher_value = torch.nan_to_num(teacher_value, nan=0.0, posinf=0.0, neginf=0.0)
            rescue_gap = teacher_value - student_value

            q_logits = (rescue_gap / self.regap_tau).masked_fill(~candidate_mask, neg_large)
            branch_target = F.softmax(q_logits, dim=-1).masked_fill(~candidate_mask, 0.0)
            branch_target = torch.nan_to_num(branch_target, nan=0.0, posinf=0.0, neginf=0.0)
            branch_target = branch_target / branch_target.sum(dim=-1, keepdim=True).clamp_min(self.regap_eps)

            teacher_branch = teacher_value / teacher_value.sum(dim=-1, keepdim=True).clamp_min(self.regap_eps)
            mixture = 0.5 * (branch_target + teacher_branch)
            js_teacher_value = 0.5 * (
                teacher_branch
                * (teacher_branch.clamp_min(self.regap_eps).log() - mixture.clamp_min(self.regap_eps).log())
                + branch_target
                * (branch_target.clamp_min(self.regap_eps).log() - mixture.clamp_min(self.regap_eps).log())
            ).sum(dim=-1)
            js_teacher_value = torch.nan_to_num(js_teacher_value, nan=0.0, posinf=0.0, neginf=0.0)
            agreement = (1.0 - js_teacher_value / math.log(2.0)).clamp(min=0.0, max=1.0)

            positive_rescue_gap = rescue_gap.clamp_min(0.0).max(dim=-1).values
            candidate_disagreement = 0.5 * (teacher_value - student_value).abs().sum(dim=-1)
            positive_rescue_gap = torch.nan_to_num(positive_rescue_gap, nan=0.0, posinf=0.0, neginf=0.0)
            candidate_disagreement = torch.nan_to_num(
                candidate_disagreement, nan=0.0, posinf=0.0, neginf=0.0
            )

            decision_scores = self.regap_gap_weight * positive_rescue_gap
            decision_scores = decision_scores + self.regap_disagreement_weight * candidate_disagreement
            if self.regap_student_entropy_weight > 0:
                student_log_probs_fp32 = student_log_probs.detach().float()
                student_probs = student_log_probs_fp32.exp()
                student_entropy = -(student_probs * student_log_probs_fp32).sum(dim=-1)
                student_entropy = student_entropy / math.log(max(vocab_size, 2))
                decision_scores = decision_scores + self.regap_student_entropy_weight * student_entropy
                del student_probs, student_log_probs_fp32, student_entropy
            decision_scores = torch.nan_to_num(decision_scores, nan=0.0, posinf=0.0, neginf=0.0)

            decision_mask = self._select_regap_decision_mask(decision_scores, mask)
            decision_float = decision_mask.float()

            dead_branch_mask = (
                (teacher_value < self.regap_dead_teacher_threshold)
                & (student_value > self.regap_dead_student_threshold)
                & candidate_mask
            )
            mean_rescue_gap = (
                (positive_rescue_gap * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            ).detach()
            mean_candidate_disagreement = (
                (candidate_disagreement * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            ).detach()
            mean_agreement = (
                (agreement * decision_float).sum() / decision_float.sum().clamp_min(1.0)
            ).detach()
            mean_pi_kl_weight = (
                (agreement * decision_float).sum() / mask.float().sum().clamp_min(1.0)
            ).detach()
            dead_branch_fraction = (
                (dead_branch_mask.float().sum(dim=-1).clamp(max=1.0) * decision_float).sum()
                / decision_float.sum().clamp_min(1.0)
            ).detach()
            decision_fraction = (
                decision_float.sum() / mask.float().sum().clamp_min(1.0)
            ).detach()

        # Fix: mask out -inf values before multiplication to avoid 0 * (-inf) = NaN
        student_candidate_log_norm_safe = student_candidate_log_norm.masked_fill(~candidate_mask, 0.0)
        regap_branch_ce = -(branch_target * student_candidate_log_norm_safe).sum(dim=-1)
        regap_branch_ce = torch.nan_to_num(regap_branch_ce, nan=0.0, posinf=0.0, neginf=0.0)
        student_candidate_probs = student_candidate_log_probs.exp().masked_fill(~candidate_mask, 0.0)
        student_candidate_probs = torch.nan_to_num(
            student_candidate_probs, nan=0.0, posinf=1.0 - self.regap_eps, neginf=0.0
        ).clamp(min=0.0, max=1.0 - self.regap_eps)
        dead_branch_terms = -torch.log1p(-student_candidate_probs).masked_fill(~dead_branch_mask, 0.0)
        dead_branch_loss = torch.nan_to_num(
            dead_branch_terms.sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0
        )

        decision_float = decision_mask.float()
        regap_branch_ce = regap_branch_ce * decision_float
        dead_branch_loss = dead_branch_loss * decision_float
        pi_kl_weight = agreement.to(dtype=student_log_probs.dtype) * decision_float

        stats = {
            "regap_mean_gap": mean_rescue_gap,
            "regap_candidate_disagreement": mean_candidate_disagreement,
            "regap_agreement": mean_agreement,
            "regap_pi_weight": mean_pi_kl_weight,
            "regap_dead_fraction": dead_branch_fraction,
            "regap_decision_fraction": decision_fraction,
        }
        return regap_branch_ce, pi_kl_weight, dead_branch_loss, decision_mask, stats

    def _compute_regap_sample_weights(self, student_log_probs, teacher_log_probs, mask):
        """
        Compute detached sample-level weights from ReGap diagnostics.

        This mode deliberately avoids branch CE. It keeps full OPSD as the only
        token-level objective and uses counterfactual rescue/suspicious gaps as
        reliability weights:

            w(x) = clip(1 + alpha * E[max(G, 0)] - beta * E[max(-G, 0)], lo, hi)

        where G is the teacher-vs-student branch mass gap over the selected
        candidate set at ReGap decision positions.
        """
        with torch.no_grad():
            vocab_size = student_log_probs.shape[-1]
            top_k = min(self.regap_top_k, vocab_size)
            neg_large = torch.finfo(student_log_probs.dtype).min

            student_top_ids = torch.topk(student_log_probs.detach(), k=top_k, dim=-1).indices
            teacher_top_ids = torch.topk(teacher_log_probs, k=top_k, dim=-1).indices
            candidate_ids = torch.cat([student_top_ids, teacher_top_ids], dim=-1)

            candidate_count = candidate_ids.shape[-1]
            candidate_pos = torch.arange(candidate_count, device=candidate_ids.device)
            same_candidate = candidate_ids.unsqueeze(-1) == candidate_ids.unsqueeze(-2)
            earlier_candidate = candidate_pos.view(1, 1, 1, -1) < candidate_pos.view(1, 1, -1, 1)
            duplicate_candidate = (same_candidate & earlier_candidate).any(dim=-1)
            candidate_mask = (~duplicate_candidate) & mask.unsqueeze(-1)

            student_candidate_log_probs = torch.gather(
                student_log_probs.detach(), dim=-1, index=candidate_ids
            )
            teacher_candidate_log_probs = torch.gather(teacher_log_probs, dim=-1, index=candidate_ids)
            student_candidate_log_probs = torch.nan_to_num(
                student_candidate_log_probs,
                nan=neg_large,
                posinf=0.0,
                neginf=neg_large,
            ).clamp(max=0.0)
            teacher_candidate_log_probs = torch.nan_to_num(
                teacher_candidate_log_probs,
                nan=neg_large,
                posinf=0.0,
                neginf=neg_large,
            ).clamp(max=0.0)

            student_candidate_masked = student_candidate_log_probs.masked_fill(~candidate_mask, neg_large)
            teacher_candidate_masked = teacher_candidate_log_probs.masked_fill(~candidate_mask, neg_large)
            student_candidate_log_norm = student_candidate_masked - torch.logsumexp(
                student_candidate_masked, dim=-1, keepdim=True
            )
            teacher_candidate_log_norm = teacher_candidate_masked - torch.logsumexp(
                teacher_candidate_masked, dim=-1, keepdim=True
            )
            student_candidate_log_norm = torch.nan_to_num(
                student_candidate_log_norm,
                nan=0.0,
                posinf=0.0,
                neginf=neg_large,
            )
            teacher_candidate_log_norm = torch.nan_to_num(
                teacher_candidate_log_norm,
                nan=0.0,
                posinf=0.0,
                neginf=neg_large,
            )

            student_value = student_candidate_log_norm.exp().masked_fill(~candidate_mask, 0.0)
            teacher_value = teacher_candidate_log_norm.exp().masked_fill(~candidate_mask, 0.0)
            student_value = torch.nan_to_num(student_value, nan=0.0, posinf=0.0, neginf=0.0)
            teacher_value = torch.nan_to_num(teacher_value, nan=0.0, posinf=0.0, neginf=0.0)
            rescue_gap = teacher_value - student_value

            positive_rescue_gap = rescue_gap.clamp_min(0.0).max(dim=-1).values
            candidate_disagreement = 0.5 * rescue_gap.abs().sum(dim=-1)
            positive_rescue_gap = torch.nan_to_num(positive_rescue_gap, nan=0.0, posinf=0.0, neginf=0.0)
            candidate_disagreement = torch.nan_to_num(
                candidate_disagreement, nan=0.0, posinf=0.0, neginf=0.0
            )

            decision_scores = self.regap_gap_weight * positive_rescue_gap
            decision_scores = decision_scores + self.regap_disagreement_weight * candidate_disagreement
            if self.regap_student_entropy_weight > 0:
                student_log_probs_fp32 = student_log_probs.detach().float()
                student_probs = student_log_probs_fp32.exp()
                student_entropy = -(student_probs * student_log_probs_fp32).sum(dim=-1)
                student_entropy = student_entropy / math.log(max(vocab_size, 2))
                decision_scores = decision_scores + self.regap_student_entropy_weight * student_entropy
                del student_probs, student_log_probs_fp32, student_entropy
            decision_scores = torch.nan_to_num(decision_scores, nan=0.0, posinf=0.0, neginf=0.0)

            decision_mask = self._select_regap_decision_mask(decision_scores, mask)
            decision_candidate_mask = decision_mask.unsqueeze(-1) & candidate_mask
            denom = decision_candidate_mask.float().sum(dim=(1, 2)).clamp_min(1.0)
            positive_score = (
                rescue_gap.clamp_min(0.0) * decision_candidate_mask.float()
            ).sum(dim=(1, 2)) / denom
            suspicious_score = (
                (-rescue_gap).clamp_min(0.0) * decision_candidate_mask.float()
            ).sum(dim=(1, 2)) / denom
            positive_score = torch.nan_to_num(positive_score, nan=0.0, posinf=0.0, neginf=0.0)
            suspicious_score = torch.nan_to_num(suspicious_score, nan=0.0, posinf=0.0, neginf=0.0)

            weights = 1.0 + self.regap_weight_alpha * positive_score
            weights = weights - self.regap_weight_beta * suspicious_score
            weights = weights.clamp(min=self.regap_weight_min, max=self.regap_weight_max)
            weights = torch.nan_to_num(weights, nan=1.0, posinf=self.regap_weight_max, neginf=self.regap_weight_min)

            decision_float = decision_mask.float()
            dead_branch_mask = (
                (teacher_value < self.regap_dead_teacher_threshold)
                & (student_value > self.regap_dead_student_threshold)
                & candidate_mask
            )
            stats = {
                "regap_weight_mean": weights.mean().detach(),
                "regap_weight_min": weights.min().detach(),
                "regap_weight_max": weights.max().detach(),
                "regap_positive_score": positive_score.mean().detach(),
                "regap_suspicious_score": suspicious_score.mean().detach(),
                "regap_mean_gap": (
                    (positive_rescue_gap * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
                ).detach(),
                "regap_candidate_disagreement": (
                    (candidate_disagreement * mask.float()).sum()
                    / mask.float().sum().clamp_min(1.0)
                ).detach(),
                "regap_decision_fraction": (
                    decision_float.sum() / mask.float().sum().clamp_min(1.0)
                ).detach(),
                "regap_dead_fraction": (
                    (dead_branch_mask.float().sum(dim=-1).clamp(max=1.0) * decision_float).sum()
                    / decision_float.sum().clamp_min(1.0)
                ).detach(),
            }
            return weights.to(dtype=student_log_probs.dtype), stats

    def _compute_sample_weights(self, student_token_lp, teacher_token_lp, labels, mask,
                                 student_logits=None, teacher_logits=None):
        """
        Compute ReNIO sample weights via fixed-threshold S/T log-ratio filtering.

        Args:
            student_token_lp: Pre-extracted log-probs of sampled tokens, shape (batch_size, seq_len).
                If None, computed from student_logits.
            teacher_token_lp: Pre-extracted log-probs of sampled tokens, shape (batch_size, seq_len).
                If None, computed from teacher_logits.
            labels: shape (batch_size, seq_len), -100 for padding
            mask: Boolean tensor of shape (batch_size, seq_len) indicating valid tokens
            student_logits: Raw logits (before temperature), shape (batch_size, seq_len, vocab_size).
                Only used when student_token_lp is None.
            teacher_logits: Raw logits (before temperature), shape (batch_size, seq_len, vocab_size).
                Only used when teacher_token_lp is None.

        Returns:
            sample_weights: Tensor of shape (batch_size,), detached from computation graph
        """
        if not self.use_renio:
            return torch.ones(labels.shape[0], device=labels.device)

        # Extract small tensors from raw logits if not provided
        if student_token_lp is None or teacher_token_lp is None:
            with torch.no_grad():
                student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
                teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
                token_indices = labels.clamp(0).unsqueeze(-1)
                student_token_lp = torch.gather(student_log_probs, dim=-1, index=token_indices).squeeze(-1).detach()
                teacher_token_lp = torch.gather(teacher_log_probs, dim=-1, index=token_indices).squeeze(-1)
                del student_log_probs, teacher_log_probs

        return self._compute_weights_ratio38(student_token_lp, teacher_token_lp, mask)

    def _compute_weights_ratio38(self, student_token_lp, teacher_token_lp, mask):
        """
        Sample weighting via S/T log-ratio with fixed threshold filtering.

        Args:
            student_token_lp: log P_S(x_t) for sampled tokens, shape (batch_size, seq_len)
            teacher_token_lp: log P_T(x_t) for sampled tokens, shape (batch_size, seq_len)
            mask: Boolean tensor of shape (batch_size, seq_len) indicating valid tokens

        Returns:
            effective_weights: shape (batch_size,), detached
        """
        with torch.no_grad():
            # log(P_S / P_T), clamped
            log_ratio = (student_token_lp - teacher_token_lp).clamp(max=self.kd_clamp)

            # Select tokens exceeding the fixed threshold, only among valid (non-padding) positions
            threshold_mask = (log_ratio >= self.imp_token_threshold) & mask  # (batch_size, seq_len)

            # Mean log-ratio over selected tokens; fall back to 0.0 (weight=1) if none selected
            selected_values = log_ratio.masked_fill(~threshold_mask, 0.0)
            selected_count = threshold_mask.sum(dim=-1).float().clamp(min=1)
            mean_log_ratio = selected_values.sum(dim=-1) / selected_count

            # exp with temperature
            effective_weights = torch.exp(mean_log_ratio / self.kd_sgo_tem)

            # Update global EMA statistics (before normalization)
            if self.weight_norm_type == "ema":
                batch_mean = effective_weights.mean().item()
                if self._global_weight_count == 0:
                    self._global_weight_mean = batch_mean
                else:
                    alpha = 0.1
                    self._global_weight_mean = alpha * batch_mean + (1 - alpha) * self._global_weight_mean
                self._global_weight_count += effective_weights.size(0)

            # Normalization
            if self.weight_norm_type == "batch_mean":
                effective_weights = effective_weights / effective_weights.mean()
                effective_weights = effective_weights.clamp(max=10)
            elif self.weight_norm_type == "ema":
                effective_weights = effective_weights / self._global_weight_mean
            elif self.weight_norm_type == "none":
                pass
            elif self.weight_norm_type == "clamp":
                effective_weights = effective_weights.clamp(min=0.1, max=10)
                effective_weights = effective_weights / effective_weights.mean()

        return effective_weights

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
        top_k=None,
        token_clip=None,
    ):
        """

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')
            top_k:
                If set, restricts the loss to only the top-k tokens of the teacher distribution. Both student and
                teacher distributions are renormalized over these k tokens before computing JSD. This reduces memory
                and focuses distillation on the teacher's most probable tokens. (default: None = full vocabulary)

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        if logits_are_probs:
            student_log_probs = torch.log(student_logits.clamp_min(1e-8))
            teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
        else:
            # Apply temperature scaling to logits before computing probabilities
            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature

            if top_k is not None and top_k > 0:
                # Restrict to top-k tokens of the teacher distribution and renormalize.
                # Shape: [batch, seq_len, top_k]
                _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
                student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
                teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

            # Compute log probabilities for student and probabilities for teacher
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student
        
        # Per-token clipping: cap each token's divergence value
        if token_clip is not None:
            jsd = jsd.clamp(max=token_clip)

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def _update_ema(self):
        """Update EMA parameters after an optimizer step.

        On the very first call this lazily initializes the EMA state as an exact copy of the
        current (trainable) model parameters, then returns without applying a decay step.
        Subsequent calls apply: ema = decay * ema + (1 - decay) * student.

        Only trainable parameters are tracked (i.e. LoRA adapter weights for PEFT models,
        or all parameters for full fine-tuning).

        ZeRO-3 note: with ZeRO-3 each rank only holds a shard of every parameter.
        We use `deepspeed.zero.GatheredParameters` (read-only, modifier_rank=None) so that
        every rank sees the full parameter tensor when snapshotting / updating the EMA.
        The EMA tensors are therefore full-sized copies, which is also required by
        `_ema_teacher_context` when it swaps the gathered student weights with EMA values.
        """
        decay = self.ema_decay
        unwrapped = self.accelerator.unwrap_model(self.model)

        # Detect ZeRO-3 (same pattern used elsewhere in this file)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            trainable = [(name, param) for name, param in unwrapped.named_parameters() if param.requires_grad]
            params_list = [p for _, p in trainable]

            # modifier_rank=None → read-only gather; original partitions are restored on exit.
            with deepspeed.zero.GatheredParameters(params_list):
                if self._ema_params is None:
                    self._ema_params = {name: param.data.clone().detach() for name, param in trainable}
                    n_tensors = len(self._ema_params)
                    n_params = sum(p.numel() for p in self._ema_params.values())
                    print(
                        f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                        f"(decay={decay})"
                    )
                    return  # first call = initialization only, no decay update

                for name, param in trainable:
                    if name not in self._ema_params:
                        continue
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    ema.mul_(decay).add_(param.data, alpha=1.0 - decay) # operate locally
        else:
            if self._ema_params is None:
                # Lazy init: snapshot the current weights as the initial EMA state.
                self._ema_params = {
                    name: param.data.clone().detach()
                    for name, param in unwrapped.named_parameters()
                    if param.requires_grad
                }
                n_tensors = len(self._ema_params)
                n_params = sum(p.numel() for p in self._ema_params.values())
                print(
                    f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                    f"(decay={decay})"
                )
                return  # first call = initialization only, no decay update

            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                # Move EMA buffer to the same device as the live param (handles multi-GPU setups)
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                ema.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @contextmanager
    def _ema_teacher_context(self, model):
        """Context manager that temporarily loads EMA weights for the teacher forward pass.

        Swaps `param.data` of every tracked (trainable) parameter with its EMA counterpart,
        runs the body (teacher forward), then restores the student weights unconditionally.
        Safe to use inside `torch.no_grad()`.  If EMA has not been initialized yet (step 0),
        this is a no-op and the current student weights are used instead.

        ZeRO-3 note: direct `param.data` assignment bypasses ZeRO-3's shard lifecycle and
        corrupts its internal state, causing size-mismatch errors during gradient-checkpoint
        recomputation.  When ZeRO-3 is active we therefore wrap the swap inside
        `deepspeed.zero.GatheredParameters` so the parameters are fully materialised on every
        rank before we touch them, and ZeRO-3 re-partitions cleanly when the context exits.
        """
        if self._ema_params is None:
            yield  # EMA not yet initialized; fall back to current weights
            return

        unwrapped = self.accelerator.unwrap_model(model)

        # Detect ZeRO-3 (same pattern used elsewhere in this file)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            name_to_param = {
                name: param
                for name, param in unwrapped.named_parameters()
                if param.requires_grad and name in self._ema_params
            }
            params_list = list(name_to_param.values())

            # modifier_rank=0 causes ZeRO-3 to re-partition from rank-0's param.data on exit,
            # which will be the restored student weights.
            with deepspeed.zero.GatheredParameters(params_list, modifier_rank=0):
                saved = {}
                for name, param in name_to_param.items():
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    saved[name] = param.data.clone()
                    param.data.copy_(ema)
                try:
                    yield
                finally:
                    for name, param in name_to_param.items():
                        if name in saved:
                            param.data.copy_(saved[name])
        else:
            saved = {}
            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                saved[name] = param.data
                param.data = ema
            try:
                yield
            finally:
                for name, param in unwrapped.named_parameters():
                    if name in saved:
                        param.data = saved[name]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the self-distillation loss with memory-efficient log-prob extraction.

        Memory optimization: Extract only needed log-probs immediately and free large tensors.
        """
        # Get batch-level prompt lengths
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        shifted_labels = inputs["labels"][:, student_prompt_len:]
        student_hidden_states = None
        teacher_hidden_states = None

        # === STUDENT FORWARD - Extract log-probs immediately ===
        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
            output_hidden_states=self.use_repr_aux,
        )
        if self.use_repr_aux:
            student_hidden_states = outputs_student.hidden_states

        # Clone the slice to release the full (batch, full_seq_len, vocab) output tensor immediately.
        # A plain slice is a *view* that keeps the entire output alive — for coding tasks
        # with long prompts this can waste 10+ GB of GPU memory per forward pass.
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :].contiguous()
        del outputs_student  # Full output can now be freed

        if self.use_thinking_machines_loss:
            # For reverse KL, we only need log-probs of sampled tokens
            sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            student_log_probs_sampled = torch.gather(
                student_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            del student_logits, student_log_probs, sampled_token_ids  # Free immediately!
        else:
            # For JSD, keep logits (temperature will be applied in generalized_jsd_loss)
            student_logits_for_loss = student_logits
            del student_logits

        # Free the full outputs (but keep reference for return_outputs if needed)
        if return_outputs:
            # Create a minimal output object to return (just the loss, no logits)
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()

        empty_cache()

        # === TEACHER FORWARD - Extract log-probs immediately ===
        # Choose teacher model based on mode:
        #   use_separate_teacher → use separate teacher model
        #   use_ema_teacher      → swap in EMA weights temporarily
        #   fixed_teacher        → disable LoRA adapters (base model = initial policy)
        #   default (dynamic)    → no-op, use current student weights
        if self.use_separate_teacher:
            # Use separate teacher model
            teacher_model_to_use = self.teacher_model
            adapter_context = nullcontext()
        elif self.use_ema_teacher:
            adapter_context = self._ema_teacher_context(model)
            teacher_model_to_use = model
        elif self.fixed_teacher and is_peft_model(model):
            adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
            teacher_model_to_use = model
        else:
            adapter_context = nullcontext()
            teacher_model_to_use = model

        with torch.no_grad(), adapter_context:
            outputs_teacher = teacher_model_to_use(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
                output_hidden_states=self.use_repr_aux,
            )
            if self.use_repr_aux:
                teacher_hidden_states = outputs_teacher.hidden_states

            # Clone the slice to release the full (batch, full_seq_len, vocab) output tensor.
            # Same rationale as the student side — a view would keep the entire teacher
            # output alive, which is especially costly with a separate 8B teacher model.
            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :].contiguous()
            del outputs_teacher  # Full output can now be freed

            # When teacher has a larger vocabulary than student (e.g., cross-size distillation),
            # the extra tokens are unused padding. Trim to match student vocab size.
            if teacher_logits.size(-1) > self.student_vocab_size:
                teacher_logits = teacher_logits[:, :, : self.student_vocab_size]

            if self.use_thinking_machines_loss:
                # For reverse KL, we only need log-probs of sampled tokens
                sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
                teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
                entropy_gate = self._compute_entropy_gate_from_log_probs(teacher_log_probs, shifted_labels != -100)
                teacher_log_probs_sampled = torch.gather(
                    teacher_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                del teacher_logits, teacher_log_probs, sampled_token_ids  # Free immediately!
            else:
                teacher_logits_for_loss = teacher_logits
                del teacher_logits

            empty_cache()

        # === COMPUTE LOSS with only small tensors ===
        if self.use_thinking_machines_loss:
            # Thinking Machines uses RL-style policy gradient:
            # Advantage = log π_teacher(x) - log π_student(x)
            # Loss = -E[Advantage * log π_student(x)]
            #
            # CRITICAL: advantage must be detached to prevent gradients flowing through it.
            # We want: ∇θ L = -E[A(x) * ∇θ log π_student(x)]
            # NOT: ∇θ L = -E[(T(x) - S(x)) * ∇θ S(x)] where both terms differentiate

            advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()

            # Apply masking before computing loss
            if shifted_labels is not None:
                mask = shifted_labels != -100
                advantage = advantage[mask]
                student_log_probs_sampled_masked = student_log_probs_sampled[mask]
                if self.use_entropy_gating:
                    entropy_gate = entropy_gate[mask]
            else:
                student_log_probs_sampled_masked = student_log_probs_sampled

            # Policy gradient loss: -advantage * log π_student
            # Negative because we minimize loss (gradient descent), but want to maximize reward
            if self.use_entropy_gating:
                loss = -(advantage * entropy_gate * student_log_probs_sampled_masked).mean()
            else:
                loss = -(advantage * student_log_probs_sampled_masked).mean()

            del (
                student_log_probs_sampled,
                teacher_log_probs_sampled,
                advantage,
                student_log_probs_sampled_masked,
            )
            if self.use_entropy_gating:
                del entropy_gate
        else:
            mask = shifted_labels != -100

            if self.top_k_loss is not None and self.top_k_loss > 0:
                # top_k_loss changes log_softmax semantics (subset normalization vs full),
                # so JSD and sample weight log_probs are not equivalent.
                # Fall back to independent computation for correctness.
                per_token_jsd = self._compute_jsd_per_token(student_logits_for_loss, teacher_logits_for_loss)

                mask_float = mask.float()
                entropy_gate = self._compute_entropy_gate_from_logits(teacher_logits_for_loss, mask)
                per_token_jsd = per_token_jsd * entropy_gate

                per_sample_jsd = (per_token_jsd * mask_float).sum(dim=-1) / mask_float.sum(dim=-1).clamp(min=1)
                sample_weights = self._compute_sample_weights(
                    None, None, shifted_labels, mask,
                    student_logits=student_logits_for_loss, teacher_logits=teacher_logits_for_loss
                )

                loss = (per_sample_jsd * sample_weights).mean()
                del student_logits_for_loss, teacher_logits_for_loss
            else:
                # Optimized path: compute log_softmax once, reuse for JSD, sample weights, and entropy.
                # This avoids redundant (B,S,V) allocations and allows early release of raw logits.
                # Note: student_log_probs must be outside no_grad to preserve gradients for JSD backprop.
                student_log_probs = F.log_softmax(student_logits_for_loss / self.temperature, dim=-1)
                student_log_probs = self._sanitize_log_probs_for_loss(
                    student_log_probs, "student_log_probs"
                )
                with torch.no_grad():
                    teacher_log_probs = F.log_softmax(teacher_logits_for_loss / self.temperature, dim=-1)
                    teacher_log_probs = self._sanitize_log_probs_for_loss(
                        teacher_log_probs, "teacher_log_probs"
                    )

                    # Extract small tensors for sample weights before releasing logits
                    if self.use_renio or self.use_rift_routing:
                        token_indices = shifted_labels.clamp(0).unsqueeze(-1)
                        student_token_lp = torch.gather(student_log_probs, dim=-1, index=token_indices).squeeze(-1).detach()
                        teacher_token_lp = torch.gather(teacher_log_probs, dim=-1, index=token_indices).squeeze(-1)

                # Release raw logits — they are no longer needed
                del student_logits_for_loss, teacher_logits_for_loss

                # Per-token JSD from pre-computed log_probs. RIFT-ASC needs the
                # pre-clip value only for its sparse soft-clamp branch; the legacy
                # hard-clipped tensor remains the routing/recovery input.
                if self.rift_asymmetric_soft_clamp:
                    (
                        per_token_jsd_raw,
                        per_token_jsd,
                    ) = self._compute_raw_and_legacy_jsd_from_log_probs(
                        student_log_probs, teacher_log_probs
                    )
                else:
                    per_token_jsd_raw = None
                    per_token_jsd = self._compute_jsd_from_log_probs(
                        student_log_probs, teacher_log_probs
                    )
                mask_float = mask.float()

                if self.use_rift_routing:
                    rift_privileged_jsd_mean = (
                        (per_token_jsd.detach() * mask_float).sum()
                        / mask_float.sum().clamp(min=1.0)
                    )
                    # Score the same sampled completion under the fixed teacher without
                    # privileged answer context.  This third pass is no-grad and keeps
                    # the original dense OPSD objective everywhere except detected forks.
                    with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
                        outputs_base_teacher = model(
                            input_ids=inputs["student_input_ids"],
                            attention_mask=inputs["student_attention_mask"],
                        )
                        base_teacher_logits = outputs_base_teacher.logits[
                            :, student_prompt_len - 1 : -1, :
                        ].contiguous()
                        del outputs_base_teacher
                        if base_teacher_logits.size(-1) > self.student_vocab_size:
                            base_teacher_logits = base_teacher_logits[:, :, : self.student_vocab_size]
                        base_teacher_log_probs = F.log_softmax(
                            base_teacher_logits / self.temperature, dim=-1
                        )
                        base_teacher_log_probs = self._sanitize_log_probs_for_loss(
                            base_teacher_log_probs, "rift_base_teacher_log_probs"
                        )
                        del base_teacher_logits

                        base_teacher_token_lp = torch.gather(
                            base_teacher_log_probs, dim=-1, index=token_indices
                        ).squeeze(-1)

                        # Normalized student entropy identifies genuine uncertainty forks.
                        student_probs_detached = student_log_probs.detach().float().exp()
                        student_entropy = -(
                            student_probs_detached * student_log_probs.detach().float()
                        ).sum(dim=-1)
                        student_entropy = student_entropy / math.log(max(self.student_vocab_size, 2))
                        del student_probs_detached

                    per_token_jsd_base = self._compute_jsd_from_log_probs(
                        student_log_probs, base_teacher_log_probs
                    )

                    with torch.no_grad():
                        privileged_advantage = teacher_token_lp - student_token_lp
                        base_advantage = base_teacher_token_lp - student_token_lp
                        sign_conflict = (
                            (privileged_advantage <= -self.rift_sign_margin)
                            & (base_advantage >= -self.rift_sign_margin)
                            & mask
                        )

                        entropy_thresholds = []
                        for batch_idx in range(student_entropy.shape[0]):
                            valid_entropy = student_entropy[batch_idx][mask[batch_idx]]
                            if valid_entropy.numel() == 0:
                                threshold = student_entropy.new_tensor(float("inf"))
                            else:
                                threshold = torch.quantile(
                                    valid_entropy, self.rift_entropy_quantile
                                )
                            entropy_thresholds.append(threshold)
                        entropy_thresholds = torch.stack(entropy_thresholds).unsqueeze(-1)
                        high_entropy = (student_entropy >= entropy_thresholds) & mask
                        candidate_mask = sign_conflict & high_entropy
                        if self.rift_hard_entropy_quantile >= 0:
                            hard_entropy_thresholds = []
                            for batch_idx in range(student_entropy.shape[0]):
                                valid_entropy = student_entropy[batch_idx][mask[batch_idx]]
                                if valid_entropy.numel() == 0:
                                    threshold = student_entropy.new_tensor(float("inf"))
                                else:
                                    threshold = torch.quantile(
                                        valid_entropy, self.rift_hard_entropy_quantile
                                    )
                                hard_entropy_thresholds.append(threshold)
                            hard_entropy_thresholds = torch.stack(hard_entropy_thresholds).unsqueeze(-1)
                            hard_entropy = (student_entropy >= hard_entropy_thresholds) & mask
                        else:
                            hard_entropy_thresholds = student_entropy.new_full(
                                (student_entropy.shape[0], 1), float("inf")
                            )
                            hard_entropy = torch.zeros_like(mask)
                        hard_candidate_mask = candidate_mask & hard_entropy
                        recovery_score = torch.zeros_like(per_token_jsd.detach())
                        ad_score = ad_risk_score(
                            student_token_lp,
                            teacher_token_lp,
                        ).to(recovery_score.dtype)
                        routing_score = recovery_score
                        recovery_eligible = mask
                        future_valid = torch.zeros_like(mask)
                        full_window_valid = torch.zeros_like(mask)
                        adaptive_candidate_mask = candidate_mask
                        route_calibration_audit = None
                        recovery_thresholds = recovery_score.new_full(
                            (recovery_score.shape[0],), self.rift_recovery_margin
                        )
                        hard_recovery_thresholds = recovery_score.new_full(
                            (recovery_score.shape[0],), float("inf")
                        )
                        if self.rift_recovery_window > 0:
                            privileged_jsd_detached = per_token_jsd.detach()
                            future_min = torch.full_like(privileged_jsd_detached, float("inf"))
                            sequence_length = privileged_jsd_detached.shape[1]
                            max_offset = min(self.rift_recovery_window, max(sequence_length - 1, 0))
                            for offset in range(1, max_offset + 1):
                                value_tail = torch.full_like(
                                    privileged_jsd_detached[:, :offset], float("inf")
                                )
                                valid_tail = torch.zeros_like(mask[:, :offset])
                                shifted_values = torch.cat(
                                    [privileged_jsd_detached[:, offset:], value_tail], dim=1
                                )
                                shifted_valid = torch.cat(
                                    [mask[:, offset:], valid_tail], dim=1
                                )
                                future_min = torch.minimum(
                                    future_min,
                                    torch.where(shifted_valid, shifted_values, future_min),
                                )
                                future_valid = future_valid | shifted_valid
                            if max_offset == self.rift_recovery_window:
                                full_window_valid[:, :-self.rift_recovery_window] = mask[
                                    :, self.rift_recovery_window:
                                ]
                            recovery_score = torch.where(
                                future_valid,
                                privileged_jsd_detached - future_min,
                                torch.zeros_like(privileged_jsd_detached),
                            )
                            routing_score = (
                                ad_score
                                if self.rift_routing_score == "ad_risk"
                                else recovery_score
                            )
                            window_eligible = (
                                full_window_valid
                                if self.rift_require_full_window
                                else future_valid
                            )
                            adaptive_candidate_mask = candidate_mask & window_eligible
                            if self.rift_recovery_quantile >= 0:
                                if self.rift_exact_rank:
                                    route_calibration_audit = routing_audit(
                                        adaptive_candidate_mask,
                                        routing_score,
                                        self.rift_recovery_quantile,
                                    )
                                    recovery_eligible = route_calibration_audit[
                                        "exact_mask"
                                    ]
                                    recovery_thresholds = route_calibration_audit[
                                        "exact_thresholds"
                                    ]
                                else:
                                    adaptive_eligible = []
                                    adaptive_thresholds = []
                                    adaptive_hard_thresholds = []
                                    for batch_idx in range(recovery_score.shape[0]):
                                        valid_candidates = adaptive_candidate_mask[batch_idx]
                                        if (
                                            self.rift_groupwise_recovery_quantiles
                                            and self.rift_hard_recovery_quantile >= 0
                                        ):
                                            valid_hard_candidates = (
                                                hard_candidate_mask[batch_idx]
                                                & window_eligible[batch_idx]
                                            )
                                            valid_regular_candidates = (
                                                valid_candidates
                                                & ~hard_candidate_mask[batch_idx]
                                            )
                                            candidate_scores = routing_score[batch_idx][
                                                valid_regular_candidates
                                            ].float()
                                            hard_candidate_scores = routing_score[batch_idx][
                                                valid_hard_candidates
                                            ].float()
                                        else:
                                            candidate_scores = routing_score[batch_idx][
                                                valid_candidates
                                            ].float()
                                            hard_candidate_scores = candidate_scores
                                        if candidate_scores.numel() == 0:
                                            threshold = candidate_scores.new_tensor(float("inf"))
                                        else:
                                            threshold = torch.quantile(
                                                candidate_scores,
                                                self.rift_recovery_quantile,
                                            )
                                        if self.rift_hard_recovery_quantile >= 0:
                                            if hard_candidate_scores.numel() == 0:
                                                hard_threshold = hard_candidate_scores.new_tensor(
                                                    float("inf")
                                                )
                                            else:
                                                hard_threshold = torch.quantile(
                                                    hard_candidate_scores,
                                                    self.rift_hard_recovery_quantile,
                                                )
                                        else:
                                            hard_threshold = threshold.new_tensor(float("inf"))
                                        adaptive_thresholds.append(
                                            threshold.to(recovery_score.dtype)
                                        )
                                        adaptive_hard_thresholds.append(
                                            hard_threshold.to(recovery_score.dtype)
                                        )
                                        if self.rift_hard_recovery_quantile >= 0:
                                            token_threshold = torch.where(
                                                hard_candidate_mask[batch_idx],
                                                hard_threshold.to(recovery_score.dtype),
                                                threshold.to(recovery_score.dtype),
                                            )
                                        else:
                                            token_threshold = threshold.to(recovery_score.dtype)
                                        adaptive_eligible.append(
                                            valid_candidates
                                            & (
                                                routing_score[batch_idx]
                                                >= token_threshold
                                            )
                                        )
                                    recovery_thresholds = torch.stack(adaptive_thresholds)
                                    hard_recovery_thresholds = torch.stack(
                                        adaptive_hard_thresholds
                                    )
                                    recovery_eligible = torch.stack(adaptive_eligible)
                                    if self.rift_hard_recovery_quantile < 0:
                                        route_calibration_audit = routing_audit(
                                            adaptive_candidate_mask,
                                            routing_score,
                                            self.rift_recovery_quantile,
                                        )
                            else:
                                recovery_eligible = (
                                    window_eligible
                                    & (recovery_score >= self.rift_recovery_margin)
                                    & mask
                                )
                        reference_route_mask = candidate_mask & recovery_eligible
                        route_mask = reference_route_mask
                        base_persistence_support = torch.zeros_like(
                            recovery_score, dtype=torch.long
                        )
                        fork_onset_mask = torch.zeros_like(candidate_mask)
                        fork_age = torch.zeros_like(candidate_mask, dtype=torch.long)
                        reflection_token_mask = torch.zeros_like(candidate_mask)
                        rs_protected_mask = torch.zeros_like(candidate_mask)
                        rs_soft_only_mask = torch.zeros_like(candidate_mask)
                        asc_protected_mask = torch.zeros_like(candidate_mask)
                        alc_protected_mask = torch.zeros_like(candidate_mask)
                        if self.rift_base_persistence_routing:
                            route_pool_mask = candidate_mask & future_valid
                            route_mask, base_persistence_support = (
                                self._select_rift_base_persistent_budget(
                                    candidate_mask,
                                    route_pool_mask,
                                    reference_route_mask,
                                    recovery_score,
                                    base_advantage,
                                    mask,
                                    self.rift_base_persistence_window,
                                    self.rift_sign_margin,
                                    self.rift_base_persistence_min_gain,
                                )
                            )
                        if self.rift_fork_onset_routing:
                            route_pool_mask = candidate_mask & future_valid
                            route_mask, fork_onset_mask, fork_age = self._frontload_rift_fork_budget(
                                candidate_mask,
                                route_pool_mask,
                                reference_route_mask,
                                recovery_score,
                                self.rift_fork_onset_gap,
                            )
                        if (
                            self.rift_reflection_safe_weighting
                            or self.rift_asymmetric_soft_clamp
                            or self.rift_asymmetric_log_compression
                        ):
                            fork_onset_mask = self._rift_fork_onset_mask(
                                candidate_mask, self.rift_fork_onset_gap
                            )
                            target_token_ids = inputs["student_input_ids"][
                                :, student_prompt_len:
                            ]
                            reflection_token_mask = self._match_rift_reflection_tokens(
                                target_token_ids,
                                mask,
                                self.rift_reflection_token_sequences,
                            )
                            boundary_mask = (
                                fork_onset_mask | reflection_token_mask
                            ) & mask
                            if self.rift_reflection_safe_weighting:
                                rs_protected_mask = boundary_mask
                                rs_soft_only_mask = rs_protected_mask & ~route_mask
                                route_gate = self._apply_rift_reflection_protection(
                                    route_mask,
                                    rs_protected_mask,
                                    self.rift_route_weight,
                                    self.rift_reflection_protection_weight,
                                    per_token_jsd.dtype,
                                )
                            elif self.rift_asymmetric_soft_clamp:
                                asc_protected_mask = boundary_mask
                                route_gate = (
                                    route_mask.to(per_token_jsd.dtype)
                                    * self.rift_route_weight
                                )
                            else:
                                alc_protected_mask = boundary_mask
                                route_gate = (
                                    route_mask.to(per_token_jsd.dtype)
                                    * self.rift_route_weight
                                )
                        else:
                            route_gate = (
                                route_mask.to(per_token_jsd.dtype) * self.rift_route_weight
                            )

                    privileged_jsd_for_mix = per_token_jsd
                    if self.rift_asymmetric_soft_clamp:
                        (
                            privileged_jsd_for_mix,
                            asc_active_mask,
                            asc_suppression_mask,
                            asc_threshold,
                            asc_scale,
                        ) = self._apply_rift_asymmetric_soft_clamp(
                            per_token_jsd,
                            per_token_jsd_raw,
                            privileged_advantage,
                            asc_protected_mask,
                            route_mask,
                            mask,
                            self.rift_soft_clamp_multiplier,
                            self.rift_sign_margin,
                            None,
                        )
                    elif self.rift_asymmetric_log_compression:
                        (
                            privileged_jsd_for_mix,
                            alc_active_mask,
                            alc_scale,
                        ) = self._apply_rift_asymmetric_log_compression(
                            per_token_jsd,
                            privileged_advantage,
                            alc_protected_mask,
                            route_mask,
                            mask,
                            self.rift_sign_margin,
                        )

                    per_token_jsd = (
                        (1.0 - route_gate) * privileged_jsd_for_mix
                        + route_gate * per_token_jsd_base
                    )
                    del privileged_jsd_for_mix, per_token_jsd_raw

                    valid_count = mask_float.sum().clamp(min=1.0)
                    mode = "train" if self.model.training else "eval"
                    self._metrics[mode]["rift_route_fraction"].append(
                        float(route_mask.float().sum() / valid_count)
                    )
                    candidate_count = candidate_mask.float().sum().clamp(min=1.0)
                    if route_calibration_audit is not None:
                        audit_candidate_counts = route_calibration_audit[
                            "candidate_counts"
                        ].float()
                        audit_target_counts = route_calibration_audit[
                            "target_counts"
                        ].float()
                        audit_exact_counts = route_calibration_audit[
                            "exact_counts"
                        ].float()
                        audit_threshold_counts = route_calibration_audit[
                            "threshold_counts"
                        ].float()
                        actual_counts = route_mask.sum(dim=-1).float()
                        audit_candidate_total = audit_candidate_counts.sum().clamp(
                            min=1.0
                        )
                        active_trajectories = (
                            audit_candidate_counts > 0
                        ).float().sum().clamp(min=1.0)
                        self._metrics[mode][
                            "rift_p0_candidate_count_mean"
                        ].append(
                            float(audit_candidate_counts.sum() / active_trajectories)
                        )
                        self._metrics[mode][
                            "rift_p0_target_route_count_mean"
                        ].append(
                            float(audit_target_counts.sum() / active_trajectories)
                        )
                        self._metrics[mode][
                            "rift_p0_actual_route_count_mean"
                        ].append(
                            float(actual_counts.sum() / active_trajectories)
                        )
                        self._metrics[mode][
                            "rift_p0_exact_route_count_mean"
                        ].append(
                            float(audit_exact_counts.sum() / active_trajectories)
                        )
                        self._metrics[mode][
                            "rift_p0_threshold_route_count_mean"
                        ].append(
                            float(audit_threshold_counts.sum() / active_trajectories)
                        )
                        self._metrics[mode]["rift_p0_boundary_tie_count_mean"].append(
                            float(
                                route_calibration_audit[
                                    "boundary_tie_counts"
                                ].float().sum()
                                / active_trajectories
                            )
                        )
                        self._metrics[mode]["rift_p0_tie_rate"].append(
                            float(
                                route_calibration_audit[
                                    "boundary_tie_counts"
                                ].float().sum()
                                / audit_candidate_total
                            )
                        )
                        self._metrics[mode]["rift_p0_tie_excess_mean"].append(
                            float(
                                route_calibration_audit[
                                    "threshold_tie_excess"
                                ].float().sum()
                                / active_trajectories
                            )
                        )
                        self._metrics[mode][
                            "rift_p0_threshold_budget_delta_mean"
                        ].append(
                            float(
                                route_calibration_audit[
                                    "threshold_budget_delta"
                                ].float().sum()
                                / active_trajectories
                            )
                        )
                        self._metrics[mode][
                            "rift_p0_threshold_budget_shortfall_mean"
                        ].append(
                            float(
                                route_calibration_audit[
                                    "threshold_budget_shortfall"
                                ].float().sum()
                                / active_trajectories
                            )
                        )
                        self._metrics[mode]["rift_p0_budget_error_mean_abs"].append(
                            float(
                                (actual_counts - audit_target_counts).abs().sum()
                                / active_trajectories
                            )
                        )
                        self._metrics[mode][
                            "rift_p0_threshold_exact_mask_diff_fraction"
                        ].append(
                            float(
                                route_calibration_audit[
                                    "mask_difference_counts"
                                ].float().sum()
                                / audit_candidate_total
                            )
                        )
                        self._metrics[mode]["rift_p0_route_mask_checksum53"].append(
                            float(mask_checksum53(route_mask))
                        )
                        self._metrics[mode][
                            "rift_p0_candidate_mask_checksum53"
                        ].append(float(mask_checksum53(adaptive_candidate_mask)))
                        self._metrics[mode][
                            "rift_p0_exact_mask_checksum53"
                        ].append(
                            float(route_calibration_audit["exact_checksum53"])
                        )
                        self._metrics[mode][
                            "rift_p0_threshold_mask_checksum53"
                        ].append(
                            float(route_calibration_audit["threshold_checksum53"])
                        )
                        sequence_positions = torch.arange(
                            route_mask.shape[1],
                            device=route_mask.device,
                            dtype=torch.float32,
                        ).unsqueeze(0)
                        normalized_positions = sequence_positions / max(
                            route_mask.shape[1] - 1,
                            1,
                        )
                        exact_mask = route_calibration_audit["exact_mask"]
                        threshold_mask = route_calibration_audit["threshold_mask"]
                        route_total = route_mask.float().sum().clamp(min=1.0)
                        self._metrics[mode]["rift_p0_route_position_mean"].append(
                            float(
                                (
                                    normalized_positions
                                    * route_mask.float()
                                ).sum()
                                / route_total
                            )
                        )
                        exact_total = exact_mask.float().sum().clamp(min=1.0)
                        threshold_total = threshold_mask.float().sum().clamp(min=1.0)
                        self._metrics[mode][
                            "rift_p0_exact_route_position_mean"
                        ].append(
                            float(
                                (
                                    normalized_positions
                                    * exact_mask.float()
                                ).sum()
                                / exact_total
                            )
                        )
                        self._metrics[mode][
                            "rift_p0_threshold_route_position_mean"
                        ].append(
                            float(
                                (
                                    normalized_positions
                                    * threshold_mask.float()
                                ).sum()
                                / threshold_total
                            )
                        )
                        for quartile_index in range(4):
                            lower = quartile_index / 4.0
                            upper = (quartile_index + 1) / 4.0
                            quartile_mask = normalized_positions >= lower
                            if quartile_index == 3:
                                quartile_mask &= normalized_positions <= upper
                            else:
                                quartile_mask &= normalized_positions < upper
                            self._metrics[mode][
                                f"rift_p0_route_position_q{quartile_index + 1}_share"
                            ].append(
                                float(
                                    (route_mask & quartile_mask).float().sum()
                                    / route_total
                                )
                            )
                            self._metrics[mode][
                                f"rift_p0_exact_route_position_q{quartile_index + 1}_share"
                            ].append(
                                float(
                                    (exact_mask & quartile_mask).float().sum()
                                    / exact_total
                                )
                            )
                            self._metrics[mode][
                                f"rift_p0_threshold_route_position_q{quartile_index + 1}_share"
                            ].append(
                                float(
                                    (threshold_mask & quartile_mask).float().sum()
                                    / threshold_total
                                )
                            )
                        incomplete_window_candidates = (
                            candidate_mask & ~full_window_valid
                        )
                        self._metrics[mode][
                            "rift_p0_near_eos_candidate_fraction"
                        ].append(
                            float(
                                incomplete_window_candidates.float().sum()
                                / candidate_count
                            )
                        )
                        self._metrics[mode]["rift_p0_near_eos_route_rate"].append(
                            float(
                                (
                                    route_mask
                                    & incomplete_window_candidates
                                ).float().sum()
                                / route_total
                            )
                        )
                        self._metrics[mode]["rift_p0_routing_score_mean"].append(
                            float(
                                (
                                    routing_score
                                    * adaptive_candidate_mask.float()
                                ).sum()
                                / audit_candidate_total
                            )
                        )
                    if (
                        self.rift_fork_onset_routing
                        or self.rift_reflection_safe_weighting
                        or self.rift_asymmetric_soft_clamp
                        or self.rift_asymmetric_log_compression
                        or self.rift_base_persistence_routing
                    ):
                        reference_route_count = reference_route_mask.float().sum().clamp(min=1.0)
                        route_count = route_mask.float().sum().clamp(min=1.0)
                        self._metrics[mode]["rift_reference_route_fraction"].append(
                            float(reference_route_mask.float().sum() / valid_count)
                        )
                    if self.rift_fork_onset_routing:
                        self._metrics[mode]["rift_fork_onset_fraction"].append(
                            float(fork_onset_mask.float().sum() / valid_count)
                        )
                        self._metrics[mode]["rift_fork_onset_share"].append(
                            float(fork_onset_mask.float().sum() / candidate_count)
                        )
                        self._metrics[mode]["rift_fork_route_onset_share"].append(
                            float((route_mask & fork_onset_mask).float().sum() / route_count)
                        )
                        self._metrics[mode]["rift_fork_route_swap_fraction"].append(
                            float((route_mask & ~reference_route_mask).float().sum() / route_count)
                        )
                        self._metrics[mode]["rift_fork_route_mean_age"].append(
                            float((fork_age.float() * route_mask.float()).sum() / route_count)
                        )
                        self._metrics[mode]["rift_reference_route_mean_age"].append(
                            float(
                                (fork_age.float() * reference_route_mask.float()).sum()
                                / reference_route_count
                            )
                        )
                    if self.rift_reflection_safe_weighting:
                        protected_count = rs_protected_mask.float().sum().clamp(min=1.0)
                        self._metrics[mode]["rift_rs_reflection_fraction"].append(
                            float(reflection_token_mask.float().sum() / valid_count)
                        )
                        self._metrics[mode]["rift_rs_onset_fraction"].append(
                            float(fork_onset_mask.float().sum() / valid_count)
                        )
                        self._metrics[mode]["rift_rs_protected_fraction"].append(
                            float(rs_protected_mask.float().sum() / valid_count)
                        )
                        self._metrics[mode]["rift_rs_soft_only_fraction"].append(
                            float(rs_soft_only_mask.float().sum() / valid_count)
                        )
                        self._metrics[mode]["rift_rs_protected_hard_route_share"].append(
                            float((rs_protected_mask & route_mask).float().sum() / protected_count)
                        )
                        self._metrics[mode]["rift_rs_effective_route_fraction"].append(
                            float(route_gate.float().sum() / valid_count)
                        )
                        self._metrics[mode]["rift_rs_extra_equivalent_route_fraction"].append(
                            float(
                                rs_soft_only_mask.float().sum()
                                * self.rift_reflection_protection_weight
                                / valid_count
                            )
                        )
                    if self.rift_asymmetric_soft_clamp:
                        active_count = asc_active_mask.float().sum()
                        active_denom = active_count.clamp(min=1.0)
                        self._metrics[mode]["rift_asc_reflection_fraction"].append(
                            float(reflection_token_mask.float().sum() / valid_count)
                        )
                        self._metrics[mode]["rift_asc_onset_fraction"].append(
                            float(fork_onset_mask.float().sum() / valid_count)
                        )
                        self._metrics[mode]["rift_asc_protected_fraction"].append(
                            float(asc_protected_mask.float().sum() / valid_count)
                        )
                        self._metrics[mode]["rift_asc_suppression_fraction"].append(
                            float(asc_suppression_mask.float().sum() / valid_count)
                        )
                        self._metrics[mode]["rift_asc_active_fraction"].append(
                            float(active_count / valid_count)
                        )
                        self._metrics[mode]["rift_asc_threshold"].append(
                            float(asc_threshold)
                        )
                        self._metrics[mode]["rift_asc_mean_active_scale"].append(
                            float(
                                (asc_scale * asc_active_mask.float()).sum()
                                / active_denom
                            )
                        )
                        del (
                            asc_active_mask,
                            asc_suppression_mask,
                            asc_threshold,
                            asc_scale,
                        )
                    if self.rift_asymmetric_log_compression:
                        alc_active_count = alc_active_mask.float().sum()
                        alc_active_denom = alc_active_count.clamp(min=1.0)
                        self._metrics[mode]["rift_alc_reflection_fraction"].append(
                            float(reflection_token_mask.float().sum() / valid_count)
                        )
                        self._metrics[mode]["rift_alc_onset_fraction"].append(
                            float(fork_onset_mask.float().sum() / valid_count)
                        )
                        self._metrics[mode]["rift_alc_protected_fraction"].append(
                            float(alc_protected_mask.float().sum() / valid_count)
                        )
                        self._metrics[mode]["rift_alc_active_fraction"].append(
                            float(alc_active_count / valid_count)
                        )
                        self._metrics[mode]["rift_alc_mean_active_scale"].append(
                            float(
                                (alc_scale * alc_active_mask.float()).sum()
                                / alc_active_denom
                            )
                        )
                        del alc_active_mask, alc_scale
                    if self.rift_base_persistence_routing:
                        route_count = route_mask.float().sum().clamp(min=1.0)
                        support_scale = float(self.rift_base_persistence_window)
                        incoming_swap_mask = route_mask & ~reference_route_mask
                        outgoing_swap_mask = reference_route_mask & ~route_mask
                        incoming_swap_count = incoming_swap_mask.float().sum()
                        self._metrics[mode]["rift_bsp_route_swap_fraction"].append(
                            float(incoming_swap_count / route_count)
                        )
                        self._metrics[mode]["rift_bsp_candidate_support_fraction"].append(
                            float(
                                (base_persistence_support.float() * candidate_mask.float()).sum()
                                / (candidate_count * support_scale)
                            )
                        )
                        self._metrics[mode]["rift_bsp_selected_support_fraction"].append(
                            float(
                                (base_persistence_support.float() * route_mask.float()).sum()
                                / (route_count * support_scale)
                            )
                        )
                        if self.rift_base_persistence_min_gain > 0:
                            support_gain = (
                                (
                                    base_persistence_support.float()
                                    * incoming_swap_mask.float()
                                ).sum()
                                - (
                                    base_persistence_support.float()
                                    * outgoing_swap_mask.float()
                                ).sum()
                            ) / incoming_swap_count.clamp(min=1.0)
                            self._metrics[mode]["rift_asg_route_count_difference"].append(
                                float((route_mask.sum() - reference_route_mask.sum()).abs())
                            )
                            self._metrics[mode]["rift_asg_incoming_swap_fraction"].append(
                                float(incoming_swap_count / route_count)
                            )
                            self._metrics[mode]["rift_asg_mean_support_gain"].append(
                                float(support_gain)
                            )
                    self._metrics[mode]["rift_candidate_fraction"].append(
                        float(candidate_mask.float().sum() / valid_count)
                    )
                    hard_candidate_count = hard_candidate_mask.float().sum().clamp(min=1.0)
                    regular_candidate_mask = candidate_mask & ~hard_candidate_mask
                    regular_candidate_count = regular_candidate_mask.float().sum().clamp(min=1.0)
                    self._metrics[mode]["rift_recovered_candidate_fraction"].append(
                        float((candidate_mask & recovery_eligible).float().sum() / candidate_count)
                    )
                    self._metrics[mode]["rift_hard_candidate_fraction"].append(
                        float(hard_candidate_mask.float().sum() / valid_count)
                    )
                    self._metrics[mode]["rift_hard_candidate_share"].append(
                        float(hard_candidate_mask.float().sum() / candidate_count)
                    )
                    self._metrics[mode]["rift_hard_recovered_candidate_fraction"].append(
                        float((hard_candidate_mask & recovery_eligible).float().sum() / hard_candidate_count)
                    )
                    self._metrics[mode]["rift_regular_recovered_candidate_fraction"].append(
                        float((regular_candidate_mask & recovery_eligible).float().sum() / regular_candidate_count)
                    )
                    self._metrics[mode]["rift_candidate_recovery_score"].append(
                        float((recovery_score * candidate_mask.float()).sum() / candidate_count)
                    )
                    finite_recovery_thresholds = recovery_thresholds[torch.isfinite(recovery_thresholds)]
                    if finite_recovery_thresholds.numel() > 0:
                        mean_recovery_threshold = finite_recovery_thresholds.float().mean()
                    else:
                        mean_recovery_threshold = recovery_score.new_tensor(0.0).float()
                    self._metrics[mode]["rift_recovery_threshold"].append(
                        float(mean_recovery_threshold)
                    )
                    finite_hard_thresholds = hard_recovery_thresholds[
                        torch.isfinite(hard_recovery_thresholds)
                    ]
                    if finite_hard_thresholds.numel() > 0:
                        mean_hard_threshold = finite_hard_thresholds.float().mean()
                    else:
                        mean_hard_threshold = recovery_score.new_tensor(0.0).float()
                    self._metrics[mode]["rift_hard_recovery_threshold"].append(
                        float(mean_hard_threshold)
                    )
                    candidate_recovery_values = recovery_score[candidate_mask].float()
                    if candidate_recovery_values.numel() > 0:
                        recovery_quantiles = torch.quantile(
                            candidate_recovery_values,
                            candidate_recovery_values.new_tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
                        )
                    else:
                        recovery_quantiles = recovery_score.new_zeros(5).float()
                    for quantile_name, quantile_value in zip(
                        ("min", "q25", "q50", "q75", "max"), recovery_quantiles
                    ):
                        self._metrics[mode][f"rift_recovery_{quantile_name}"].append(
                            float(quantile_value)
                        )
                    self._metrics[mode]["rift_sign_conflict_fraction"].append(
                        float(sign_conflict.float().sum() / valid_count)
                    )
                    self._metrics[mode]["rift_high_entropy_fraction"].append(
                        float(high_entropy.float().sum() / valid_count)
                    )
                    self._metrics[mode]["rift_privileged_advantage"].append(
                        float((privileged_advantage * mask_float).sum() / valid_count)
                    )
                    self._metrics[mode]["rift_base_advantage"].append(
                        float((base_advantage * mask_float).sum() / valid_count)
                    )
                    self._metrics[mode]["rift_privileged_jsd"].append(
                        float(rift_privileged_jsd_mean)
                    )
                    self._metrics[mode]["rift_base_jsd"].append(
                        float((per_token_jsd_base.detach() * mask_float).sum() / valid_count)
                    )

                    del (
                        base_teacher_log_probs,
                        base_teacher_token_lp,
                        per_token_jsd_base,
                        student_entropy,
                        privileged_advantage,
                        base_advantage,
                        sign_conflict,
                        high_entropy,
                        hard_entropy,
                        hard_entropy_thresholds,
                        candidate_mask,
                        hard_candidate_mask,
                        regular_candidate_mask,
                        candidate_recovery_values,
                        recovery_quantiles,
                        recovery_score,
                        ad_score,
                        routing_score,
                        recovery_thresholds,
                        hard_recovery_thresholds,
                        recovery_eligible,
                        future_valid,
                        full_window_valid,
                        adaptive_candidate_mask,
                        route_calibration_audit,
                        reference_route_mask,
                        route_mask,
                        route_gate,
                        fork_onset_mask,
                        fork_age,
                        reflection_token_mask,
                        rs_protected_mask,
                        rs_soft_only_mask,
                        asc_protected_mask,
                        alc_protected_mask,
                        base_persistence_support,
                        entropy_thresholds,
                        rift_privileged_jsd_mean,
                    )

                entropy_gate = self._compute_entropy_gate_from_log_probs(teacher_log_probs, mask)
                per_token_jsd_full = per_token_jsd * entropy_gate
                per_sample_jsd_full = (
                    (per_token_jsd_full * mask_float).sum(dim=-1)
                    / mask_float.sum(dim=-1).clamp(min=1)
                )
                regap_sample_weights = None

                if self.use_regap:
                    if self.regap_mode == "weighted":
                        regap_sample_weights, regap_stats = self._compute_regap_sample_weights(
                            student_log_probs, teacher_log_probs, mask
                        )
                        per_sample_jsd = per_sample_jsd_full

                        mode = "train" if self.model.training else "eval"
                        self._metrics[mode]["regap_opsd_full"].append(
                            float(per_sample_jsd_full.detach().mean())
                        )
                        self._metrics[mode]["regap_weight_mean"].append(
                            float(regap_stats["regap_weight_mean"])
                        )
                        self._metrics[mode]["regap_weight_min"].append(
                            float(regap_stats["regap_weight_min"])
                        )
                        self._metrics[mode]["regap_weight_max"].append(
                            float(regap_stats["regap_weight_max"])
                        )
                        self._metrics[mode]["regap_positive_score"].append(
                            float(regap_stats["regap_positive_score"])
                        )
                        self._metrics[mode]["regap_suspicious_score"].append(
                            float(regap_stats["regap_suspicious_score"])
                        )
                        self._metrics[mode]["regap_mean_gap"].append(
                            float(regap_stats["regap_mean_gap"])
                        )
                        self._metrics[mode]["regap_candidate_disagreement"].append(
                            float(regap_stats["regap_candidate_disagreement"])
                        )
                        self._metrics[mode]["regap_dead_fraction"].append(
                            float(regap_stats["regap_dead_fraction"])
                        )
                        self._metrics[mode]["regap_decision_fraction"].append(
                            float(regap_stats["regap_decision_fraction"])
                        )
                    else:
                        (
                            regap_branch_ce,
                            pi_kl_weight,
                            dead_branch_loss,
                            decision_mask,
                            regap_stats,
                        ) = self._compute_regap_lite_components(student_log_probs, teacher_log_probs, mask)

                        decision_float = decision_mask.float()
                        decision_count = decision_float.sum(dim=-1).clamp(min=1)
                        per_sample_branch = regap_branch_ce.sum(dim=-1) / decision_count
                        per_sample_pi_kl = (per_token_jsd * pi_kl_weight).sum(dim=-1) / decision_count
                        per_sample_dead = dead_branch_loss.sum(dim=-1) / decision_count
                        per_sample_regap = (
                            self.regap_branch_weight * per_sample_branch
                            + self.regap_lambda_pi * per_sample_pi_kl
                            + self.regap_eta_dead * per_sample_dead
                        )

                        if self.regap_mode == "additive":
                            per_sample_jsd = per_sample_jsd_full + per_sample_regap
                        else:
                            per_sample_jsd = per_sample_regap

                        mode = "train" if self.model.training else "eval"
                        self._metrics[mode]["regap_opsd_full"].append(float(per_sample_jsd_full.detach().mean()))
                        self._metrics[mode]["regap_branch_weight"].append(float(self.regap_branch_weight))
                        self._metrics[mode]["regap_branch_ce"].append(float(per_sample_branch.detach().mean()))
                        self._metrics[mode]["regap_pi_kl"].append(float(per_sample_pi_kl.detach().mean()))
                        self._metrics[mode]["regap_dead_loss"].append(float(per_sample_dead.detach().mean()))
                        self._metrics[mode]["regap_mean_gap"].append(float(regap_stats["regap_mean_gap"]))
                        self._metrics[mode]["regap_candidate_disagreement"].append(
                            float(regap_stats["regap_candidate_disagreement"])
                        )
                        self._metrics[mode]["regap_agreement"].append(float(regap_stats["regap_agreement"]))
                        self._metrics[mode]["regap_pi_weight"].append(float(regap_stats["regap_pi_weight"]))
                        self._metrics[mode]["regap_dead_fraction"].append(
                            float(regap_stats["regap_dead_fraction"])
                        )
                        self._metrics[mode]["regap_decision_fraction"].append(
                            float(regap_stats["regap_decision_fraction"])
                        )
                else:
                    per_sample_jsd = per_sample_jsd_full

                # Sample weights
                if self.use_renio:
                    sample_weights = self._compute_sample_weights(
                        student_token_lp, teacher_token_lp, shifted_labels, mask,
                    )
                else:
                    sample_weights = torch.ones(shifted_labels.shape[0], device=shifted_labels.device)
                if regap_sample_weights is not None:
                    sample_weights = sample_weights * regap_sample_weights.detach()

                loss = (per_sample_jsd * sample_weights).mean()
                del student_log_probs, teacher_log_probs

        if self.use_entropy_gating:
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["entropy_gate_mix"].append(float(self._last_entropy_gate_mix))

        if self.use_repr_aux and self.repr_aux_weight > 0:
            repr_aux_loss, repr_layers, repr_positions = self._compute_hidden_transition_repr_loss(
                student_hidden_states,
                teacher_hidden_states,
                student_prompt_len,
                teacher_prompt_len,
                shifted_labels != -100,
            )
            loss = loss + self.repr_aux_weight * repr_aux_loss.to(dtype=loss.dtype)

            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["repr_aux_loss"].append(float(repr_aux_loss.detach()))
            self._metrics[mode]["repr_aux_weight"].append(float(self.repr_aux_weight))
            self._metrics[mode]["repr_aux_layers"].append(float(repr_layers))
            self._metrics[mode]["repr_aux_positions"].append(float(repr_positions))

        del student_hidden_states, teacher_hidden_states

        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return (loss, minimal_output)
        else:
            return loss

    def generate_teacher_reasoning(
        self, model, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher's reasoning about the solution."""
        if self.use_vllm:
            # vLLM only supports using current student weights
            # Separate teacher and EMA teacher require different weights
            if self.use_separate_teacher or self.use_ema_teacher:
                raise NotImplementedError(
                    "vLLM acceleration is not supported for separate teacher or EMA teacher in reason_first mode. "
                    "Please set use_vllm=False."
                )
            # Use vLLM for fast reasoning generation
            return self._generate_teacher_reasoning_vllm(teacher_reasoning_prompts)
        else:
            # Use transformers generation (slower)
            with torch.no_grad():
                # Temporarily enable KV cache
                original_use_cache = model.config.use_cache
                original_gen_use_cache = self.reasoning_generation_config.use_cache

                model.config.use_cache = True
                self.reasoning_generation_config.use_cache = True

                # If fixed_teacher=True, disable LoRA adapters
                adapter_context = (
                    self.accelerator.unwrap_model(model).disable_adapter()
                    if self.fixed_teacher and is_peft_model(model)
                    else nullcontext()
                )

                try:
                    with adapter_context:
                        reasoning_outputs = model.generate(
                            input_ids=teacher_reasoning_prompts,
                            attention_mask=teacher_reasoning_attention_mask,
                            generation_config=self.reasoning_generation_config,
                            return_dict_in_generate=True,
                            use_cache=True,
                        )
                        reasoning_ids = reasoning_outputs.sequences
                finally:
                    model.config.use_cache = original_use_cache
                    self.reasoning_generation_config.use_cache = original_gen_use_cache

                return reasoning_ids

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs from student prompts only."""
        import time

        start_time = time.time()

        # Temporarily enable KV cache for generation if it was disabled for training
        original_use_cache = model.config.use_cache
        original_gen_use_cache = generation_config.use_cache

        model.config.use_cache = True
        generation_config.use_cache = True

        print(f"\n{'='*80}")
        print(f"GENERATION DEBUG INFO:")
        print(f"  Model dtype: {model.dtype}")
        print(f"  Model config use_cache: {model.config.use_cache}")
        print(f"  Attention implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")
        print(f"  Generation config use_cache: {generation_config.use_cache}")
        print(f"  Batch size: {inputs['student_prompts'].shape[0]}")
        print(f"  Prompt length: {inputs['student_prompts'].shape[1]}")
        print(f"  Max new tokens: {generation_config.max_new_tokens}")
        print(f"{'='*80}\n")

        # Generate output with respect to the student prompt only
        try:
            generated_outputs = model.generate(
                input_ids=inputs["student_prompts"],
                attention_mask=inputs.get("student_prompt_attention_mask", None),
                generation_config=generation_config,
                return_dict_in_generate=True,
                use_cache=True,
            )
            # Get the generated token IDs
            generated_tokens = generated_outputs.sequences
        finally:
            # Restore original settings
            model.config.use_cache = original_use_cache
            generation_config.use_cache = original_gen_use_cache

        elapsed_time = time.time() - start_time
        num_prompts = generated_tokens.shape[0]
        total_completion_tokens = generated_tokens.shape[1] - inputs["student_prompts"].shape[1]
        num_tokens = total_completion_tokens * num_prompts
        avg_completion_length = total_completion_tokens
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {num_tokens}, avg length: {avg_completion_length}, speed: {tokens_per_sec:.1f} tok/s"
        )

        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs from student prompts using vLLM.

        Note: In this version, student prompts are decoded with ``skip_special_tokens=True``
        before vLLM generation, so chat-template tokens are absent from the student's input
        while the teacher retains them. This means the student generates without chat-template
        formatting (e.g. no <think> mode, more non-thinking style generation). This was the
        setting used to produce the paper results — we found that a thinking teacher can still
        effectively supervise a "non-thinking" student, and this may improve learning efficiency
        since we mainly generate ~2k tokens and this mismatch avoids drastically reduced
        generation length, as a non-thinking teacher tends to supervise in a more concise manner. OPSD does not require student and teacher prompts to match, and both can be treated as optimizable components of the framework.
        """
        import time

        device = self.accelerator.device

        # Decode student prompts for vLLM (without special tokens - vLLM expects clean text)
        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=True,
        )
        # Remove padding token text if it appears, as vLLM expects clean prompts
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [
                p.replace(self.processing_class.pad_token, "") for p in prompts_text_for_vllm
            ]

        # Also decode prompts WITH special tokens for logging
        prompts_text_with_special = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )

        max_completion_length = generation_config.max_new_tokens
        temperature = generation_config.temperature
        # vLLM uses top_k=-1 for no top_k, transformers uses 0 or None.
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        # top_p, repetition_penalty, min_p, presence_penalty are not directly in generation_config, get from trainer args
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        repetition_penalty = self.args.repetition_penalty if hasattr(self.args, "repetition_penalty") else 1.0
        min_p = self.args.min_p if hasattr(self.args, "min_p") else 0.0
        presence_penalty = self.args.presence_penalty if hasattr(self.args, "presence_penalty") else 0.0
        trajectory_rollouts = self.trajectory_selection_rollouts

        # Start timing for vLLM generation
        start_time = time.time()

        if self.vllm_mode == "server":
            if trajectory_rollouts > 1:
                raise ValueError(
                    "trajectory_selection_rollouts > 1 is currently supported with vllm_mode='colocate' only."
                )
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,  # In GKD, we generate 1 completion per prompt from student
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    presence_penalty=presence_penalty,
                    guided_decoding_regex=self.vllm_guided_decoding_regex,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_for_vllm),
                (self.accelerator.process_index + 1) * len(prompts_text_for_vllm),
            )
            completion_ids = completion_ids[process_slice]
        elif self.vllm_mode == "colocate":
            if self.vllm_guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(
                    backend="outlines", regex=self.vllm_guided_decoding_regex
                )
            else:
                guided_decoding = None
            sampling_params = SamplingParams(
                n=trajectory_rollouts,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                presence_penalty=presence_penalty,
                guided_decoding=guided_decoding,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(
                    gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group
                )
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_id_groups = [
                [output.token_ids for output in request_output.outputs]
                for request_output in all_outputs
            ]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_id_groups = completion_id_groups[tp_slice]

            if trajectory_rollouts > 1:
                completion_ids = self._select_trajectory_candidates(
                    completion_id_groups,
                    inputs["answer_gt"],
                )
            else:
                completion_ids = [candidate_group[0] for candidate_group in completion_id_groups]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        # Calculate and print vLLM generation statistics
        elapsed_time = time.time() - start_time
        total_completion_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        avg_completion_length = total_completion_tokens / num_prompts if num_prompts > 0 else 0
        tokens_per_sec = total_completion_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"vLLM generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {total_completion_tokens}, avg length: {avg_completion_length:.1f}, speed: {tokens_per_sec:.1f} tok/s"
        )
        if trajectory_rollouts > 1:
            print(
                f"Trajectory selection kept 1 of {trajectory_rollouts} sampled completions per prompt "
                f"using mode={self.trajectory_selection_mode}."
            )

        # We need to combine prompt and completion for new_input_ids
        # Tokenize prompts again to get prompt_ids on the correct device and format
        # Use prompts_text_for_vllm (without special tokens) for tokenization since vLLM expects clean text
        # Ensure add_special_tokens=False as vLLM typically handles prompts as raw text
        # Calculate max_length for prompts, ensuring it's positive
        prompt_max_length = (
            max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        )
        prompt_tokenized = self.processing_class(
            prompts_text_for_vllm,
            return_tensors="pt",
            padding="longest",
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids
        prompt_ids = self._sanitize_token_ids_for_model(prompt_ids, "vllm_prompt_ids")

        padding_token_id = (
            pad_token_id if pad_token_id is not None else self.model_safe_pad_token_id
        )

        # Manually pad/truncate completions to max_completion_length length before using pad function
        padded_completion_ids_list = []
        sanitized_completion_ids = []
        for completion_idx, ids in enumerate(completion_ids):
            completion_tensor = torch.tensor(ids, device=device, dtype=torch.long)
            completion_tensor = self._sanitize_token_ids_for_model(
                completion_tensor, f"vllm_completion_ids[{completion_idx}]"
            )
            sanitized_completion_ids.append(completion_tensor.detach().cpu().tolist())

            if len(completion_tensor) > max_completion_length:
                # Truncate if longer than max_completion_length
                padded_completion_ids_list.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                # Pad if shorter than max_completion_length
                padding_needed = max_completion_length - len(completion_tensor)
                padded_tensor = torch.cat(
                    [
                        completion_tensor,
                        torch.full(
                            (padding_needed,), padding_token_id, device=device, dtype=completion_tensor.dtype
                        ),
                    ]
                )
                padded_completion_ids_list.append(padded_tensor)
            else:
                # Already the right length
                padded_completion_ids_list.append(completion_tensor)
        completion_ids = sanitized_completion_ids

        # Now all tensors are the same length, so we can stack them
        padded_completion_ids = torch.stack(padded_completion_ids_list)
        padded_completion_ids = self._sanitize_token_ids_for_model(
            padded_completion_ids, "vllm_padded_completion_ids"
        )

        # Ensure prompt_ids and padded_completion_ids are 2D
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)

        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        new_labels[new_labels == padding_token_id] = -100
        new_attention_mask[new_input_ids == padding_token_id] = 0
        if pad_token_id is not None and pad_token_id != padding_token_id:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        # Extract completion texts from the generated completion IDs
        completion_texts = []
        for comp_ids in completion_ids:
            completion_text = self.processing_class.decode(comp_ids, skip_special_tokens=False)
            completion_texts.append(completion_text)

        return new_input_ids, new_attention_mask, new_labels, prompts_text_with_special, completion_texts

    def _generate_teacher_reasoning_vllm(
        self, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher's reasoning using vLLM."""
        import time

        device = self.accelerator.device

        # Decode prompts for vLLM
        prompts_text = self.processing_class.batch_decode(
            teacher_reasoning_prompts,
            skip_special_tokens=True,
        )
        if self.processing_class.pad_token:
            prompts_text = [p.replace(self.processing_class.pad_token, "") for p in prompts_text]

        max_reasoning_length = self.reasoning_generation_config.max_new_tokens
        temperature = self.reasoning_generation_config.temperature
        top_k = (
            self.reasoning_generation_config.top_k
            if self.reasoning_generation_config.top_k and self.reasoning_generation_config.top_k > 0
            else -1
        )
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0

        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_reasoning_length,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text),
                (self.accelerator.process_index + 1) * len(prompts_text),
            )
            completion_ids = completion_ids[process_slice]

        elif self.vllm_mode == "colocate":
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_reasoning_length,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                orig_size = len(prompts_text)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.vllm_tp_group)
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)

        elapsed_time = time.time() - start_time
        total_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        print(
            f"vLLM teacher reasoning generation done - elapsed: {elapsed_time:.2f}s, prompts: {num_prompts}, tokens: {total_tokens}, speed: {total_tokens/elapsed_time:.1f} tok/s"
        )

        # Combine prompt + completion
        prompt_tokenized = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        padded_completions = pad(
            completion_ids_tensors, padding_value=self.processing_class.pad_token_id, padding_side="right"
        )

        reasoning_ids = torch.cat([prompt_ids, padded_completions], dim=1)

        return reasoning_ids

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with student vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            # recurse into the child
            self._sync_fsdp_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = (
                            self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                        )
                        llm_model.load_weights([(full_name, param.data)])

    def _move_model_to_vllm(self):
        """Synchronize student model weights to vLLM engine."""
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["weights"])

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                # use memory-efficient post-order traversal for FSDP
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                # For DeepSpeed ZeRO-3, gather each parameter individually like GRPO trainer
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

    def _wake_vllm_if_needed(self):
        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["kv_cache"])
    
    def _save_generation_outputs(self, step: int):
        """Save generation outputs to disk."""
        if not self.accelerator.is_main_process:
            return

        if len(self._generation_outputs_buffer) == 0:
            return

        import json
        from pathlib import Path

        # Create generations directory in output_dir
        generations_dir = Path(self.args.output_dir) / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        output_file = generations_dir / f"generations_step_{step}.json"

        output_data = {
            "step": step,
            "num_samples": len(self._generation_outputs_buffer),
            "generations": self._generation_outputs_buffer,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Saved {len(self._generation_outputs_buffer)} generation outputs to:")
        print(f"  {output_file}")
        print(f"{'='*80}\n")
        print(f"Ratio type {getattr(self, 'ratio_type', 'uniform')}")

        # Clear buffer after saving
        self._generation_outputs_buffer.clear()

    @profiling_decorator
    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Perform a training step with self-distillation.

        If reason_first=True:
        1. Generate teacher's reasoning about the solution
        2. Append reasoning to teacher prompt
        3. Generate completions from student prompts
        4. Compute JSD loss

        Otherwise:
        1. Generate completions from student prompts
        2. Construct full sequences for both student and teacher with the generation
        3. Compute JSD loss on the generation tokens
        """
        on_policy = True

        # === REASONING PHASE (if enabled) ===
        if self.reason_first:
            print(f"\n{'='*80}")
            print("REASONING PHASE: Teacher analyzing solution...")
            print(f"{'='*80}\n")

            # Choose teacher model for reasoning generation
            if self.use_separate_teacher:
                teacher_model_for_reasoning = self.teacher_model
            else:
                teacher_model_for_reasoning = model

            with unwrap_model_for_generation(teacher_model_for_reasoning, self.accelerator) as unwrapped_model:
                # Generate teacher's reasoning
                teacher_reasoning_ids = self.generate_teacher_reasoning(
                    unwrapped_model,
                    inputs["teacher_reasoning_prompts"],
                    inputs.get("teacher_reasoning_attention_mask"),
                )

                # Decode reasoning
                reasoning_prompt_len = inputs["teacher_reasoning_prompt_length"]
                reasoning_completions = teacher_reasoning_ids[:, reasoning_prompt_len:]
                reasoning_texts = self.processing_class.batch_decode(
                    reasoning_completions, skip_special_tokens=True
                )

                # Update teacher prompts with reasoning
                # Construct: [teacher_reasoning_prompt][reasoning][transition_to_teaching]
                teacher_prompts_with_reasoning = torch.cat(
                    [
                        inputs["teacher_reasoning_prompts"],
                        reasoning_completions,
                        inputs["teacher_transition_tokens"],
                    ],
                    dim=1,
                )

                # Update inputs with new teacher prompts
                inputs["teacher_prompts"] = teacher_prompts_with_reasoning
                teacher_attention_mask = torch.ones_like(teacher_prompts_with_reasoning)
                if self.processing_class.pad_token_id is not None:
                    teacher_attention_mask[
                        teacher_prompts_with_reasoning == self.processing_class.pad_token_id
                    ] = 0
                inputs["teacher_prompt_attention_mask"] = teacher_attention_mask
                inputs["teacher_prompt_length"] = teacher_prompts_with_reasoning.shape[1]

        # === GENERATION PHASE ===
        if self.use_vllm:
            self._wake_vllm_if_needed()
            result = self._generate_on_policy_outputs_vllm(
                inputs, self.generation_config, self.processing_class.pad_token_id
            )
            generated_ids, generated_attention_mask, _, prompt_texts, completion_texts = result
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                result = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
                generated_ids, generated_attention_mask, _ = result
                # Decode for logging
                prompt_texts = self.processing_class.batch_decode(
                    inputs["student_prompts"], skip_special_tokens=False
                )
                student_prompt_len = inputs["student_prompt_length"]
                completion_ids = generated_ids[:, student_prompt_len:]
                completion_texts = self.processing_class.batch_decode(
                    completion_ids, skip_special_tokens=False
                )

        generated_ids = self._sanitize_token_ids_for_model(generated_ids, "generated_ids")

        if self.use_vllm:
            # vLLM generation decodes the prompt text and tokenizes it again with
            # add_special_tokens=False. Use that actual prompt width for slicing
            # and loss masking; the original chat-template prompt length can differ.
            vllm_prompt_len = generated_ids.shape[1] - self.generation_config.max_new_tokens
            inputs["student_prompt_length"] = vllm_prompt_len
            prompt_token_ids = generated_ids[:, :vllm_prompt_len]
            if self.processing_class.pad_token_id is not None:
                prompt_lengths = (prompt_token_ids != self.processing_class.pad_token_id).sum(dim=-1)
            else:
                prompt_lengths = torch.full(
                    (generated_ids.shape[0],), vllm_prompt_len, device=generated_ids.device, dtype=torch.long
                )
            inputs["student_prompt_lengths_per_example"] = prompt_lengths

        # Get batch-level student prompt length
        student_prompt_len = inputs["student_prompt_length"]

        # Extract generation part (same slice for all examples since prompts are padded)
        generation_ids = generated_ids[:, student_prompt_len:]

        # Construct student full sequence: [student_prompt][generation]
        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask

        # Construct teacher full sequence: [teacher_prompt][generation]
        teacher_prompts = inputs["teacher_prompts"]
        teacher_full_ids = torch.cat([teacher_prompts, generation_ids], dim=1)
        teacher_full_ids = self._sanitize_token_ids_for_model(teacher_full_ids, "teacher_full_ids")

        # Create attention mask for teacher
        teacher_attention_mask = torch.ones_like(teacher_full_ids)
        if self.processing_class.pad_token_id is not None:
            teacher_attention_mask[teacher_full_ids == self.processing_class.pad_token_id] = 0
        teacher_attention_mask[teacher_full_ids == self.model_safe_pad_token_id] = 0

        inputs["teacher_input_ids"] = teacher_full_ids
        inputs["teacher_attention_mask"] = teacher_attention_mask

        # Create labels for generation tokens
        # Mask prompt tokens (use per-example lengths for accurate masking)
        labels = generated_ids.clone()
        for i in range(labels.shape[0]):
            actual_prompt_len = inputs["student_prompt_lengths_per_example"][i].item()
            labels[i, :actual_prompt_len] = -100  # Mask actual prompt

        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100
        labels[labels == self.model_safe_pad_token_id] = -100

        inputs["labels"] = labels
        
        # Collect generation outputs for saving
        for prompt, completion in zip(prompt_texts, completion_texts):
            self._generation_outputs_buffer.append(
                {"step": self.state.global_step, "prompt": prompt, "completion": completion}
            )

        loss = super().training_step(model, inputs, num_items_in_batch)

        # Release the large input tensors that were added to `inputs` during this step.
        # After backward the computation graph is freed, but the input dict still holds
        # references to GPU tensors (student_input_ids, teacher_input_ids, labels, etc.).
        # Explicitly deleting them here prevents transient memory spikes between steps,
        # which is critical when running a separate 8B teacher model with tight GPU budget.
        for key in [
            "student_input_ids", "student_attention_mask",
            "teacher_input_ids", "teacher_attention_mask",
            "labels",
        ]:
            inputs.pop(key, None)
        empty_cache()
        
        # Save generation outputs every N steps
        if (
            self.state.global_step > 0
            and self.state.global_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(self.state.global_step)

        loss_scalar = float(loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_equiv = 1.0 / ga

        if on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv
        return loss
    
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }

        if mode == "train":
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                ],
                dtype=torch.float64,
                device=device,
            )

            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            on_sum, off_sum, on_eq, off_eq = vec.tolist()

            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0

            if self.trajectory_selection_rollouts > 1:
                selection_vec = torch.tensor(
                    [
                        self._trajectory_selection_sums["samples"],
                        self._trajectory_selection_sums["candidate_count"],
                        self._trajectory_selection_sums["candidate_correct"],
                        self._trajectory_selection_sums["oracle_correct"],
                        self._trajectory_selection_sums["selected_correct"],
                        self._trajectory_selection_sums["formatted_candidates"],
                        self._trajectory_selection_sums["selected_consensus"],
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                if (
                    getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                    and dist.is_available()
                    and dist.is_initialized()
                ):
                    dist.all_reduce(selection_vec, op=dist.ReduceOp.SUM)

                (
                    selection_samples,
                    candidate_count,
                    candidate_correct,
                    oracle_correct,
                    selected_correct,
                    formatted_candidates,
                    selected_consensus,
                ) = selection_vec.tolist()
                if selection_samples > 0:
                    logs["trajsel_selected_correct_rate"] = round(selected_correct / selection_samples, 4)
                    logs["trajsel_oracle_correct_rate"] = round(oracle_correct / selection_samples, 4)
                    logs["trajsel_candidate_correct_rate"] = round(candidate_correct / candidate_count, 4)
                    logs["trajsel_formatted_candidate_rate"] = round(
                        formatted_candidates / candidate_count, 4
                    )
                    logs["trajsel_mean_selected_consensus"] = round(
                        selected_consensus / selection_samples, 4
                    )
                self._trajectory_selection_sums.clear()

        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        self._metrics[mode].clear()
        super().log(logs, start_time)
