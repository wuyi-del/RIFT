# RIFT: Future-Recovery Routing for Privileged On-Policy Self-Distillation

This repository accompanies **RIFT** (*Recovery-Informed Forked Training*) and **ReGap** (*Counterfactual Rescue Gap analysis*) for privileged on-policy self-distillation (OPSD) of reasoning models.

RIFT asks a token-level question that standard privileged distillation leaves open: when the privileged, solution-conditioned view disagrees with the student's sampled token, should training preserve the student's exploration or override it with privileged supervision? RIFT answers this using recovery on the **realized future student rollout**. Recoverable conflicts are trained toward an unprivileged reference distribution \(q^0\); persistent conflicts are trained toward the privileged reference distribution \(q^+\). A deterministic exact-rank rule keeps the routing budget exactly fixed within each trajectory.

The repository contains the submission-ready manuscript, compiled PDFs, vector figures, RIFT implementation, matched OPSD control, evaluation code, analysis scripts, reproducibility configurations, unit tests, and the aggregate tables reported in the paper.

For the full experimental protocol, frozen assets, hyperparameters, benchmark
denominators, statistical procedures, complete result tables, and reproduction
commands, see [**`EXPERIMENTS.md`**](EXPERIMENTS.md).

> **Release boundary.** Model weights, licensed benchmark copies, completion-level generations, raw server logs, machine configurations, and credentials are intentionally not distributed. All required public asset identifiers and reproduction assumptions are documented below and in [`code/DATASETS.md`](code/DATASETS.md).

## Overview

On a rollout \(y_{1:T} \sim p_\theta(\cdot\mid x)\), RIFT evaluates the same frozen initial policy in two contexts:

- **Unprivileged reference \(q^0\):** the problem without a verified answer.
- **Privileged reference \(q^+\):** the same problem with a verified solution.

At high-entropy sign conflicts, it measures the best reduction in divergence to the privileged view over a future window \(H\):

\[
\rho_t = d_t - \min_{1 \le k \le H} d_{t+k}, \qquad
d_t = \operatorname{JSD}\!\left(p_{\theta,t}\,\middle\|\,q_t^+\right).
\]

Within trajectory \(i\), candidates are sorted by decreasing \(\rho_t\), then by position and a fixed hash. With routing quantile \(q\), RIFT sends exactly

\[
k_i = \min\!\left\{n_i,\max\!\left\{1,\left\lceil(1-q)n_i\right\rceil\right\}\right\}
\]

of the \(n_i\) candidates to \(q^0\), and routes the remainder to \(q^+\). This makes the allocation deterministic, budget-faithful, and robust to tied scores. All routing is a training-time operation; inference uses only the trained student.

[Open the RIFT and ReGap overview figure (PDF)](paper/figures/rift_main_figure_ppt_outlined_cropped.pdf)

## Repository layout

```text
.
├── paper/
│   ├── RIFT_AAAI27_main.tex              # Main manuscript source
│   ├── RIFT_AAAI27_main_8pages.pdf        # Compiled main paper
│   ├── RIFT_AAAI27_supplement.tex/.pdf    # Technical supplement
│   ├── RIFT_AAAI27_refs.bib               # Bibliography
│   ├── RIFT_AAAI27_result_slots.tex       # Centralized result definitions
│   ├── aaai2027.sty                       # AAAI author-kit style
│   └── figures/                           # Main and analysis figures (PDF)
└── code/
    ├── opsd_train.py                      # Training entry point
    ├── opsd_trainer.py                    # On-policy distillation trainer
    ├── data_collator.py                   # Dual-context batch construction
    ├── rift_p0_routing.py                 # RIFT candidate and exact-rank logic
    ├── eval/                              # Math and code evaluators
    ├── analysis/                          # Signal analysis and figure scripts
    ├── scripts/                           # Training launch recipes
    ├── configs/                           # Conda, Accelerate, dependency pins
    ├── tables/                            # Aggregate paper result tables
    └── tests/                             # Routing unit tests
```

## Installation

The reference environment targets Linux with CUDA and multi-GPU training. The provided experiments use Qwen3 models, PyTorch 2.8, CUDA 12, Accelerate, TRL, DeepSpeed, vLLM, and `math-verify`.

```bash
git clone https://github.com/wuyi-del/RIFT.git
cd RIFT
conda env create -f code/configs/environment.yml
conda activate renio
```

`code/configs/DEPENDENCIES.txt` records the fuller frozen package inventory. For a fresh system, install the appropriate PyTorch/CUDA build first, then install the remaining packages compatible with that build.

## Public assets

Download the model and data yourself; they are not committed to this repository.

| Asset | Identifier / protocol |
|---|---|
| Student models | `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B` |
| Math training data | `siyanzhao/Openthoughts_math_30k_opsd` (frozen 29,434-row snapshot) |
| Code training data | Fixed 30,000-example subset of `open-thoughts/OpenThoughts-114k` |
| Math evaluation | Frozen AIME24, AIME25, MATH-500, AMC23, HMMT February 2025, HMMT November 2025 snapshots |
| Code evaluation | HumanEval+ and MBPP+ using EvalPlus v0.3.1 |
| Math grading | `math-verify==0.8.0` |

Place the Qwen3-4B checkpoint at `/models/Qwen3-4B`, or edit the model path in the launch script. Place the prepared training data at `data/openthoughts_math_30k`, or set `DATA=/path/to/data`. See [`code/DATASETS.md`](code/DATASETS.md) for the complete asset and evaluator description.

## Quick start

The provided scripts default to the budget-matched 50-update Qwen3-4B configuration. They default to eight GPUs, but `NGPU`, `GRAD`, `BATCH_SIZE`, `DATA`, `MAX_STEPS`, `SEED`, and output paths can be overridden as shell variables.

### Train RIFT

```bash
cd code
NGPU=8 \
SEED=42 \
MAX_STEPS=50 \
EXACT_RANK=1 \
RECOVERY_QUANTILE=0.25 \
DATA=/path/to/openthoughts_math_30k \
OUTPUT_DIR=/path/to/results/rift_seed42 \
bash scripts/run_rift_v2_4b.sh
```

The important RIFT controls are:

| Variable | Default | Meaning |
|---|---:|---|
| `EXACT_RANK` | `0` | Set to `1` to enforce an exact per-trajectory budget. |
| `RECOVERY_QUANTILE` | `-1` | Exact-rank quantile; use `0.25` for the q25 allocation. |
| `RECOVERY_WINDOW` | `32` | Future window \(H\) for recovery scoring. |
| `SIGN_MARGIN` | `0.05` | Candidate gate suppression margin. |
| `ENTROPY_QUANTILE` | `0.75` | Student-entropy candidate threshold. |
| `ROUTING_SCORE` | `future_recovery` | Score used to order eligible candidates. |
| `REQUIRE_FULL_WINDOW` | `0` | Restrict scoring to candidates with a complete look-ahead window. |

### Train the budget-matched OPSD control

```bash
cd code
NGPU=8 \
SEED=42 \
MAX_STEPS=50 \
DATA=/path/to/openthoughts_math_30k \
bash scripts/run_matched_opsd_4b.sh
```

The RIFT and matched-control scripts share the same base model, data, optimization setup, completion length, fixed-teacher setting, decoding configuration, and update budget. The former additionally enables `--use_rift_routing` and its routing parameters.

## Evaluation

`code/eval/evaluate_math.py` provides the mathematical evaluator with final-answer extraction and `math-verify` grading. Its command-line arguments include model path, LoRA adapter path, task selection, sampling seed, decoding parameters, and an optional output file for detailed predictions.

```bash
cd code
python eval/evaluate_math.py --help
python eval/evaluate_code.py --help
```

The reported mathematical evaluation uses twelve samples per problem (Avg@12). Code evaluation uses four completions per problem (Avg@4). Preserve the frozen benchmark snapshot, evaluator version, generation seed, and decoding configuration when making a direct comparison.

## Reproduce analyses and figures

```bash
cd code

# Validate the deterministic exact-rank selector.
pytest -q tests/test_rift_p0_routing.py

# Recompute recovery-signal analysis from an out-of-fold input table.
python analysis/p0_02_signal_analysis.py --help

# Build the result figure from the aggregate tables.
python analysis/make_rift_results_figure.py --help
```

The scripts require the appropriate evaluation records or out-of-fold feature tables as inputs. Those completion-level records are intentionally excluded; the aggregate values used in the paper are available under `code/tables/`.

## Reported aggregate results

The following tables are versioned together with the code and paper. They are aggregate statistics, not a replacement for the controlled evaluation protocol described above.

### Qwen3-4B 50-update triad (Avg@12)

| Seed | Matched OPSD | RIFT exact-rank q25 | Difference |
|---:|---:|---:|---:|
| 42 | 678 / 1080 (62.7778) | 689 / 1080 (63.7963) | +1.0185 pp |
| 43 | 654 / 1080 (60.5556) | 673 / 1080 (62.3148) | +1.7593 pp |
| 44 | 650 / 1080 (60.1852) | 676 / 1080 (62.5926) | +2.4074 pp |

### Recovery-signal analysis

| Predictor | AUPRC | AUROC | Δ AUPRC vs. current covariates |
|---|---:|---:|---:|
| Current covariates | 0.90649 | 0.62511 | — |
| Future recovery | 0.90980 | 0.62760 | +0.00331 |
| Future recovery + current | 0.91840 | 0.63200 | +0.01191 |

### Context-conditioned target arbitration

| Policy | Recoverable target | Persistent target | Avg@12 |
|---|---|---|---:|
| Uniform \(q^0\) | \(q^0\) | \(q^0\) | 63.0556 |
| Uniform \(q^+\) | \(q^+\) | \(q^+\) | 63.5185 |
| Reversed | \(q^+\) | \(q^0\) | 62.8704 |
| Proposed arbitration | \(q^0\) | \(q^+\) | **63.9815** |

For the extended 100-update six-benchmark comparison, 50-update cross-scale evaluation, routing ablations, ReGap estimates, and code-domain evaluation, see [`code/tables/`](code/tables/).

## ReGap: counterfactual rescue analysis

ReGap separates three quantities that should not be conflated:

1. **Privileged-context advantage:** compare different contexts while holding a fixed state-action pair \((s,a)\).
2. **Action advantage:** compare actions under a common prefix.
3. **Transferable recovery skill:** compare positive, matched-neutral, and vanilla arms on held-out states with a common-before difference-in-differences design.

ReGap reports branch NLL, recovery JSD, or verifier success as endpoints. It is an analysis protocol, not an inference-time component of RIFT.

## Reproducibility checklist

For each reported run, record at minimum:

- model checkpoint and tokenizer revision;
- training data snapshot and benchmark split/hash;
- training seed, evaluation seed, and sampling configuration;
- total updates, effective batch size, optimizer and learning-rate settings;
- candidate-gate parameters, future window, recovery quantile, and target map;
- exact-rank candidate count, selected count, budget error, tie excess, and checksum;
- evaluator/grader version and per-benchmark correct-count totals.

The manuscript, technical supplement, launcher defaults, and versioned CSV tables collectively document the configurations used for this release.

## Citation

The manuscript is currently provided as a submission/preprint artifact. If you use this repository, cite the paper title and link to this repository until a public bibliographic record is available:

```bibtex
@misc{rift2026,
  title        = {RIFT: Future-Recovery Routing for Privileged On-Policy Self-Distillation},
  author       = {Anonymous},
  year         = {2026},
  howpublished = {\url{https://github.com/wuyi-del/RIFT}}
}
```

## License

The implementation in [`code/`](code/) is released under the MIT License; see [`code/LICENSE`](code/LICENSE). Third-party models, datasets, benchmarks, and evaluation libraries remain subject to their own licenses and terms.

## Contact

Please use GitHub Issues for reproducibility questions, bug reports, or feature requests.
