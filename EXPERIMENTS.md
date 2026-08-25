# Complete Experimental Record

This document is the experiment-level companion to the paper source in
[`paper/`](paper/) and the reproducibility code in [`code/`](code/). It states
what was trained, what was evaluated, which comparisons are directly matched,
how uncertainty was computed, and which artifacts are intentionally absent
from this public release.

The document distinguishes three evaluation families that must not be pooled:

1. **Registered 50-update mathematical experiments.** These are the primary,
   budget-matched RIFT-versus-OPSD comparisons.
2. **Fixed-checkpoint 100-update experiments.** These use a longer shared
   training horizon and broader benchmark suite; they are reported separately
   from the 50-update study.
3. **Analysis and transfer experiments.** These include signal prediction,
   routing calibration, target arbitration, ReGap, code-domain replication,
   and the exploratory long-budget diagnostic.

## 1. Reproducibility scope

### Included in this repository

- Main paper and technical supplement source, compiled PDFs, bibliography,
  style file, and PDF figures.
- RIFT routing implementation, OPSD trainer, data collator, and launchers.
- Mathematical and code evaluators.
- Conda/Accelerate/dependency configurations.
- Unit test for deterministic exact-rank routing.
- Aggregate result tables used by the manuscript.

### Not included

- Qwen model weights and LoRA checkpoints.
- Licensed or redistributable benchmark copies.
- Training corpus files.
- Completion-level generations, prompts, or model logits.
- Server hostnames, mounting paths, job logs, credentials, and private keys.

The exclusions protect third-party licensing, data provenance, and operational
security. Consequently, an independent reproduction must obtain the public
models and datasets and rerun training/evaluation; the aggregate tables are
provided for verification against the reported targets.

## 2. Method configuration

### 2.1 RIFT target-routing protocol

For a current student rollout \(y_{1:T}\sim p_\theta(\cdot\mid x)\), the same
frozen initial policy is evaluated under two contexts:

- \(q_t^0\): no verified answer (unprivileged reference);
- \(q_t^+\): verified solution supplied (privileged reference).

At token \(t\), candidate selection requires all of the following:

\[
H(p_{\theta,t}) \ge Q_\eta(H),\qquad
a_t^+ \le -m,\qquad a_t^0 \ge -m,
\]

where \(a_t^v=\log q_t^v(y_t)-\log p_{\theta,t}(y_t)\), \(m\) is the
suppression margin, and \(Q_\eta(H)\) is the within-rollout student-entropy
quantile.

The future-recovery score is

\[
d_t=\operatorname{JSD}(p_{\theta,t}\,\|\,q_t^+),\qquad
\rho_t=d_t-\min_{1\le k\le H}d_{t+k}.
\]

Larger \(\rho_t\) indicates a larger later reduction in divergence to the
privileged reference. RIFT routes the most recoverable candidate tokens to
\(q^0\) and persistent tokens to \(q^+\).

For trajectory \(i\) with \(n_i\) candidate tokens and exact-rank quantile
\(q\), the number routed to \(q^0\) is

\[
k_i=\min\left\{n_i,\max\left\{1,\left\lceil(1-q)n_i\right\rceil\right\}\right\}
\quad\text{for }n_i>0.
\]

Candidates are ordered lexicographically by \((-\rho_t,\,\text{position},\,
\text{fixed hash})\). The selector is implemented in
[`code/rift_p0_routing.py`](code/rift_p0_routing.py) and audited through
candidate counts, selected counts, budget error, tie excess, and a stable
mask checksum.

### 2.2 Primary routing hyperparameters

| Hyperparameter | Value | Code control |
|---|---:|---|
| Student-entropy quantile \(\eta\) | 0.75 | `ENTROPY_QUANTILE` |
| Sign suppression margin \(m\) | 0.05 | `SIGN_MARGIN` |
| Recovery window \(H\) | 32 tokens | `RECOVERY_WINDOW` |
| Recovery margin | 0.005 | `RECOVERY_MARGIN` |
| Exact-rank quantile \(q\) | 0.25 | `EXACT_RANK=1`, `RECOVERY_QUANTILE=0.25` |
| Routing score | future recovery | `ROUTING_SCORE=future_recovery` |
| Routed-loss weight | 1.0 | `ROUTE_WEIGHT` |
| Full-window requirement | protocol-specific | `REQUIRE_FULL_WINDOW` |

## 3. Frozen assets and runtime

### 3.1 Primary mathematical configuration

| Item | Frozen identifier / version |
|---|---|
| Base model | `Qwen/Qwen3-4B` |
| Training corpus | `siyanzhao/Openthoughts_math_30k_opsd`, 29,434 rows |
| Base-model tree SHA-256 | `a74046c45e691429f9fad0d846d9054675d82604c5a0a6d49928a6155f7aa179` |
| Corpus Parquet SHA-256 | `0fa57ec6d7e5f4b40f85b4fdbbc1493e40c4d947f9192eabf589ecfb6e687dd2` |
| AIME24 (30 rows) SHA-256 | `77909a38a34db21b254cade51c6833aeb5e65e340b0e88dead5312d6dd7cf9b2` |
| AIME25 (30 rows) SHA-256 | `f5ba7f5de0a6a31a5ec4d56e8f7d310986f20658b3e38b7f579680c738479366` |
| HMMT Feb. 2025 (30 rows) SHA-256 | `4909c00ff08b46cc38add3ee5ba158123b19a5ac16adf51a4cc85faa528454fa` |
| Math evaluator SHA-256 | `cc824dcd4869519634846d88d858b52df68c46089b9d74446f4c12612b500d11` |
| Runtime | `math-verify==0.8.0`, `vllm==0.11.0`, `transformers==4.57.1` |

The public model/data identifiers and the extended benchmark list are also in
[`code/DATASETS.md`](code/DATASETS.md). The environment file is
[`code/configs/environment.yml`](code/configs/environment.yml).

### 3.2 Evaluator and decoding

The math evaluator (`code/eval/evaluate_math.py`) extracts the final
`\boxed{...}` answer using balanced-brace parsing. It wraps predicted and
reference answers in math delimiters, evaluates symbolic equivalence with
`math-verify`, and uses normalized exact string equality only if the symbolic
path raises an exception.

The registered mathematical evaluations use:

| Setting | Value |
|---|---:|
| Thinking mode | enabled |
| Temperature | 1.0 |
| Top-p | 0.95 |
| Top-k | disabled |
| Context limit | 40,960 tokens |
| Maximum new tokens | 38,912 |
| Samples per math problem | 12 (Avg@12) |

Each 30-problem benchmark therefore has 360 completions; the AIME24/AIME25/
HMMT25 triad has 1,080 completions per ordinary method. Random routing uses
three masks, hence 3,240 triad completions.

## 4. Registered 50-update mathematical study

### 4.1 Common training protocol

The primary Qwen3-4B comparison holds the following fixed across Matched OPSD
and RIFT:

| Setting | Value |
|---|---:|
| Optimizer updates | 50 |
| Learning rate | \(5\times10^{-6}\) |
| On-policy completion length | 1,024 tokens |
| Temperature | 1.1 |
| Top-p | 0.95 |
| Top-k | 20 |
| Attention implementation | SDPA |
| Precision | bfloat16 |
| Fine-tuning | LoRA, \(r=64\), \(\alpha=128\) |
| LoRA modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Teacher | fixed initial policy |
| Batch size per device | 1 |
| Gradient accumulation | 2 |
| Default process count | 8 |
| JSD token clip | 0.05 |

The launch recipes are:

- [`code/scripts/run_rift_v2_4b.sh`](code/scripts/run_rift_v2_4b.sh)
- [`code/scripts/run_matched_opsd_4b.sh`](code/scripts/run_matched_opsd_4b.sh)

### 4.2 Primary triad results

| Seed | Method | AIME24 | AIME25 | HMMT25 | Total correct | Avg@12 |
|---:|---|---:|---:|---:|---:|---:|
| 42 | Matched OPSD | 264/360 | 252/360 | 162/360 | 678/1,080 | 62.7778 |
| 42 | RIFT exact-rank q25 | 274/360 | 250/360 | 165/360 | **689/1,080** | **63.7963** |
| 43 | Matched OPSD | 263/360 | 234/360 | 157/360 | 654/1,080 | 60.5556 |
| 43 | RIFT exact-rank q25 | 275/360 | 240/360 | 158/360 | **673/1,080** | **62.3148** |
| 44 | Matched OPSD | 260/360 | 240/360 | 150/360 | 650/1,080 | 60.1852 |
| 44 | RIFT exact-rank q25 | 271/360 | 245/360 | 160/360 | **676/1,080** | **62.5926** |

### 4.3 Paired inference for the triad

| Comparison | Difference | 95% CI | p-value | Interpretation |
|---|---:|---|---:|---|
| RIFT − Matched, seed 42 | +1.0185 pp | [−0.8333, +2.8704] | 0.2810 | Point improvement; interval includes zero. |
| RIFT − Matched, seed 43 | +1.7593 pp | [+0.0926, +3.4259] | 0.0390 | Positive paired result. |
| RIFT − Matched, seeds 42/43 | +1.3889 pp | [+0.2315, +2.5463] | 0.0148 | Hierarchical seed/problem-cluster result. |
| RIFT − Matched, seed 44 | +2.4074 pp | [+0.3704, +4.4444] | 0.0216 | Registered independent comparison. |

The single-seed intervals are obtained by resampling problem clusters. The
42/43 result hierarchically resamples training seed and benchmark/problem
cluster. These intervals quantify evaluation variation; they do not turn the
three recorded seeds into a new independently sampled model family.

### 4.4 Routing and baseline controls at 50 updates

| Method | Seed | AIME24 | AIME25 | HMMT25 | Correct | Avg@12 |
|---|---:|---:|---:|---:|---:|---:|
| Current-JSD routing | 42 | 262/360 | 238/360 | 150/360 | 650/1,080 | 60.1852 |
| Random routing, three masks | 42 | 795/1,080 | 721/1,080 | 485/1,080 | 2,001/3,240 | 61.7593 |
| AD-OPSD | 42 | 268/360 | 237/360 | 148/360 | 653/1,080 | 60.4630 |
| DemoPSD | 42 | 259/360 | 241/360 | 157/360 | 657/1,080 | 60.8333 |
| Future recovery → \(q^0\) | 42 | 266/360 | 237/360 | 157/360 | 660/1,080 | 61.1111 |
| Future recovery → zero loss | 42 | 264/360 | 239/360 | 158/360 | 661/1,080 | 61.2037 |

The Random row aggregates three independently fixed route masks and is not
directly a 1,080-completion row. It must be compared using its aggregate
denominator or the per-mask records, not by treating 2,001/3,240 as one
ordinary seed-level result.

## 5. Fixed-checkpoint 100-update SixBench study

### 5.1 Design

The fixed-checkpoint study freezes the initial model for seeds 42–46 and
evaluates checkpoint 100. It uses the same Qwen3-4B family and frozen evaluator
but a broader SixBench suite:

| Benchmark | Problems | Completions at Avg@12 |
|---|---:|---:|
| AIME24 | 30 | 360 |
| AIME25 | 30 | 360 |
| MATH-500 | 500 | 6,000 |
| AMC23 | 40 | 480 |
| HMMT February 2025 | 30 | 360 |
| HMMT November 2025 | 30 | 360 |
| **Total** | **660** | **7,920** |

Macro is the unweighted mean of the six benchmark scores. The 100-update
results are not pooled with the 50-update triad because the optimization
horizon differs.

### 5.2 SixBench results

| Method | Updates | AIME24 | AIME25 | MATH-500 | AMC23 | HMMT-Feb | HMMT-Nov | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0 | 70.60 | 64.70 | 94.20 | 95.30 | 44.20 | 45.60 | 69.10 |
| SFT | 100 | 70.00 | 64.20 | 94.50 | 95.50 | 43.80 | 45.40 | 68.90 |
| GRPO | 100 | 72.60 | 66.70 | 94.90 | 95.90 | 43.20 | 50.50 | 70.63 |
| Matched OPSD | 100 | 73.40 | 67.20 | 95.10 | 96.10 | 43.80 | 50.45 | 71.01 |
| PHF | 100 | 74.35 | 68.15 | 95.28 | 96.55 | 44.95 | 51.65 | 71.82 |
| Purified OPSD | 100 | 74.55 | 68.35 | 95.31 | 96.70 | 45.20 | 51.95 | 72.01 |
| **RIFT** | **100** | **75.95** | **69.85** | **95.65** | **97.35** | **47.10** | **53.05** | **73.16** |

The aggregate table is
[`code/tables/sixbench_100update.csv`](code/tables/sixbench_100update.csv).

## 6. Cross-scale 50-update study

At every model scale, Matched OPSD and RIFT use the same 50-update budget, data
order, effective batch, rollout length, LoRA configuration, decoding protocol,
training-token budget, and routing hyperparameters. The six-benchmark macro
results are:

| Model | Seed | Matched OPSD macro | RIFT macro | Difference |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 42 | 52.67 | 54.43 | +1.76 pp |
| Qwen3-4B | 42 | 71.08 | 72.77 | +1.69 pp |
| Qwen3-8B | 42 | 74.32 | 76.45 | +2.13 pp |

The per-benchmark values are in
[`code/tables/cross_scale_50update.csv`](code/tables/cross_scale_50update.csv).
These scale checks use one training seed per listed scale and should be read as
matched robustness checks rather than multi-seed estimates at each scale.

## 7. Future-recovery signal study

### 7.1 Forced-continuation cohort

Signal analysis uses 512 out-of-fold candidate branches. The robust-recovery
positive class contains 442 branches (prevalence 0.86328), leaving 70 persistent
branches. This high prevalence makes raw AUPRC values large even for a random
predictor, so the persistent-class diagnostics and downstream routing ablation
are also reported.

### 7.2 Incremental prediction results

All deltas below are relative to current covariates only and use
problem-cluster resampling.

| Predictor | AUPRC | AUROC | Δ AUPRC vs. current | 95% CI | p-value |
|---|---:|---:|---:|---|---:|
| Random | 0.86559 | 0.50136 | −0.04090 | [−0.0568, −0.0251] | <0.0001 |
| Position only | 0.89885 | 0.59686 | −0.00764 | [−0.0158, +0.0004] | 0.063 |
| Current covariates only | 0.90649 | 0.62511 | — | — | — |
| Future recovery | 0.90980 | 0.62760 | +0.00331 | [−0.0012, +0.0078] | 0.149 |
| Future recovery + current | **0.91840** | **0.63200** | **+0.01191** | **[+0.0049, +0.0188]** | **0.0018** |

The point estimate for future recovery alone is positive but its increment over
current covariates is not statistically distinguished from zero in this test.
The combined feature set is the significant incremental predictor. The source
table is [`code/tables/recovery_signal.csv`](code/tables/recovery_signal.csv).

### 7.3 Prevalence-aware diagnostics

| Predictor | Persistent AP | Δ AP vs. current | AUROC | Δ AUROC vs. current |
|---|---:|---:|---:|---:|
| Current-only | 0.612 | — | 0.781 | — |
| True future | 0.648 | +0.036 | 0.806 | +0.025 |
| Future + current | 0.689 | +0.077 | 0.832 | +0.051 |

For persistent AP, Holm-adjusted p-values are 0.028 and 0.006 for true future
and future+current. For persistent-class AUROC, they are 0.031 and 0.004.
Additional calibration diagnostics: current-only / true future / future+current
have balanced accuracy 0.711 / 0.732 / 0.751, MCC 0.412 / 0.448 / 0.487, Brier
score 0.089 / 0.084 / 0.079, and ECE 0.031 / 0.028 / 0.023.

### 7.4 Temporal placebos

| Predictor | AUPRC | AUROC |
|---|---:|---:|
| Random | 0.86559 | 0.50136 |
| Past recovery | 0.84784 | 0.47875 |
| Future shuffled | 0.85951 | 0.47028 |
| True future recovery | 0.88674 | 0.47573 |

The temporal-placebo table is descriptive. The registered incremental test is
the current-covariates comparison above.

## 8. Exact-rank calibration and routing-score ablation

### 8.1 Calibration

| Calibration rule | All-token route fraction | Route CV | Tie excess |
|---|---:|---:|---:|
| Fixed threshold | 0.07997 | 0.05308 | 15.4588 |
| Global q25 | 0.06016 | 0.04957 | 3.0763 |
| Per-trajectory threshold q25 | 0.06260 | 0.04791 | 2.0363 |
| Exact-rank q25 | 0.06044 | **0.04276** | **0** |

Exact rank guarantees zero per-trajectory budget error by construction. Its
unit test is [`code/tests/test_rift_p0_routing.py`](code/tests/test_rift_p0_routing.py).

### 8.2 Downstream routing-score ablation

The candidate gate, token partition, exact-rank q25 budget, and target map are
held fixed; only the ranking score changes.

| Routing score | SixBench macro | Δ vs. current-only | 95% CI | Holm p-value |
|---|---:|---:|---|---:|
| Current-only | 71.0083 | — | — | — |
| Future shuffled | 70.8426 | −0.1657 | [−0.8124, +0.4513] | 0.612 |
| True future | 71.5412 | +0.5329 | [+0.1186, +0.9472] | 0.028 |
| Future + current | **71.7328** | **+0.7245** | **[+0.2964, +1.1526]** | **0.012** |

This is the downstream mechanism check that separates the score from the gate,
budget, and target mapping. The source table is
[`code/tables/routing_score_ablation.csv`](code/tables/routing_score_ablation.csv).

## 9. Context-conditioned target arbitration

All policies below share one fixed token partition and the same training
budget. The only difference is the target distribution assigned to recoverable
and persistent groups.

| Policy | Recoverable target | Persistent target | Correct | Avg@12 |
|---|---|---|---:|---:|
| Uniform \(q^0\) | \(q^0\) | \(q^0\) | 681/1,080 | 63.0556 |
| Uniform \(q^+\) | \(q^+\) | \(q^+\) | 686/1,080 | 63.5185 |
| Reversed | \(q^+\) | \(q^0\) | 679/1,080 | 62.8704 |
| **Proposed arbitration** | \(q^0\) | \(q^+\) | **691/1,080** | **63.9815** |

| Paired contrast | Difference | 95% CI | Raw p | Holm p |
|---|---:|---|---:|---:|
| Proposed − Uniform \(q^0\) | +0.9259 pp | [+0.1852, +1.6667] | 0.0140 | 0.0280 |
| Proposed − Uniform \(q^+\) | +0.4630 pp | [+0.0463, +0.8796] | 0.0320 | 0.0320 |
| Proposed − Reversed | +1.1111 pp | [+0.3241, +1.8981] | 0.0060 | 0.0180 |

The target-arbitration aggregate table is
[`code/tables/target_arbitration.csv`](code/tables/target_arbitration.csv).

## 10. ReGap counterfactual transfer study

### 10.1 Design

ReGap distinguishes three estimands:

1. **Privileged-context advantage:** compare context variants while holding a
   state-action pair \((s,a)\) fixed.
2. **Action advantage:** compare actions from the same unprivileged prefix.
3. **Transferable recovery skill:** compare positive, matched-neutral, and
   vanilla arms on held-out states with a common-before difference-in-
   differences design.

The transfer study uses three training seeds. Each seed has 24 training
problems and 24 held-out problems. Each condition uses four fixed continuation
seeds, yielding 2,048 continuations for the 512-branch signal cohort. Endpoints
are Branch NLL, recovery JSD, and/or verifier success. Lower NLL and recovery
JSD are better.

### 10.2 Pooled difference-in-differences

| Contrast | Endpoint | Pooled DiD | 95% cluster CI | p-value | Hierarchical 95% CI |
|---|---|---:|---|---:|---|
| Positive − Neutral | Branch NLL | −1.3606e−4 | [−2.6319e−4, −1.4391e−5] | 0.0285 | [−2.9048e−4, −1.8425e−5] |
| Positive − Neutral | Recovery JSD | −2.2423e−5 | [−3.2214e−5, −1.3226e−5] | <0.0001 | [−3.7862e−5, −1.1034e−5] |
| Positive − Vanilla | Branch NLL | −2.0716e−4 | [−5.0225e−4, +1.0112e−4] | 0.1826 | [−5.4861e−4, +1.7958e−4] |
| Positive − Vanilla | Recovery JSD | −3.3739e−5 | [−4.6554e−5, −2.1884e−5] | <0.0001 | [−5.0514e−5, −2.0097e−5] |

The aggregate source is [`code/tables/regap_pooled.csv`](code/tables/regap_pooled.csv).

## 11. Code-domain replication

The code-domain study trains Qwen3-4B for 100 updates on a fixed 30,000-example
OpenThoughts Coding subset. Base, Matched OPSD, ReNIO, and RIFT share training
seed 42, a fixed evaluator protocol, and checkpoint 100. HumanEval+ and MBPP+
use four completions per problem (Avg@4).

| Method | Updates | HumanEval+ Avg@4 | MBPP+ Avg@4 | Code macro | Δ vs. Matched | 95% CI | Raw p |
|---|---:|---:|---:|---:|---:|---|---:|
| Base | 0 | 86.74 | 77.31 | 82.03 | — | — | — |
| Matched OPSD | 100 | 86.31 | 77.46 | 81.89 | — | — | — |
| ReNIO | 100 | 86.62 | 77.74 | 82.18 | +0.29 | [−0.18, +0.77] | 0.221 |
| **RIFT** | **100** | **86.95** | **78.02** | **82.49** | **+0.60** | **[+0.08, +1.11]** | **0.026** |

Problem-cluster intervals in this study quantify problem and generation
uncertainty, not training-seed variation. See
[`code/tables/code_replication.csv`](code/tables/code_replication.csv).

## 12. Exploratory long-budget diagnostic

This diagnostic uses 90 problems and one generation per problem. It is
descriptive and is not part of the primary statistical claim.

| Method | 4k Pass@1 | 8k Pass@1 | 16k Pass@1 |
|---|---:|---:|---:|
| Base | 4/90 (4.44%) | 20/90 (22.22%) | 43/90 (47.78%) |
| Continued OPSD | 2/90 (2.22%) | 18/90 (20.00%) | 41/90 (45.56%) |
| AD-risk-only | 2/90 (2.22%) | 20/90 (22.22%) | 43/90 (47.78%) |
| RIFT | 6/90 (6.67%) | 26/90 (28.89%) | 48/90 (53.33%) |

## 13. Statistical protocol

| Analysis family | Unit of resampling | Reported statistics |
|---|---|---|
| Single-seed mathematical comparison | Problem cluster | Paired bootstrap 95% CI and two-sided p-value |
| Multi-seed mathematical summary | Training seed + problem cluster | Hierarchical 95% CI and pooled comparison |
| Signal prediction | Problem cluster | OOF AUPRC/AUROC increment, bootstrap CI, p-value |
| Target arbitration | Problem cluster | Paired CI, raw p-value, Holm-adjusted p-value |
| Routing-score ablation | Problem cluster | Difference vs. current-only, CI, Holm p-value |
| ReGap | Seed + branch/problem hierarchy | Pooled DiD, cluster CI, p-value, hierarchical CI |
| Code-domain replication | Problem cluster | Paired CI and p-value |

All primary comparisons preserve the method-specific inference family. Do not
combine rows across different update horizons, benchmarks, model sizes, or
training corpora into one unreported pooled statistic.

## 14. Commands and verification

### 14.1 RIFT 50-update run

```bash
cd code
NGPU=8 SEED=42 MAX_STEPS=50 EXACT_RANK=1 RECOVERY_QUANTILE=0.25 \
DATA=/path/to/openthoughts_math_30k \
OUTPUT_DIR=/path/to/results/rift_seed42 \
bash scripts/run_rift_v2_4b.sh
```

### 14.2 Matched control

```bash
cd code
NGPU=8 SEED=42 MAX_STEPS=50 \
DATA=/path/to/openthoughts_math_30k \
bash scripts/run_matched_opsd_4b.sh
```

### 14.3 Determinism test

```bash
cd code
pytest -q tests/test_rift_p0_routing.py
```

### 14.4 Evaluation and analysis entry points

```bash
cd code
python eval/evaluate_math.py --help
python eval/evaluate_code.py --help
python analysis/p0_02_signal_analysis.py --help
python analysis/make_rift_results_figure.py --help
```

## 15. File-to-result map

| Result family | Primary repository artifact |
|---|---|
| 50-update triad | `code/tables/triad_50update.csv` |
| Cross-scale checks | `code/tables/cross_scale_50update.csv` |
| 100-update SixBench | `code/tables/sixbench_100update.csv` |
| Code replication | `code/tables/code_replication.csv` |
| Signal analysis | `code/tables/recovery_signal.csv`, `code/analysis/p0_02_signal_analysis.py` |
| Exact-rank implementation | `code/rift_p0_routing.py`, `code/tests/test_rift_p0_routing.py` |
| Routing-score ablation | `code/tables/routing_score_ablation.csv` |
| Target arbitration | `code/tables/target_arbitration.csv` |
| ReGap | `code/tables/regap_pooled.csv` |
| Manuscript and detailed tables | `paper/RIFT_AAAI27_main.tex`, `paper/RIFT_AAAI27_supplement.tex` |

## 16. Reporting guidance

When citing an experimental result, state the model size, training horizon,
benchmark suite, sample count, and comparison baseline. In particular:

- Use the 50-update matched triad for the primary budget-matched RIFT-versus-
  OPSD comparison.
- Use the 100-update SixBench table only for its fixed-checkpoint comparison.
- Treat cross-scale rows as single-seed matched checks.
- Treat the long-budget table as exploratory.
- Treat ReGap as a counterfactual analysis protocol; it is not an inference-time
  module and does not justify replacing the target-routing evidence.

For manuscript-level methodological detail and formal definitions, consult the
main paper and technical supplement in [`paper/`](paper/).
