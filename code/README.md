# Anonymous RIFT Code and Data Supplement

This archive accompanies submission 20523. It contains an anonymous snapshot
of the RIFT implementation, evaluation utilities, analysis code, configuration
files, and aggregate values used by the paper tables.

## Scope

The archive is intended for reviewer inspection and method reproduction. It
does not contain model weights, licensed benchmark copies, private machine
paths, server credentials, or completion-level raw generations. Public model
and dataset identifiers are listed in `DATASETS.md`; aggregate paper values are
provided under `tables/`.

## Layout

- `opsd_train.py`, `opsd_trainer.py`, `data_collator.py`: on-policy
  self-distillation training implementation.
- `rift_p0_routing.py`: candidate construction, future-recovery scoring,
  exact-rank selection, and routing utilities.
- `sft_train.py`, `grpo_train.py`: training-paradigm baselines.
- `scripts/`: budget-matched RIFT, Matched OPSD, SFT, and GRPO launch recipes.
- `eval/`: math and code evaluation programs.
- `analysis/`: recovery-signal analysis and paper-figure generation.
- `tests/`: routing unit tests.
- `configs/`: environment and distributed-training configurations.
- `tables/`: aggregate values reported in the paper and supplement.

## Environment

Create the environment from `configs/environment.yml`, then install the pinned
or recorded packages in `configs/DEPENDENCIES.txt`. The reported evaluator uses
`math-verify==0.8.0`; the code replication uses EvalPlus v0.3.1.

## Training

The launch scripts expect the public Qwen3 model under `/models/` and a local
copy of the selected training corpus under `data/`. Paths, step count, seed,
and output directory can be overridden through environment variables.

Example 100-update RIFT run:

```bash
MAX_STEPS=100 \
SEED=42 \
EXACT_RANK=1 \
RECOVERY_QUANTILE=0.25 \
DATA=data/openthoughts_math_30k \
bash scripts/run_rift_v2_4b.sh
```

Budget-matched control:

```bash
MAX_STEPS=100 SEED=42 DATA=data/openthoughts_math_30k \
bash scripts/run_matched_opsd_4b.sh
```

The paper uses seeds 42--46 for the checkpoint-100 Qwen3-4B confirmatory study.
The controlled 50-update scale checks use the seeds stated in the paper.

## Evaluation and analysis

`eval/evaluate_math.py` implements the final-answer extraction and
`math-verify` scoring path used by the mathematical evaluation. The code
evaluation entry point is `eval/evaluate_code.py`. Recovery-signal OOF analysis
is implemented in `analysis/p0_02_signal_analysis.py`.

## Reproducibility boundary

The archive documents the method, launch recipes, evaluator, frozen public
asset identifiers, and aggregate result tables. Large public datasets and
model weights must be obtained from their original providers. The technical
supplement records the benchmark hashes and statistical protocols associated
with the reported results.

