# RIFT: Future-Recovery Routing for Privileged On-Policy Self-Distillation

This repository contains the paper source, compiled manuscript, figures, and
the reproducibility code package for **RIFT** (Recovery-Informed Forked
Training) and the accompanying **ReGap** counterfactual analysis.

## Repository layout

- `paper/`: AAAI manuscript source, bibliography, style file, figures, and
  compiled main paper and supplement PDFs.
- `code/`: training implementation, exact-rank routing, evaluation utilities,
  experiment configurations, analysis scripts, unit tests, and aggregate
  tables used in the paper.

## Reproducing the code package

See [`code/README.md`](code/README.md) for environment, public model/dataset
identifiers, training commands, evaluation, and the reproducibility boundary.
The repository deliberately excludes model weights, licensed benchmark copies,
completion-level generations, server information, and credentials.

## Citation

Citation metadata will be added after the anonymous-review period. Until then,
please refer to the manuscript in [`paper/RIFT_AAAI27_main_8pages.pdf`](paper/RIFT_AAAI27_main_8pages.pdf).

## License

The implementation in `code/` is released under the MIT License; see
[`code/LICENSE`](code/LICENSE).
