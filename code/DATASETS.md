# Public Assets and Frozen Evaluation Identifiers

## Models and training data

- Model family: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, and `Qwen/Qwen3-8B`.
- Mathematical training corpus: `siyanzhao/Openthoughts_math_30k_opsd`
  (29,434 rows in the frozen training snapshot).
- Code-domain training corpus: a fixed 30,000-example coding subset of
  `open-thoughts/OpenThoughts-114k`.

## Mathematical evaluation

- AIME 2024: 30-problem frozen snapshot.
- AIME 2025: 30-problem frozen snapshot.
- MATH-500: 500-problem frozen snapshot.
- AMC23: 40-problem frozen snapshot.
- HMMT February 2025: 30-problem frozen snapshot.
- HMMT November 2025: 30-problem frozen snapshot.

The exact frozen identifiers, file hashes, decoding configuration, and
evaluator hash are listed in the anonymous technical supplement. Dataset files
are not redistributed in this archive; obtain them from the public providers
cited by the paper.

## Code evaluation

- HumanEval+ and MBPP+ through EvalPlus v0.3.1.
- Four completions per problem; Code Macro is the unweighted mean of the two
  benchmark-level Avg@4 values.

