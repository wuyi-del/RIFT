#!/usr/bin/env python3
"""Run C3's preregistered P0-02 recovery-signal validity analysis.

The analysis deliberately keeps the P0-01 candidate set and recovery outcome
fixed.  Predictor directions and probabilities are fit out-of-fold by problem,
then all ranking, calibration, coverage, and bootstrap statistics use those
held-out probabilities.  No benchmark checkpoint is selected here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "P0-02-signal-validity-v1"
FOLDS = 5
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20260720
SHUFFLE_SEED = 20260720
COVERAGE = 0.25
EPSILON = 1e-8


def cst_now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S %Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_uint64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return pd.DataFrame(rows)


def safe_float(value: float | np.floating | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic(features: np.ndarray, labels: np.ndarray, l2: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a small L2 logistic model without an external ML dependency."""
    medians = np.nanmedian(features, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(features), features, medians)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales = np.where(scales > EPSILON, scales, 1.0)
    standard = (filled - means) / scales
    design = np.column_stack((np.ones(len(standard)), standard))
    prior = float(np.clip(labels.mean(), EPSILON, 1.0 - EPSILON))
    weights = np.zeros(design.shape[1])
    weights[0] = math.log(prior / (1.0 - prior))
    penalty = np.full(design.shape[1], l2)
    penalty[0] = 0.0

    for _ in range(100):
        probabilities = sigmoid(design @ weights)
        gradient = design.T @ (probabilities - labels) + penalty * weights
        curvature = probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * curvature[:, None]) + np.diag(penalty)
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            update = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        step = 1.0
        old_objective = (
            -np.sum(labels * np.log(probabilities + EPSILON) + (1.0 - labels) * np.log(1.0 - probabilities + EPSILON))
            + 0.5 * float(np.sum(penalty * weights * weights))
        )
        while step >= 1.0 / 128.0:
            candidate = weights - step * update
            candidate_probabilities = sigmoid(design @ candidate)
            candidate_objective = (
                -np.sum(
                    labels * np.log(candidate_probabilities + EPSILON)
                    + (1.0 - labels) * np.log(1.0 - candidate_probabilities + EPSILON)
                )
                + 0.5 * float(np.sum(penalty * candidate * candidate))
            )
            if candidate_objective <= old_objective:
                weights = candidate
                break
            step *= 0.5
        if np.max(np.abs(step * update)) < 1e-7:
            break
    return weights, medians, np.column_stack((means, scales))


def predict_logistic(features: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weights, medians, normalizer = model
    filled = np.where(np.isfinite(features), features, medians)
    standard = (filled - normalizer[:, 0]) / normalizer[:, 1]
    return sigmoid(np.column_stack((np.ones(len(standard)), standard)) @ weights)


def folds_for_problems(problem_ids: pd.Series, folds: int = FOLDS) -> np.ndarray:
    return np.asarray([stable_uint64(f"P0-02-fold-v1:{value}") % folds for value in problem_ids.astype(str)])


def oof_probabilities(features: np.ndarray, labels: np.ndarray, problem_ids: pd.Series) -> np.ndarray:
    assignments = folds_for_problems(problem_ids)
    output = np.empty(len(labels), dtype=float)
    for fold in range(FOLDS):
        train = assignments != fold
        test = ~train
        if not test.any():
            continue
        if labels[train].min() == labels[train].max():
            output[test] = float(labels[train].mean())
            continue
        model = fit_logistic(features[train], labels[train])
        output[test] = predict_logistic(features[test], model)
    return output


def stable_rank_order(scores: np.ndarray, ids: Iterable[str]) -> np.ndarray:
    return np.lexsort((np.asarray(list(ids), dtype=str), -np.asarray(scores, dtype=float)))


def average_precision(labels: np.ndarray, scores: np.ndarray, ids: Iterable[str]) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    ordered = labels[stable_rank_order(scores, ids)]
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    return float((ordered * (np.cumsum(ordered) / ranks)).sum() / positives)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels.astype(bool)].sum())
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) <= EPSILON or np.std(right) <= EPSILON:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return pearson(pd.Series(left).rank(method="average").to_numpy(), pd.Series(right).rank(method="average").to_numpy())


def coverage_metrics(labels: np.ndarray, rates: np.ndarray, scores: np.ndarray, ids: Iterable[str]) -> dict[str, float | int | None]:
    count = max(1, int(round(COVERAGE * len(labels))))
    chosen = stable_rank_order(scores, ids)[:count]
    positives = int(labels.sum())
    return {
        "coverage": COVERAGE,
        "selected_count": count,
        "precision": safe_float(labels[chosen].mean()),
        "recall": safe_float(labels[chosen].sum() / positives) if positives else None,
        "mean_functional_recovery_rate": safe_float(rates[chosen].mean()),
        "robust_recovery_rate": None,
    }


def calibration_bins(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> list[dict[str, float | int | None]]:
    order = np.argsort(scores, kind="mergesort")
    chunks = np.array_split(order, bins)
    output: list[dict[str, float | int | None]] = []
    for index, chunk in enumerate(chunks, start=1):
        if not len(chunk):
            continue
        output.append(
            {
                "bin": index,
                "count": int(len(chunk)),
                "mean_predicted_probability": safe_float(scores[chunk].mean()),
                "observed_binary_recovery": safe_float(labels[chunk].mean()),
            }
        )
    return output


def metric_row(
    signal: str,
    labels: np.ndarray,
    rates: np.ndarray,
    robust: np.ndarray,
    scores: np.ndarray,
    ids: Iterable[str],
) -> dict[str, Any]:
    coverage = coverage_metrics(labels, rates, scores, ids)
    chosen = stable_rank_order(scores, ids)[: int(coverage["selected_count"])]
    coverage["robust_recovery_rate"] = safe_float(robust[chosen].mean())
    return {
        "record_type": "primary_signal_metric",
        "signal": signal,
        "candidate_count": int(len(labels)),
        "binary_recovery_prevalence": safe_float(labels.mean()),
        "functional_recovery_rate_mean": safe_float(rates.mean()),
        "average_precision": average_precision(labels, scores, ids),
        "auroc": auroc(labels, scores),
        "brier": safe_float(np.mean((scores - labels) ** 2)),
        "spearman_functional_recovery_rate": spearman(scores, rates),
        "coverage": coverage,
    }


def bootstrap_metric_distribution(
    labels: np.ndarray,
    rates: np.ndarray,
    robust: np.ndarray,
    predictions: dict[str, np.ndarray],
    problem_ids: np.ndarray,
    ids: np.ndarray,
) -> dict[str, dict[str, list[float]]]:
    groups = pd.Series(problem_ids).drop_duplicates().tolist()
    indices = {group: np.flatnonzero(problem_ids == group) for group in groups}
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    output = {
        signal: {"average_precision": [], "auroc": [], "brier": [], "coverage_precision": [], "coverage_recall": []}
        for signal in predictions
    }
    for _ in range(BOOTSTRAPS):
        drawn = generator.integers(0, len(groups), size=len(groups))
        selected = np.concatenate([indices[groups[index]] for index in drawn])
        bootstrap_ids = np.asarray([f"{ids[index]}:{repeat}" for repeat, index in enumerate(selected)])
        for signal, scores in predictions.items():
            metrics = metric_row(signal, labels[selected], rates[selected], robust[selected], scores[selected], bootstrap_ids)
            output[signal]["average_precision"].append(float(metrics["average_precision"]))
            output[signal]["auroc"].append(float(metrics["auroc"]))
            output[signal]["brier"].append(float(metrics["brier"]))
            output[signal]["coverage_precision"].append(float(metrics["coverage"]["precision"]))
            recall = metrics["coverage"]["recall"]
            output[signal]["coverage_recall"].append(float(recall) if recall is not None else float("nan"))
    return output


def ci(values: list[float]) -> list[float | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if not len(finite):
        return [None, None]
    return [safe_float(np.quantile(finite, 0.025)), safe_float(np.quantile(finite, 0.975))]


def deterministic_within_problem_shuffle(frame: pd.DataFrame) -> np.ndarray:
    shuffled = np.empty(len(frame), dtype=float)
    for problem, indices in frame.groupby("problem_id", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        generator = np.random.default_rng(stable_uint64(f"P0-02-shuffle-v1:{SHUFFLE_SEED}:{problem}"))
        shuffled[positions] = frame.loc[positions, "future_recovery"].to_numpy(dtype=float)[generator.permutation(len(positions))]
    return shuffled


def quantile_labels(values: pd.Series, name: str) -> pd.Series:
    if values.nunique(dropna=True) < 4:
        return pd.Series([f"{name}_unavailable"] * len(values), index=values.index)
    return pd.qcut(values.rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"])


def metric_for_subset(
    signal: str,
    frame: pd.DataFrame,
    scores: np.ndarray,
    subset: np.ndarray,
    stratum_type: str,
    stratum: str,
) -> dict[str, Any]:
    subset_frame = frame.iloc[np.flatnonzero(subset)]
    labels = subset_frame["functional_recovery_binary"].to_numpy(dtype=int)
    rates = subset_frame["functional_recovery_rate"].to_numpy(dtype=float)
    robust = subset_frame["robust_recovery"].to_numpy(dtype=int)
    identifiers = subset_frame["candidate_id"].astype(str).to_numpy()
    if labels.min() == labels.max():
        ap = None
        roc = None
    else:
        ap = average_precision(labels, scores[subset], identifiers)
        roc = auroc(labels, scores[subset])
    return {
        "record_type": "stratum_metric",
        "signal": signal,
        "stratum_type": stratum_type,
        "stratum": stratum,
        "candidate_count": int(len(subset_frame)),
        "binary_recovery_prevalence": safe_float(labels.mean()),
        "mean_functional_recovery_rate": safe_float(rates.mean()),
        "average_precision": ap,
        "auroc": roc,
        "mean_score": safe_float(scores[subset].mean()),
        "coverage": coverage_metrics(labels, rates, scores[subset], identifiers),
    }


def validate_inputs(frame: pd.DataFrame, continuations: pd.DataFrame) -> None:
    required = {
        "candidate_id", "problem_id", "position", "remaining_window_tokens", "entropy", "current_jsd",
        "suppression_margin", "ad_risk", "past_recovery", "future_recovery",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidates missing required columns: {missing}")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("candidate_id must be unique")
    if not {"candidate_id", "continuation_seed", "correct"}.issubset(continuations.columns):
        raise ValueError("continuations must include candidate_id, continuation_seed, correct")
    grouped = continuations.groupby("candidate_id")["continuation_seed"].nunique()
    if not grouped.eq(4).all():
        raise ValueError("P0-02 requires exactly four distinct continuation seeds per candidate")
    candidate_ids = set(frame["candidate_id"].astype(str))
    continuation_ids = set(continuations["candidate_id"].astype(str))
    if candidate_ids != continuation_ids:
        raise ValueError("candidate and continuation IDs do not match exactly")
    for table_name, table in (("candidates", frame), ("continuations", continuations)):
        if "reference_in_prompt" in table.columns and table["reference_in_prompt"].fillna(False).astype(bool).any():
            raise ValueError(f"{table_name} contains reference_in_prompt=True")


def file_hash_records(paths: Iterable[Path]) -> list[str]:
    return [f"{sha256(path)}  {path.name}" for path in paths]


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates_path = args.candidates.resolve()
    continuations_path = args.continuations.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_manifest.resolve()
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"task_id": "P0-02"}
    manifest.update(
        {
            "status": "running",
            "analysis_version": ANALYSIS_VERSION,
            "start_time_cst": cst_now(),
            "data_path": str(candidates_path.parent),
            "data_sha256": {"candidates.parquet": sha256(candidates_path), "continuations.jsonl": sha256(continuations_path)},
            "command_file": "P0-02_command.sh",
        }
    )
    write_json(manifest_path, manifest)

    candidates = pd.read_parquet(candidates_path).reset_index(drop=True)
    continuations = read_jsonl(continuations_path)
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    continuations["candidate_id"] = continuations["candidate_id"].astype(str)
    validate_inputs(candidates, continuations)
    outcomes = continuations.groupby("candidate_id", sort=False)["correct"].agg(["mean", "max", "sum"]).reset_index()
    outcomes = outcomes.rename(columns={"mean": "functional_recovery_rate", "max": "functional_recovery_binary", "sum": "correct_count"})
    outcomes["functional_recovery_binary"] = outcomes["functional_recovery_binary"].astype(int)
    outcomes["robust_recovery"] = outcomes["correct_count"].ge(2).astype(int)
    frame = candidates.merge(outcomes, on="candidate_id", how="inner", validate="one_to_one").reset_index(drop=True)
    if len(frame) != len(candidates):
        raise ValueError("outcome merge changed candidate count")

    labels = frame["functional_recovery_binary"].to_numpy(dtype=int)
    rates = frame["functional_recovery_rate"].to_numpy(dtype=float)
    robust = frame["robust_recovery"].to_numpy(dtype=int)
    identifiers = frame["candidate_id"].astype(str).to_numpy()
    problems = frame["problem_id"].astype(str)
    shuffled_future = deterministic_within_problem_shuffle(frame)
    numeric_columns = [
        "entropy", "current_jsd", "suppression_margin", "ad_risk", "past_recovery", "future_recovery",
        "position", "remaining_window_tokens",
    ]
    if not set(numeric_columns).issubset(frame.columns):
        raise ValueError("missing a numerical P0-02 feature")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    current_features = ["entropy", "current_jsd", "suppression_margin", "ad_risk", "position", "remaining_window_tokens"]
    predictor_definitions: dict[str, tuple[np.ndarray, list[str], str]] = {
        "position_only": (frame[["position", "remaining_window_tokens"]].to_numpy(dtype=float), ["position", "remaining_window_tokens"], "current"),
        "entropy": (frame[["entropy"]].to_numpy(dtype=float), ["entropy"], "current"),
        "current_jsd": (frame[["current_jsd"]].to_numpy(dtype=float), ["current_jsd"], "current"),
        "suppression_margin": (frame[["suppression_margin"]].to_numpy(dtype=float), ["suppression_margin"], "current"),
        "ad_risk": (frame[["ad_risk"]].to_numpy(dtype=float), ["ad_risk"], "current"),
        "past_recovery": (frame[["past_recovery"]].to_numpy(dtype=float), ["past_recovery"], "past"),
        "future_shuffled": (shuffled_future.reshape(-1, 1), ["within_problem_shuffled_future_recovery"], "placebo"),
        "future_recovery": (frame[["future_recovery"]].to_numpy(dtype=float), ["future_recovery"], "future"),
        "current_covariates": (frame[current_features].to_numpy(dtype=float), current_features, "current"),
        "future_recovery_plus_current_covariates": (
            frame[current_features + ["future_recovery"]].to_numpy(dtype=float), current_features + ["future_recovery"], "future_plus_current"
        ),
    }
    optional_diagnostics: dict[str, tuple[np.ndarray, list[str], str]] = {}
    if {"privileged_advantage", "base_advantage"}.issubset(frame.columns):
        privilege_gap = (
            pd.to_numeric(frame["privileged_advantage"], errors="coerce")
            - pd.to_numeric(frame["base_advantage"], errors="coerce")
        )
        optional_diagnostics["current_privilege_gap"] = (privilege_gap.to_numpy(dtype=float).reshape(-1, 1), ["privileged_advantage_minus_base_advantage"], "current_diagnostic")
    predictions: dict[str, np.ndarray] = {
        "random": np.asarray([stable_uint64(f"P0-02-random-v1:{value}") / 2**64 for value in identifiers], dtype=float)
    }
    metadata: dict[str, Any] = {
        "random": {"temporal_direction": "none", "features": ["stable_candidate_id_hash"], "fit": "none; coverage-matched null"}
    }
    for signal, (features, names, temporal_direction) in {**predictor_definitions, **optional_diagnostics}.items():
        predictions[signal] = oof_probabilities(features, labels, problems)
        metadata[signal] = {
            "temporal_direction": temporal_direction,
            "features": names,
            "fit": f"{FOLDS}-fold problem-held-out L2 logistic calibration",
        }

    main_metrics = [metric_row(signal, labels, rates, robust, scores, identifiers) for signal, scores in predictions.items()]
    current_only = [
        metric for metric in main_metrics
        if metadata[metric["signal"]]["temporal_direction"] in {"current", "current_diagnostic"}
    ]
    strongest_current = max(current_only, key=lambda metric: float(metric["average_precision"]))
    bootstrap = bootstrap_metric_distribution(labels, rates, robust, predictions, problems.to_numpy(), identifiers)
    bootstrap_rows: list[dict[str, Any]] = []
    strongest_name = str(strongest_current["signal"])
    future_bootstrap = np.asarray(bootstrap["future_recovery"]["average_precision"])
    strongest_bootstrap = np.asarray(bootstrap[strongest_name]["average_precision"])
    for signal, values in bootstrap.items():
        row = {"record_type": "problem_cluster_bootstrap", "signal": signal, "bootstrap_count": BOOTSTRAPS}
        for metric_name, samples in values.items():
            row[f"{metric_name}_95_ci"] = ci(samples)
        if signal != strongest_name:
            delta = np.asarray(values["average_precision"]) - strongest_bootstrap
            row["delta_auprc_vs_strongest_current_95_ci"] = ci(delta.tolist())
            row["delta_auprc_vs_strongest_current_mean"] = safe_float(np.mean(delta))
        bootstrap_rows.append(row)
    future_delta = future_bootstrap - strongest_bootstrap

    strata_rows: list[dict[str, Any]] = []
    position_quartile = quantile_labels(frame["position"], "position")
    remaining_quartile = quantile_labels(frame["remaining_window_tokens"], "remaining_window")
    for name, labels_for_strata in (("position_quartile", position_quartile), ("remaining_window_quartile", remaining_quartile)):
        for stratum in sorted(pd.Series(labels_for_strata.astype(str)).unique()):
            subset = labels_for_strata.astype(str).to_numpy() == stratum
            strata_rows.append(metric_for_subset("future_recovery", frame, predictions["future_recovery"], subset, name, stratum))
            strata_rows.append(metric_for_subset(strongest_name, frame, predictions[strongest_name], subset, name, stratum))
    strata_rows.append(
        {
            "record_type": "stratum_availability",
            "stratum_type": "difficulty",
            "status": "not_available_in_P0-01_handoff",
            "reason": "The frozen candidate contract has no independent problem-difficulty field; no proxy was substituted.",
        }
    )

    horizon_rows: list[dict[str, Any]] = []
    for horizon in (8, 16, 32, 64):
        future_column = f"future_recovery_h{horizon}"
        past_column = f"past_recovery_h{horizon}"
        future_full_column = f"future_h{horizon}_full"
        past_full_column = f"past_h{horizon}_full"
        if not {future_column, past_column, future_full_column, past_full_column}.issubset(frame.columns):
            horizon_rows.append({"record_type": "horizon_audit", "horizon": horizon, "status": "missing_columns"})
            continue
        future_full = frame[future_full_column].fillna(False).astype(bool).to_numpy()
        past_full = frame[past_full_column].fillna(False).astype(bool).to_numpy()
        full = future_full & past_full
        horizon_row: dict[str, Any] = {
            "record_type": "horizon_audit",
            "horizon": horizon,
            "candidate_count": int(len(frame)),
            "future_full_count": int(future_full.sum()),
            "past_full_count": int(past_full.sum()),
            "both_full_count": int(full.sum()),
            "censored_count": int((~full).sum()),
            "future_censoring_fraction": safe_float((~future_full).mean()),
            "past_censoring_fraction": safe_float((~past_full).mean()),
        }
        if full.sum() >= 10 and labels[full].min() != labels[full].max():
            future_h = oof_probabilities(pd.to_numeric(frame.loc[full, future_column], errors="coerce").to_numpy(dtype=float).reshape(-1, 1), labels[full], problems[full])
            past_h = oof_probabilities(pd.to_numeric(frame.loc[full, past_column], errors="coerce").to_numpy(dtype=float).reshape(-1, 1), labels[full], problems[full])
            future_metrics = metric_row(f"future_recovery_h{horizon}", labels[full], rates[full], robust[full], future_h, identifiers[full])
            past_metrics = metric_row(f"past_recovery_h{horizon}", labels[full], rates[full], robust[full], past_h, identifiers[full])
            horizon_row["future_mean_score"] = safe_float(pd.to_numeric(frame.loc[full, future_column], errors="coerce").mean())
            horizon_row["past_mean_score"] = safe_float(pd.to_numeric(frame.loc[full, past_column], errors="coerce").mean())
            horizon_row["future_auprc"] = future_metrics["average_precision"]
            horizon_row["past_auprc"] = past_metrics["average_precision"]
            horizon_row["future_auroc"] = future_metrics["auroc"]
            horizon_row["past_auroc"] = past_metrics["auroc"]
            horizon_row["future_route_rate"] = COVERAGE
            horizon_row["past_route_rate"] = COVERAGE
            horizon_row["future_near_eos_route_rate"] = None
            horizon_row["future_position_imbalance"] = None
            horizon_row["status"] = "analyzed_full_window_only"
        else:
            horizon_row["status"] = "insufficient_full_window_or_class_support"
        horizon_rows.append(horizon_row)

    calibration = {signal: calibration_bins(labels, scores) for signal, scores in predictions.items()}
    metric_frame = pd.json_normalize(main_metrics, sep=".")
    bootstrap_frame = pd.json_normalize(bootstrap_rows, sep=".")
    strata_frame = pd.json_normalize(strata_rows, sep=".")
    horizon_frame = pd.json_normalize(horizon_rows, sep=".")
    metric_frame.to_csv(output_dir / "table_signal_validity.csv", index=False)
    bootstrap_frame.to_csv(output_dir / "problem_cluster_bootstrap_ci.csv", index=False)
    strata_frame.to_csv(output_dir / "strata_signal_validity.csv", index=False)
    horizon_frame.to_csv(output_dir / "horizon_censoring_audit.csv", index=False)
    write_json(output_dir / "calibration_curves.json", calibration)
    with (output_dir / "metrics_raw.jsonl").open("w") as handle:
        for record in [*main_metrics, *bootstrap_rows, *strata_rows, *horizon_rows]:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    metric_by_name = {str(row["signal"]): row for row in main_metrics}
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete",
        "created_at_cst": cst_now(),
        "config": {
            "candidate_coverage": COVERAGE,
            "problem_held_out_folds": FOLDS,
            "problem_cluster_bootstraps": BOOTSTRAPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "within_problem_shuffle_seed": SHUFFLE_SEED,
            "primary_outcome": "functional_recovery_binary = any correct across exactly four forced continuations",
            "secondary_rate_outcome": "functional_recovery_rate = mean correct across exactly four forced continuations",
            "probability_estimator": "out-of-fold L2 logistic calibration by problem; random is an uncalibrated deterministic null",
        },
        "input": {
            "candidates_path": str(candidates_path),
            "continuations_path": str(continuations_path),
            "candidates_sha256": sha256(candidates_path),
            "continuations_sha256": sha256(continuations_path),
            "candidate_count": int(len(frame)),
            "problem_count": int(problems.nunique()),
            "continuation_count": int(len(continuations)),
            "continuations_per_candidate": int(continuations.groupby("candidate_id")["continuation_seed"].nunique().iloc[0]),
            "candidate_reference_leak_count": int(candidates.get("reference_in_prompt", pd.Series(False, index=candidates.index)).fillna(False).astype(bool).sum()),
            "continuation_reference_leak_count": int(continuations.get("reference_in_prompt", pd.Series(False, index=continuations.index)).fillna(False).astype(bool).sum()),
        },
        "outcome": {
            "functional_recovery_rate_mean": safe_float(rates.mean()),
            "functional_recovery_binary_fraction": safe_float(labels.mean()),
            "robust_recovery_fraction": safe_float(robust.mean()),
        },
        "signal_metadata": metadata,
        "strongest_current_only_predictor": strongest_name,
        "future_recovery": metric_by_name["future_recovery"],
        "strongest_current_only": metric_by_name[strongest_name],
        "temporal_placebo": {
            "past_recovery": metric_by_name["past_recovery"],
            "future_shuffled": metric_by_name["future_shuffled"],
        },
        "future_delta_auprc_vs_strongest_current": {
            "point_estimate": safe_float(metric_by_name["future_recovery"]["average_precision"] - metric_by_name[strongest_name]["average_precision"]),
            "problem_cluster_bootstrap_95_ci": ci(future_delta.tolist()),
        },
        "difficulty_stratum_caveat": "No independent difficulty field was present in the frozen P0-01 handoff; the requested difficulty analysis is explicitly marked unavailable rather than proxied.",
        "g1_evidence_only": "C3 reports prespecified signal, placebo, calibration, and bootstrap evidence. Gate G1 requires review of these fields and is not decided from downstream benchmark results.",
        "raw_metrics_record_count": len(main_metrics) + len(bootstrap_rows) + len(strata_rows) + len(horizon_rows),
    }
    write_json(output_dir / "summary.json", summary)
    markdown = [
        "# P0-02 Signal Validity Summary",
        "",
        f"- Candidates/problems/continuations: {len(frame)}/{problems.nunique()}/{len(continuations)}",
        f"- Primary binary outcome prevalence: {labels.mean():.6f}",
        f"- Strongest current-only predictor: `{strongest_name}`",
        f"- Future recovery AUPRC: {metric_by_name['future_recovery']['average_precision']:.6f}",
        f"- ΔAUPRC vs strongest current: {summary['future_delta_auprc_vs_strongest_current']['point_estimate']:.6f}",
        f"- Problem-cluster 95% CI: {summary['future_delta_auprc_vs_strongest_current']['problem_cluster_bootstrap_95_ci']}",
        "- G1 remains an evidence review; no downstream benchmark result is used here.",
        "- Difficulty is unavailable in the frozen handoff and was not replaced by a proxy.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(markdown))

    artifact_paths = [
        output_dir / name
        for name in (
            "table_signal_validity.csv",
            "problem_cluster_bootstrap_ci.csv",
            "strata_signal_validity.csv",
            "horizon_censoring_audit.csv",
            "calibration_curves.json",
            "metrics_raw.jsonl",
            "summary.json",
            "summary.md",
        )
    ]
    (output_dir / "artifacts.sha256").write_text("\n".join(file_hash_records(artifact_paths)) + "\n")
    expected_hashes = {
        name: checksum
        for checksum, name in (
            line.split("  ", 1) for line in (output_dir / "artifacts.sha256").read_text().splitlines()
        )
    }
    hash_errors = [
        path.name for path in artifact_paths if sha256(path) != expected_hashes.get(path.name)
    ]
    if hash_errors:
        raise RuntimeError(f"artifact hash validation failed: {hash_errors}")
    manifest.update(
        {
            "status": "complete",
            "end_time_cst": cst_now(),
            "candidate_count": int(len(frame)),
            "primary_output": str(output_dir / "summary.json"),
            "failure_reason": "",
            "artifact_hash_manifest": str(output_dir / "artifacts.sha256"),
        }
    )
    write_json(manifest_path, manifest)
    (output_dir / "DONE").write_text(
        "P0-02 signal-validity analysis completed. Input contract, raw records, summary, and SHA256 artifact manifest verified.\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--continuations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({"status": summary["status"], "summary": str(args.output_dir / "summary.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
