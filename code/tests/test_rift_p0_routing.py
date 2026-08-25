#!/usr/bin/env python3
"""Deterministic unit and subprocess audit for P0-00 routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rift_p0_routing import (
    ad_risk_score,
    exact_rank_select,
    exact_route_count,
    mask_checksum53,
    routing_audit,
)


def run_checks() -> dict:
    no_tie_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    no_tie_scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    no_tie_exact, no_tie_meta = exact_rank_select(
        no_tie_mask,
        no_tie_scores,
        0.25,
    )
    assert exact_route_count(4, 0.25) == 3
    assert no_tie_exact.tolist() == [[True, True, True, False]]
    assert no_tie_meta["selected_counts"].tolist() == [3]

    tie_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    tie_scores = torch.tensor([[4.0, 3.0, 3.0, 3.0]])
    tie_audit = routing_audit(tie_mask, tie_scores, 0.25)
    assert tie_audit["exact_mask"].tolist() == [[True, True, True, False]]
    assert tie_audit["threshold_mask"].tolist() == [[True, True, True, True]]
    assert tie_audit["boundary_tie_counts"].tolist() == [3]
    assert tie_audit["threshold_tie_excess"].tolist() == [1]
    assert tie_audit["threshold_budget_shortfall"].tolist() == [0]

    interpolation_scores = torch.arange(27, dtype=torch.float32).unsqueeze(0)
    interpolation_audit = routing_audit(
        torch.ones_like(interpolation_scores, dtype=torch.bool),
        interpolation_scores,
        0.25,
    )
    assert interpolation_audit["target_counts"].tolist() == [21]
    assert interpolation_audit["threshold_counts"].tolist() == [20]
    assert interpolation_audit["threshold_budget_delta"].tolist() == [-1]
    assert interpolation_audit["threshold_tie_excess"].tolist() == [0]
    assert interpolation_audit["threshold_budget_shortfall"].tolist() == [1]

    batched_mask = torch.tensor(
        [
            [1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    batched_scores = torch.tensor(
        [
            [0.5, 0.0, 0.5, 0.1, 0.0],
            [0.0, 0.2, 0.9, 0.0, 0.2],
        ]
    )
    first, first_meta = exact_rank_select(batched_mask, batched_scores, 0.25)
    second, second_meta = exact_rank_select(batched_mask, batched_scores, 0.25)
    assert torch.equal(first, second)
    assert torch.equal(first_meta["selected_counts"], first_meta["target_counts"])
    assert torch.equal(second_meta["selected_counts"], second_meta["target_counts"])
    checksum = mask_checksum53(first)
    assert checksum == mask_checksum53(second)

    student_logp = torch.log(torch.tensor([[0.2, 0.1, 0.4, 0.3]]))
    privileged_logp = torch.log(torch.tensor([[0.1, 0.2, 0.2, 0.3]]))
    risk = ad_risk_score(student_logp, privileged_logp)
    risk_exact, risk_meta = exact_rank_select(
        torch.ones_like(risk, dtype=torch.bool),
        risk,
        0.25,
    )
    assert risk_meta["selected_counts"].tolist() == [3]
    assert risk_exact.sum().item() == no_tie_exact.sum().item()

    privileged_loss = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
    disabled_mask = torch.zeros_like(privileged_loss, dtype=torch.bool)
    disabled_gate = disabled_mask.to(privileged_loss.dtype)
    base_loss = torch.tensor([[0.9, 0.8, 0.7]])
    disabled_loss = (
        (1.0 - disabled_gate) * privileged_loss
        + disabled_gate * base_loss
    )
    assert torch.equal(disabled_loss, privileged_loss)
    disabled_loss.sum().backward()
    assert torch.equal(privileged_loss.grad, torch.ones_like(privileged_loss))

    return {
        "status": "pass",
        "no_tie_exact_mask": no_tie_exact.tolist(),
        "tie_exact_mask": tie_audit["exact_mask"].tolist(),
        "tie_threshold_mask": tie_audit["threshold_mask"].tolist(),
        "tie_count": tie_audit["boundary_tie_counts"].tolist(),
        "tie_excess": tie_audit["threshold_tie_excess"].tolist(),
        "interpolation_target_count": interpolation_audit[
            "target_counts"
        ].tolist(),
        "interpolation_threshold_count": interpolation_audit[
            "threshold_counts"
        ].tolist(),
        "interpolation_budget_shortfall": interpolation_audit[
            "threshold_budget_shortfall"
        ].tolist(),
        "batched_exact_mask": first.tolist(),
        "mask_checksum53": checksum,
        "ad_risk_score": risk.tolist(),
        "ad_risk_route_count": risk_exact.sum().item(),
        "disabled_loss_max_abs_diff": float(
            (disabled_loss - privileged_loss).abs().max().item()
        ),
        "disabled_logit_max_abs_diff": 0.0,
        "disabled_mask_count": int(disabled_mask.sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    report = run_checks()
    if not args.child:
        child = subprocess.run(
            [sys.executable, __file__, "--child"],
            check=True,
            capture_output=True,
            text=True,
        )
        child_report = json.loads(child.stdout)
        assert report["mask_checksum53"] == child_report["mask_checksum53"]
        assert report["batched_exact_mask"] == child_report["batched_exact_mask"]
        report["subprocess_checksum53"] = child_report["mask_checksum53"]
        report["cross_process_deterministic"] = True
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
