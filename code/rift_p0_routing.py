"""Deterministic P0 routing primitives for the AAAI-27 RIFT experiments."""

from __future__ import annotations

import math

import torch


EXACT_RANK_HASH_SEED = 20260717
CHECKSUM_MODULUS = 9007199254740881  # below 2**53, exactly representable as float


def _position_hash(
    positions: torch.Tensor,
    batch_index: int,
    seed: int = EXACT_RANK_HASH_SEED,
) -> torch.Tensor:
    """Return a platform-stable integer hash used only after score and position."""
    values = (
        (positions.to(torch.int64) + 1) * 1103515245
        + (batch_index + 1) * 12345
        + seed
    )
    return torch.remainder(values, 2147483647)


def exact_route_count(candidate_count: int, quantile: float) -> int:
    """Map a lower-tail quantile to the exact upper-tail route budget."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if candidate_count <= 0:
        return 0
    return min(
        candidate_count,
        max(1, int(math.ceil((1.0 - quantile) * candidate_count))),
    )


def mask_checksum53(mask: torch.Tensor, seed: int = EXACT_RANK_HASH_SEED) -> int:
    """Compute a deterministic integer checksum that survives float logging."""
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask_checksum53 expects a rank-2 boolean mask")
    total = 0
    for batch_index in range(mask.shape[0]):
        positions = torch.nonzero(mask[batch_index], as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            continue
        hashed = _position_hash(positions, batch_index, seed)
        total = (total + int(hashed.sum().item())) % CHECKSUM_MODULUS
    return total


def exact_rank_select(
    candidate_mask: torch.Tensor,
    scores: torch.Tensor,
    quantile: float,
    hash_seed: int = EXACT_RANK_HASH_SEED,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Select an exact per-trajectory upper-tail budget.

    The lexicographic order is score descending, token position ascending, then
    a fixed hash. Token positions are unique within a trajectory, so the hash is
    a final deterministic fallback and never changes a valid token sequence.
    """
    if candidate_mask.shape != scores.shape:
        raise ValueError("candidate mask and scores must have identical shapes")
    if candidate_mask.ndim != 2 or candidate_mask.dtype != torch.bool:
        raise ValueError("candidate mask must be a rank-2 boolean tensor")
    if not torch.isfinite(scores[candidate_mask]).all():
        raise ValueError("candidate scores must be finite")

    batch_size = candidate_mask.shape[0]
    selected = torch.zeros_like(candidate_mask)
    candidate_counts = torch.zeros(batch_size, dtype=torch.long, device=scores.device)
    target_counts = torch.zeros_like(candidate_counts)
    thresholds = torch.full(
        (batch_size,),
        float("inf"),
        dtype=scores.dtype,
        device=scores.device,
    )
    boundary_tie_counts = torch.zeros_like(candidate_counts)

    for batch_index in range(batch_size):
        positions = torch.nonzero(
            candidate_mask[batch_index],
            as_tuple=False,
        ).squeeze(-1)
        candidate_count = int(positions.numel())
        target_count = exact_route_count(candidate_count, quantile)
        candidate_counts[batch_index] = candidate_count
        target_counts[batch_index] = target_count
        if target_count == 0:
            continue

        # Stable least-significant to most-significant sorting implements the
        # frozen lexicographic tie break without moving tensors to the CPU.
        hashes = _position_hash(positions, batch_index, hash_seed)
        order = torch.argsort(hashes, descending=False, stable=True)
        ordered_positions = positions[order]
        order = torch.argsort(ordered_positions, descending=False, stable=True)
        ordered_positions = ordered_positions[order]
        order = torch.argsort(
            scores[batch_index, ordered_positions],
            descending=True,
            stable=True,
        )
        ordered_positions = ordered_positions[order]
        chosen = ordered_positions[:target_count]
        selected[batch_index, chosen] = True

        boundary = scores[batch_index, ordered_positions[target_count - 1]]
        thresholds[batch_index] = boundary
        boundary_tie_counts[batch_index] = (
            scores[batch_index, positions] == boundary
        ).sum()

    selected_counts = selected.sum(dim=-1)
    if not torch.equal(selected_counts, target_counts):
        raise RuntimeError("exact-rank selector violated its per-trajectory budget")
    metadata = {
        "candidate_counts": candidate_counts,
        "target_counts": target_counts,
        "selected_counts": selected_counts,
        "thresholds": thresholds,
        "boundary_tie_counts": boundary_tie_counts,
    }
    return selected, metadata


def threshold_quantile_select(
    candidate_mask: torch.Tensor,
    scores: torch.Tensor,
    quantile: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reproduce the historical torch.quantile plus >= threshold selector."""
    if candidate_mask.shape != scores.shape:
        raise ValueError("candidate mask and scores must have identical shapes")
    selected = torch.zeros_like(candidate_mask)
    thresholds = torch.full(
        (candidate_mask.shape[0],),
        float("inf"),
        dtype=scores.dtype,
        device=scores.device,
    )
    candidate_counts = candidate_mask.sum(dim=-1)
    for batch_index in range(candidate_mask.shape[0]):
        values = scores[batch_index][candidate_mask[batch_index]].float()
        if values.numel() == 0:
            continue
        threshold = torch.quantile(values, quantile).to(scores.dtype)
        thresholds[batch_index] = threshold
        selected[batch_index] = (
            candidate_mask[batch_index]
            & (scores[batch_index] >= threshold)
        )
    return selected, {
        "candidate_counts": candidate_counts,
        "selected_counts": selected.sum(dim=-1),
        "thresholds": thresholds,
    }


def routing_audit(
    candidate_mask: torch.Tensor,
    scores: torch.Tensor,
    quantile: float,
) -> dict[str, torch.Tensor | int]:
    """Compare historical threshold routing with exact-rank routing."""
    exact_mask, exact = exact_rank_select(candidate_mask, scores, quantile)
    threshold_mask, threshold = threshold_quantile_select(
        candidate_mask,
        scores,
        quantile,
    )
    threshold_budget_delta = (
        threshold["selected_counts"] - exact["target_counts"]
    )
    threshold_tie_excess = threshold_budget_delta.clamp_min(0)
    threshold_budget_shortfall = (-threshold_budget_delta).clamp_min(0)
    return {
        "exact_mask": exact_mask,
        "threshold_mask": threshold_mask,
        "candidate_counts": exact["candidate_counts"],
        "target_counts": exact["target_counts"],
        "exact_counts": exact["selected_counts"],
        "threshold_counts": threshold["selected_counts"],
        "exact_thresholds": exact["thresholds"],
        "quantile_thresholds": threshold["thresholds"],
        "boundary_tie_counts": exact["boundary_tie_counts"],
        "threshold_budget_delta": threshold_budget_delta,
        "threshold_tie_excess": threshold_tie_excess,
        "threshold_budget_shortfall": threshold_budget_shortfall,
        "mask_difference_counts": (exact_mask ^ threshold_mask).sum(dim=-1),
        "exact_checksum53": mask_checksum53(exact_mask),
        "threshold_checksum53": mask_checksum53(threshold_mask),
    }


def ad_risk_score(
    student_token_logp: torch.Tensor,
    privileged_token_logp: torch.Tensor,
) -> torch.Tensor:
    """AD-OPSD instantaneous sampled-token suppression score."""
    if student_token_logp.shape != privileged_token_logp.shape:
        raise ValueError("student and privileged token log-probs must match")
    suppression = (student_token_logp - privileged_token_logp).clamp_min(0.0)
    return student_token_logp.float().exp() * suppression.float()
