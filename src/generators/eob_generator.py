from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any


def _derive_service_date(idx: int, base_date: str = "2026-02-01") -> str:
    start = date.fromisoformat(base_date)
    offset_days = (idx * 3) % 270
    return (start + timedelta(days=offset_days)).isoformat()


def build_eob_from_claim(
    claim: dict[str, Any],
    eob_num: int,
    seed: int = 42,
) -> dict[str, Any]:
    claim_id: str = claim["claim_id"]
    patient_id: str = claim["patient_id"]
    billed_amount = round(float(claim["billed_amount"]), 2)
    status: str = claim.get("status", "DENIED")
    clause_id: str | None = claim.get("ground_truth_clause_id")
    category: str | None = claim.get("ground_truth_category")

    rng = random.Random(f"{seed}:{claim_id}:{eob_num}")
    service_date = _derive_service_date(eob_num)

    if status == "APPROVED":
        allowed_ratio = rng.uniform(0.75, 0.95)
        paid_ratio = rng.uniform(0.75, 0.90)
        allowed_amount = round(min(billed_amount, billed_amount * allowed_ratio), 2)
        paid_amount = round(min(allowed_amount, allowed_amount * paid_ratio), 2)
        notes = (
            f"Approved under {category or 'General'} review"
            f"{f' (ref {clause_id})' if clause_id else ''}; "
            f"member cost-share applied per plan benefit terms."
        )
    else:
        allowed_amount = 0.0
        paid_amount = 0.0
        denial_info = claim.get("denial_details") or {}
        reason_text = denial_info.get("reason") or (
            f"Claim denied under {clause_id}" if clause_id else "Claim denied"
        )
        notes = f"Not payable: {reason_text}."

    return {
        "eob_id": f"EOB-{eob_num:06d}",
        "claim_id": claim_id,
        "patient_id": patient_id,
        "service_date": service_date,
        "billed_amount": billed_amount,
        "allowed_amount": allowed_amount,
        "paid_amount": paid_amount,
        "adjudication_notes": notes,
    }


def generate_eobs(
    claims: list[dict[str, Any]],
    seed: int = 42,
) -> list[dict[str, Any]]:
    return [
        build_eob_from_claim(claim=claim, eob_num=idx, seed=seed)
        for idx, claim in enumerate(claims, start=1)
    ]


def load_claims(claims_path: str | Path = "data/synthetic_output/labeled_claims.json") -> list[dict[str, Any]]:
    with Path(claims_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of claims in {claims_path}")
    return data


def save_eobs(eobs: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(eobs, f, indent=2)
    return path
