from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CATEGORY_CODE_PREFIX: dict[str, str] = {
    "Exclusion": "EXC",
    "PreAuth": "PA",
    "Limitation": "LIM",
    "MedicalNecessity": "MED",
}


def _build_reason_code(clause_id: str | None, category: str | None, idx: int) -> str:
    prefix = CATEGORY_CODE_PREFIX.get(category or "", "DEN")
    if clause_id and "-" in clause_id:
        suffix = clause_id.split("-", 1)[1]
        return f"{prefix}-{suffix}"
    return f"{prefix}-{idx:04d}"


def build_denial_from_claim(
    claim: dict[str, Any],
    denial_num: int,
) -> dict[str, Any]:
    claim_id: str = claim["claim_id"]
    clause_id: str | None = claim.get("ground_truth_clause_id")
    category: str | None = claim.get("ground_truth_category")
    denial_details: dict[str, Any] = claim.get("denial_details") or {}

    reason_code = _build_reason_code(clause_id=clause_id, category=category, idx=denial_num)
    base_reason = denial_details.get("reason") or (
        f"Non-compliance with policy clause {clause_id}"
        if clause_id
        else "Claim denied under plan adjudication rules"
    )
    clause_rule_text = denial_details.get("text") or ""

    if clause_rule_text:
        summary = (
            f"{base_reason}. Governing provision: {clause_rule_text} "
            f"Claim {claim_id} was denied; supporting clinical or authorization "
            f"documentation is required to initiate an appeal."
        )
    else:
        summary = (
            f"{base_reason}. Claim {claim_id} was denied; supporting "
            f"documentation is required to initiate an appeal."
        )

    cited_clauses = [clause_id] if clause_id else ["CLS-1001"]

    return {
        "denial_id": f"DEN-{denial_num:06d}",
        "claim_id": claim_id,
        "denial_reason_code": reason_code,
        "denial_summary": summary,
        "cited_clause_ids": cited_clauses,
    }


def generate_denials(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    denied_claims = [c for c in claims if c.get("status") == "DENIED"]
    return [
        build_denial_from_claim(claim=claim, denial_num=idx)
        for idx, claim in enumerate(denied_claims, start=1)
    ]


def load_claims(claims_path: str | Path = "data/synthetic_output/labeled_claims.json") -> list[dict[str, Any]]:
    with Path(claims_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of claims in {claims_path}")
    return data


def save_denials(denials: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(denials, f, indent=2)
    return path
