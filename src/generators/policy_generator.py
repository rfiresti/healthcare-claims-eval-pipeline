from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


BASE_CLAUSES: list[dict[str, str]] = [
    {
        "clause_id": "CLS-1001",
        "category": "Exclusion",
        "base_rule": "Experimental procedures are strictly excluded from coverage.",
        "detail": (
            "Services, devices, or therapies classified as investigational, experimental, "
            "or unproven by peer-reviewed clinical evidence or standard medical guidelines "
            "are not eligible for benefits. Routine clinical trials require separate written "
            "plan approval prior to service."
        ),
    },
    {
        "clause_id": "CLS-1002",
        "category": "PreAuth",
        "base_rule": "Advanced imaging requires prior authorization within 48 hours.",
        "detail": (
            "High-cost diagnostic imaging—including MRI, CT, and PET scans—requires "
            "prior authorization before non-emergency scheduling, or retroactive notification "
            "within 48 hours for urgent/emergency encounters. Missing authorization may "
            "result in claim denial or benefit reduction."
        ),
    },
    {
        "clause_id": "CLS-1003",
        "category": "Limitation",
        "base_rule": "Physical therapy capped at 20 visits per calendar year.",
        "detail": (
            "Outpatient physical therapy benefits are limited to a maximum of 20 visits "
            "per member per calendar year across all providers. Visits exceeding the annual "
            "cap are not covered unless an approved clinical extension applies."
        ),
    },
    {
        "clause_id": "CLS-1004",
        "category": "MedicalNecessity",
        "base_rule": "Inpatient admission requires proof of clinical deterioration.",
        "detail": (
            "Acute inpatient hospital admission requires documented clinical necessity, "
            "including objective evidence of clinical deterioration, hemodynamic or "
            "respiratory instability, or treatment needs that cannot be safely managed "
            "at a lower level of care."
        ),
    },
]

SUPPLEMENTARY_CLAUSES: list[dict[str, str]] = [
    {
        "clause_id": "CLS-1005",
        "category": "PreAuth",
        "base_rule": "Outpatient specialty infusions require pre-service authorization.",
        "detail": (
            "Non-emergent biologic or specialty medication infusions administered in "
            "outpatient clinics or home settings require prior authorization with "
            "supporting diagnosis and dosing records."
        ),
    },
    {
        "clause_id": "CLS-1006",
        "category": "Limitation",
        "base_rule": "Behavioral health outpatient sessions limited to 40 visits per year.",
        "detail": (
            "Routine outpatient psychotherapy and behavioral health counseling visits "
            "are subject to an annual visit limit of 40 sessions per calendar year, "
            "subject to standard copay or coinsurance terms."
        ),
    },
    {
        "clause_id": "CLS-1007",
        "category": "Exclusion",
        "base_rule": "Cosmetic services without functional impairment are excluded.",
        "detail": (
            "Procedures performed primarily to improve appearance or self-esteem without "
            "documented physiological or functional impairment are excluded from coverage."
        ),
    },
    {
        "clause_id": "CLS-1008",
        "category": "MedicalNecessity",
        "base_rule": "Durable medical equipment requires documented functional need.",
        "detail": (
            "Durable medical equipment (DME) rentals or purchases require a provider "
            "order establishing medical necessity and expected functional improvement in "
            "daily living activities."
        ),
    },
]


def _format_clause_text(item: dict[str, str], variant_idx: int) -> str:
    addenda = [
        "Documentation must accompany claim submission to support adjudication.",
        "Appeals require clinical records demonstrating compliance with plan criteria.",
        "Standard member cost-share rules apply when coverage conditions are met.",
    ]
    tail = addenda[variant_idx % len(addenda)]
    return f"{item['base_rule']} {item['detail']} {tail}"


def build_policy_document(
    policy_num: int = 1,
    payer_name: str = "Midwest Health Plan",
    effective_date: str = "2026-01-01",
    include_supplementary: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed + policy_num)
    source_clauses = list(BASE_CLAUSES)
    if include_supplementary:
        extra_count = rng.randint(1, len(SUPPLEMENTARY_CLAUSES))
        picked = rng.sample(SUPPLEMENTARY_CLAUSES, k=extra_count)
        source_clauses.extend(picked)

    clauses = [
        {
            "clause_id": entry["clause_id"],
            "category": entry["category"],
            "clause_text": _format_clause_text(entry, policy_num + idx),
        }
        for idx, entry in enumerate(source_clauses)
    ]

    return {
        "policy_id": f"POL-{policy_num:05d}",
        "payer_name": payer_name,
        "effective_date": effective_date,
        "clauses": clauses,
    }


def generate_policies(
    count: int = 2,
    payer_names: list[str] | None = None,
    effective_date: str = "2026-01-01",
    include_supplementary: bool = True,
    seed: int = 42,
) -> list[dict[str, Any]]:
    payers = payer_names or ["Midwest Health Plan", "Heartland Commercial PPO"]
    policies: list[dict[str, Any]] = []
    for idx in range(1, count + 1):
        payer = payers[(idx - 1) % len(payers)]
        policies.append(
            build_policy_document(
                policy_num=idx,
                payer_name=payer,
                effective_date=effective_date,
                include_supplementary=include_supplementary,
                seed=seed,
            )
        )
    return policies


def save_policies(policies: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(policies, f, indent=2)
    return path
