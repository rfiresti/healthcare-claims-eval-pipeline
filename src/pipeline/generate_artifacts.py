from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jsonschema
from fpdf import FPDF
from pypdf import PdfReader

from src.generators.denial_generator import generate_denials, load_claims
from src.generators.eob_generator import generate_eobs
from src.generators.policy_generator import generate_policies


def _ascii_safe(text: Any) -> str:
    s = str(text)
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def _write_kv_pdf(lines: list[str], pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    width = pdf.epw
    for line in lines:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            width,
            5.5,
            _ascii_safe(line),
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.output(str(pdf_path))


def _load_schema(schema_path: Path) -> dict[str, Any]:
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _policy_to_kv_lines(policy: dict[str, Any], tenant_id: str) -> list[str]:
    lines = [
        f"Document Type: PolicyDocument",
        f"Tenant ID: {tenant_id}",
        f"Policy ID: {policy['policy_id']}",
        f"Payer Name: {policy['payer_name']}",
        f"Effective Date: {policy['effective_date']}",
        f"Clause Count: {len(policy['clauses'])}",
    ]
    for idx, clause in enumerate(policy["clauses"], start=1):
        lines.append(f"Clause {idx} ID: {clause['clause_id']}")
        lines.append(f"Clause {idx} Category: {clause['category']}")
        lines.append(f"Clause {idx} Text: {clause['clause_text']}")
    return lines


def _eob_to_kv_lines(eob: dict[str, Any], tenant_id: str) -> list[str]:
    return [
        "Document Type: EOBDocument",
        f"Tenant ID: {tenant_id}",
        f"EOB ID: {eob['eob_id']}",
        f"Claim ID: {eob['claim_id']}",
        f"Patient ID: {eob['patient_id']}",
        f"Service Date: {eob.get('service_date', '')}",
        f"Billed Amount: {eob['billed_amount']:.2f}",
        f"Allowed Amount: {eob['allowed_amount']:.2f}",
        f"Paid Amount: {eob['paid_amount']:.2f}",
        f"Adjudication Notes: {eob.get('adjudication_notes', '')}",
    ]


def _denial_to_kv_lines(denial: dict[str, Any], tenant_id: str) -> list[str]:
    cited = ", ".join(denial["cited_clause_ids"])
    return [
        "Document Type: DenialLetter",
        f"Tenant ID: {tenant_id}",
        f"Denial ID: {denial['denial_id']}",
        f"Claim ID: {denial['claim_id']}",
        f"Denial Reason Code: {denial['denial_reason_code']}",
        f"Cited Clause IDs: {cited}",
        f"Denial Summary: {denial['denial_summary']}",
    ]


def _verify_pdf_text(pdf_path: Path, expected_token: str) -> None:
    reader = PdfReader(str(pdf_path))
    extracted = "\n".join((page.extract_text() or "") for page in reader.pages)
    if expected_token not in extracted:
        raise RuntimeError(f"Expected '{expected_token}' in extracted text from {pdf_path}")


def run_export(
    claims_path: Path,
    schemas_dir: Path,
    output_dir: Path,
    tenants: list[str],
    policy_count: int,
    seed: int,
    pdf_sample_limit: int,
) -> None:
    claims = load_claims(claims_path)
    policy_schema = _load_schema(schemas_dir / "policy_schema.json")
    eob_schema = _load_schema(schemas_dir / "eob_schema.json")
    denial_schema = _load_schema(schemas_dir / "denial_schema.json")

    policies = generate_policies(count=policy_count, seed=seed)
    eobs = generate_eobs(claims=claims, seed=seed)
    denials = generate_denials(claims=claims)

    for p in policies:
        jsonschema.validate(instance=p, schema=policy_schema)
    for e in eobs:
        jsonschema.validate(instance=e, schema=eob_schema)
    for d in denials:
        jsonschema.validate(instance=d, schema=denial_schema)

    for tenant_id in tenants:
        t_dir = output_dir / tenant_id
        json_dir = t_dir / "json"
        pdf_dir = t_dir / "pdfs"
        json_dir.mkdir(parents=True, exist_ok=True)

        (json_dir / "policies.json").write_text(json.dumps(policies, indent=2), encoding="utf-8")
        (json_dir / "eobs.json").write_text(json.dumps(eobs, indent=2), encoding="utf-8")
        (json_dir / "denials.json").write_text(json.dumps(denials, indent=2), encoding="utf-8")

        for p in policies[:pdf_sample_limit]:
            out_pdf = pdf_dir / "policies" / f"{p['policy_id']}.pdf"
            _write_kv_pdf(_policy_to_kv_lines(p, tenant_id), out_pdf)
            _verify_pdf_text(out_pdf, p["policy_id"])

        for e in eobs[:pdf_sample_limit]:
            out_pdf = pdf_dir / "eobs" / f"{e['eob_id']}.pdf"
            _write_kv_pdf(_eob_to_kv_lines(e, tenant_id), out_pdf)
            _verify_pdf_text(out_pdf, e["eob_id"])

        for d in denials[:pdf_sample_limit]:
            out_pdf = pdf_dir / "denials" / f"{d['denial_id']}.pdf"
            _write_kv_pdf(_denial_to_kv_lines(d, tenant_id), out_pdf)
            _verify_pdf_text(out_pdf, d["denial_id"])

    print(
        f"Exported & validated {len(policies)} policies, {len(eobs)} EOBs, and "
        f"{len(denials)} denials across {len(tenants)} tenant(s) under '{output_dir}'."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic JSON and Key:Value PDF artifacts per tenant."
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=Path("data/synthetic_output/labeled_claims.json"),
        help="Path to labeled_claims.json",
    )
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        default=Path("data/schemas"),
        help="Directory containing JSON schema files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic_output/artifacts"),
        help="Root output directory for generated artifacts",
    )
    parser.add_argument(
        "--tenants",
        nargs="+",
        default=["tenant_1", "tenant_2"],
        help="One or more tenant_id values for artifact partitioning",
    )
    parser.add_argument(
        "--policy-count",
        type=int,
        default=2,
        help="Number of synthetic policy documents to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed",
    )
    parser.add_argument(
        "--pdf-sample-limit",
        type=int,
        default=10,
        help="Number of PDFs per document type to generate and verify per tenant (0 for all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    limit = None if args.pdf_sample_limit == 0 else args.pdf_sample_limit
    run_export(
        claims_path=args.claims,
        schemas_dir=args.schemas_dir,
        output_dir=args.output_dir,
        tenants=args.tenants,
        policy_count=args.policy_count,
        seed=args.seed,
        pdf_sample_limit=limit or 10**9,
    )
