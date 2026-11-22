from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import jsonschema
from pypdf import PdfReader

KNOWN_KEYS = [
    "Document Type",
    "Tenant ID",
    "Denial ID",
    "EOB ID",
    "Policy ID",
    "Claim ID",
    "Patient ID",
    "Service Date",
    "Billed Amount",
    "Allowed Amount",
    "Paid Amount",
    "Adjudication Notes",
    "Denial Reason Code",
    "Cited Clause IDs",
    "Denial Summary",
    "Payer Name",
    "Effective Date",
    "Clause Count",
]

KEY_PATTERN = re.compile(
    r"^(?P<key>" + "|".join(re.escape(k) for k in KNOWN_KEYS) + r"):\s*(?P<val>.*)$"
)


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def parse_kv_text(raw_text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    current_key: str | None = None

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = KEY_PATTERN.match(line)
        if match:
            current_key = match.group("key")
            parsed[current_key] = match.group("val").strip()
        elif current_key is not None:
            parsed[current_key] = f"{parsed[current_key]} {line}".strip()

    return parsed


def verify_tenant(tenant_dir: Path, schemas_dir: Path) -> dict[str, int]:
    policy_schema = json.loads((schemas_dir / "policy_schema.json").read_text(encoding="utf-8"))
    eob_schema = json.loads((schemas_dir / "eob_schema.json").read_text(encoding="utf-8"))
    denial_schema = json.loads((schemas_dir / "denial_schema.json").read_text(encoding="utf-8"))

    policies: list[dict[str, Any]] = json.loads((tenant_dir / "json/policies.json").read_text(encoding="utf-8"))
    eobs: list[dict[str, Any]] = json.loads((tenant_dir / "json/eobs.json").read_text(encoding="utf-8"))
    denials: list[dict[str, Any]] = json.loads((tenant_dir / "json/denials.json").read_text(encoding="utf-8"))

    for item in policies:
        jsonschema.validate(instance=item, schema=policy_schema)
        assert item.get("policy_id"), f"Missing policy_id in {tenant_dir.name}"

    for item in eobs:
        jsonschema.validate(instance=item, schema=eob_schema)
        assert item["paid_amount"] <= item["allowed_amount"] <= item["billed_amount"], (
            f"Amount ordering failed for {item.get('eob_id')} in {tenant_dir.name}"
        )

    for item in denials:
        jsonschema.validate(instance=item, schema=denial_schema)
        assert item.get("claim_id"), f"Missing claim_id in {tenant_dir.name}"

    pdf_files = sorted((tenant_dir / "pdfs").rglob("*.pdf"))
    assert pdf_files, f"No PDF files found under {tenant_dir / 'pdfs'}"

    for pdf_path in pdf_files:
        kv = parse_kv_text(extract_pdf_text(pdf_path))
        assert kv.get("Document Type"), f"Missing Document Type in {pdf_path}"
        assert kv.get("Tenant ID") == tenant_dir.name, (
            f"Tenant ID mismatch in {pdf_path}: got {kv.get('Tenant ID')}, expected {tenant_dir.name}"
        )

    return {
        "policies": len(policies),
        "eobs": len(eobs),
        "denials": len(denials),
        "pdfs": len(pdf_files),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify generated JSON and Key:Value PDF artifacts.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("data/synthetic_output/artifacts"),
        help="Root directory containing per-tenant artifact folders",
    )
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        default=Path("data/schemas"),
        help="Directory containing JSON schema files",
    )
    parser.add_argument(
        "--tenants",
        nargs="*",
        default=None,
        help="Optional subset of tenant directory names to verify",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.artifacts_dir.exists(), f"Artifacts directory not found: {args.artifacts_dir}"

    if args.tenants:
        tenant_dirs = [args.artifacts_dir / t for t in args.tenants]
    else:
        tenant_dirs = [p for p in sorted(args.artifacts_dir.iterdir()) if p.is_dir()]

    assert tenant_dirs, f"No tenant directories found under {args.artifacts_dir}"

    for t_dir in tenant_dirs:
        assert t_dir.is_dir(), f"Tenant directory does not exist: {t_dir}"
        counts = verify_tenant(t_dir, args.schemas_dir)
        print(
            f"[OK] {t_dir.name}: policies={counts['policies']}, eobs={counts['eobs']}, "
            f"denials={counts['denials']}, pdfs={counts['pdfs']}"
        )


if __name__ == "__main__":
    main()
