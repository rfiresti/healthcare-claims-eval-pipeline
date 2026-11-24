#!/usr/bin/env python3
"""Fixed-seed dev/test split of denied claims, stratified by tenant x category.

Writes one claim ID per line to eval/dev_ids.txt and eval/test_ids.txt, the
format eval_grounding.py --ids-file reads. Refuses to overwrite an existing
split unless --force is given: the split must stay fixed once results exist.

Run on the DB server, from the repo root, with the venv active and
ADMIN_DB_URI exported (labels are only readable as admin_owner):

    python3 src/pipeline/make_split.py
"""
import argparse
import os
import random
import sys
from collections import defaultdict

from eval_retrieval import load_claims

DEFAULT_TENANTS = [
    "00000000-0000-0000-0000-000000000001",
    "00000000-0000-0000-0000-000000000002",
]


def field(row, key):
    """Claim rows may be dicts or attribute-style rows."""
    return row[key] if isinstance(row, dict) else getattr(row, key)


def load_denied(conn, tenants, denied_status):
    """Return [(tenant_id, claim_id, category)] for every denied claim."""
    out = []
    for tenant in tenants:
        claims, status_counts = load_claims(conn, tenant, denied_status)
        if not claims:
            sys.exit(f"No '{denied_status}' claims for tenant {tenant}. "
                     f"Statuses seen: {status_counts}. Check --denied-status.")
        for c in claims:
            out.append((tenant, field(c, "claim_id"), field(c, "ground_truth_category")))
    ids = [cid for _, cid, _ in out]
    if len(ids) != len(set(ids)):
        sys.exit("Claim IDs repeat across tenants, so a plain ID list is ambiguous. "
                 "Stopping; the ids-file format needs a tenant column first.")
    return out


def stratified_split(rows, dev_frac, seed):
    strata = defaultdict(list)
    for tenant, cid, cat in rows:
        strata[(tenant, cat)].append(cid)
    rng = random.Random(seed)
    dev, test, table = [], [], []
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        n_dev = round(len(ids) * dev_frac)
        dev += ids[:n_dev]
        test += ids[n_dev:]
        table.append((key[0], key[1], len(ids), n_dev, len(ids) - n_dev))
    return sorted(dev), sorted(test), table


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tenant-id", action="append", dest="tenants",
                    help="repeatable; default: the two eval tenants")
    ap.add_argument("--denied-status", default="DENIED",
                    help="claim_status value for denied claims (same as eval_retrieval.py)")
    ap.add_argument("--dev-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=516)
    ap.add_argument("--out-dir", default="eval")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = ap.parse_args()

    dev_path = os.path.join(args.out_dir, "dev_ids.txt")
    test_path = os.path.join(args.out_dir, "test_ids.txt")
    if not args.force and (os.path.exists(dev_path) or os.path.exists(test_path)):
        sys.exit(f"{dev_path} or {test_path} already exists. The split is meant to stay "
                 "fixed; pass --force only if no results depend on it yet.")

    uri = os.environ.get("ADMIN_DB_URI")
    if not uri:
        sys.exit("ADMIN_DB_URI is not set (see the session setup block).")
    import psycopg  # imported late so --help works outside the venv

    with psycopg.connect(uri) as conn:
        rows = load_denied(conn, args.tenants or DEFAULT_TENANTS, args.denied_status)

    dev, test, table = stratified_split(rows, args.dev_frac, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    for path, ids in ((dev_path, dev), (test_path, test)):
        with open(path, "w") as f:
            f.write("\n".join(ids) + "\n")

    print(f"{'tenant':<38} {'category':<18} {'total':>5} {'dev':>4} {'test':>4}")
    for tenant, cat, n, nd, nt in table:
        print(f"{tenant:<38} {str(cat):<18} {n:>5} {nd:>4} {nt:>4}")
    print(f"\n{len(rows)} denied claims -> dev {len(dev)}, test {len(test)} "
          f"(seed {args.seed}, dev_frac {args.dev_frac})")
    print(f"Wrote {dev_path} and {test_path}. Commit both by path.")


if __name__ == "__main__":
    main()

