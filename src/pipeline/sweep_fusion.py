#!/usr/bin/env python3
"""Dev-only sweep of the hybrid keyword weight (w_kw, with w_vec = 1.0) for retrieve().

Uses the same claim loading, query building and scoring as eval_retrieval.py. Run it on the dev
set only; the held-out test set must not be used to choose weights.

    python3 src/pipeline/sweep_fusion.py \
        --tenant-id 00000000-0000-0000-0000-000000000001 \
        --tenant-id 00000000-0000-0000-0000-000000000002 \
        --ids-file eval/dev_ids.txt --query paraphrase

Writes eval/runs/fusion_sweep_<split>_<query>_<timestamp>.json and prints a table. Changes no settings.
"""
import argparse
import datetime
import json
import os
import sys

import eval_retrieval as er

DEFAULT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tenant-id", action="append", required=True)
    ap.add_argument("--ids-file", required=True, help="dev IDs only (eval/dev_ids.txt)")
    ap.add_argument("--query", default="paraphrase", choices=["both", "reason", "text", "paraphrase"])
    ap.add_argument("--denied-status", default=None)
    ap.add_argument("--w-kw", type=float, nargs="+", default=list(DEFAULT_WEIGHTS))
    ap.add_argument("--out-dir", default="eval/runs")
    a = ap.parse_args()

    if "test" in os.path.basename(a.ids_file):
        sys.exit("Refusing to tune on the test set. Use eval/dev_ids.txt.")
    with open(a.ids_file) as f:
        ids = {line.strip() for line in f if line.strip()}
    split = "dev" if "dev" in os.path.basename(a.ids_file) else "custom"

    dsn = os.environ.get("APP_DB_URI")
    if not dsn:
        sys.exit("Set APP_DB_URI (app_user connection string).")
    labels_dsn = os.environ.get("ADMIN_DB_URI") or dsn

    import psycopg
    from retrieve import check_tenant, retrieve

    started = datetime.datetime.now()
    per = {w: {} for w in a.w_kw}
    pooled = {w: [] for w in a.w_kw}
    with psycopg.connect(dsn) as conn, psycopg.connect(labels_dsn) as lconn:
        for tenant in a.tenant_id:
            tenant = check_tenant(tenant)
            claims, _ = er.load_claims(lconn, tenant, a.denied_status)
            claims = [c for c in claims if c["claim_id"] in ids]
            if not claims:
                sys.exit("No claims for tenant %s after --ids-file." % tenant)
            for w in a.w_kw:
                rows = er.evaluate(lambda q, k, w=w: retrieve(conn, tenant, q, k=k, w_vec=1.0, w_kw=w),
                                   claims, a.query, scrub=True)
                per[w][tenant] = er.summarize(rows)
                pooled[w] += rows

    print("w_kw = keyword weight (w_vec = 1.0); 0 = vector only, 1.0 = current hybrid\n")
    print("%-6s %-12s %5s %7s %7s %7s %7s" % ("w_kw", "tenant", "n", "R@1", "R@2", "R@5", "MRR"))
    summary = {}
    for w in a.w_kw:
        for t, m in per[w].items():
            print("%-6.2f %-12s %5d %6.1f%% %6.1f%% %6.1f%% %7.3f" % (
                w, "…" + t[-4:], m["n"], 100 * m["recall@1"], 100 * m["recall@2"], 100 * m["recall@5"], m["mrr"]))
        p = er.summarize(pooled[w])
        summary[str(w)] = {"pooled": p, "by_tenant": per[w]}
        print("%-6.2f %-12s %5d %6.1f%% %6.1f%% %6.1f%% %7.3f\n" % (
            w, "pooled", p["n"], 100 * p["recall@1"], 100 * p["recall@2"], 100 * p["recall@5"], p["mrr"]))

    best = max(a.w_kw, key=lambda w: (round(summary[str(w)]["pooled"]["mrr"], 3), w))
    print("Best pooled MRR: w_kw = %.2f (ties go to the higher keyword weight)." % best)

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "fusion_sweep_%s_%s_%s.json" % (split, a.query, started.strftime("%Y%m%d-%H%M%S")))
    with open(out, "w") as f:
        json.dump({"kind": "fusion_sweep", "split": split, "ids_file": a.ids_file, "query": a.query,
                   "w_vec": 1.0, "w_kw_values": a.w_kw, "best_w_kw_by_mrr": best,
                   "git_commit": er.git_commit(), "started": started.isoformat(timespec="seconds"),
                   "paraphrases": er.paraphrase_meta(), "results": summary}, f, indent=1, default=str)
    print("Wrote", out)


if __name__ == "__main__":
    main()

