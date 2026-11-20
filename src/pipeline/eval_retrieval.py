"""Week 4, Task 3: baseline retrieval evaluation (Recall@5 against the labeled claims).

For every denied claim in labeled_claims.json, build a query from the denial text, run retrieve() as a
tenant, and check whether the claim's ground_truth_clause_id is among the top-k returned chunks.

    PYTHONPATH=src/pipeline python src/pipeline/eval_retrieval.py \
        --tenant-id <uuid> [--tenant-id <uuid> ...] \
        --claims data/synthetic_output/labeled_claims.json --out eval_retrieval.json

Connects as app_user through APP_DB_URI (never a superuser: RLS must apply). Needs the policy chunks to be
loaded for each tenant, with clause_id values that match ground_truth_clause_id.

What it reports: Recall@1/3/5, MRR, a per-category breakdown, the misses, and an ablation (vector only,
keyword only, hybrid) so the report can say what the hybrid step adds. Target from the runbook: Recall@5 >= 85%.
Citation Grounding (target >= 90%) needs the draft stage and is not measured here.

A hit means: any of the top-k chunks has clause_id == ground_truth_clause_id. A clause split into several
chunks can fill several of the k slots.
"""
import argparse
import collections
import json
import os
import sys

TARGET_RECALL_AT_5 = 0.85
KS = (1, 3, 5)


def build_query(claim, mode):
    d = claim["denial_details"]
    reason, text = d.get("reason") or "", d.get("text") or ""
    return {"reason": reason, "text": text, "both": (reason + ". " + text).strip(". ")}[mode]


def rank_of(clause_ids, gt):
    """1-based rank of the first chunk whose clause_id is gt, or None."""
    for i, cid in enumerate(clause_ids, start=1):
        if cid == gt:
            return i
    return None


def summarize(rows, ks=KS):
    """rows: [{"rank": int|None, ...}] -> metrics dict."""
    n = len(rows)
    out = {"n": n}
    for k in ks:
        out["recall@%d" % k] = round(sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n, 4) if n else None
    out["mrr"] = round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 4) if n else None
    return out


def evaluate(retrieve_fn, claims, query_mode, k_max=max(KS)):
    """retrieve_fn(query, k) -> [{"clause_id": ...}, ...]. Returns per-claim rows."""
    rows = []
    for c in claims:
        q = build_query(c, query_mode)
        hits = retrieve_fn(q, k_max)
        ids = [h["clause_id"] for h in hits]
        rows.append({"claim_id": c["claim_id"], "category": c.get("ground_truth_category"),
                     "gt": c["ground_truth_clause_id"], "top": ids, "rank": rank_of(ids, c["ground_truth_clause_id"])})
    return rows


def loaded_clause_ids(conn, tenant_id, model_name):
    """clause_ids this tenant can see (through RLS), to check they cover the ground-truth ids."""
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT set_config('app.current_tenant_id', %s, true)", (tenant_id,))
        cur.execute("SELECT DISTINCT clause_id FROM vector_embeddings WHERE model_name = %s", (model_name,))
        return {r[0] for r in cur.fetchall()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", default="data/synthetic_output/labeled_claims.json")
    ap.add_argument("--tenant-id", action="append", required=True, help="repeat for several tenants")
    ap.add_argument("--query", choices=["both", "reason", "text"], default="both",
                    help="which denial_details fields form the query (default: both)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N denied claims (smoke test)")
    ap.add_argument("--force", action="store_true", help="run even if some ground-truth clause_ids are not loaded")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--out", default="eval_retrieval.json")
    a = ap.parse_args()

    dsn = os.environ.get("APP_DB_URI")
    if not dsn:
        sys.exit("Set APP_DB_URI (app_user connection string).")

    import psycopg
    from embedding_settings import MODEL_NAME
    from retrieve import check_tenant, retrieve

    claims = [c for c in json.load(open(a.claims)) if c.get("status") == "denied"]
    skipped = [c for c in claims if not c.get("ground_truth_clause_id")]
    claims = [c for c in claims if c.get("ground_truth_clause_id")]
    if a.limit:
        claims = claims[:a.limit]
    print("denied claims with a ground truth: %d (skipped without one: %d)" % (len(claims), len(skipped)))
    gt_ids = {c["ground_truth_clause_id"] for c in claims}

    variants = [("hybrid", dict(w_vec=1.0, w_kw=1.0))]
    if not a.no_ablation:
        variants += [("vector only", dict(w_vec=1.0, w_kw=0.0)), ("keyword only", dict(w_vec=0.0, w_kw=1.0))]

    result = {"query_mode": a.query, "k_values": list(KS), "target_recall_at_5": TARGET_RECALL_AT_5, "tenants": {}}
    ok = True
    with psycopg.connect(dsn) as conn:
        for tenant in a.tenant_id:
            tenant = check_tenant(tenant)
            have = loaded_clause_ids(conn, tenant, MODEL_NAME)
            missing = sorted(gt_ids - have)
            print("\n=== tenant %s ===" % tenant)
            print("clause_ids visible: %d; ground-truth ids: %d; ground-truth ids not loaded: %d" %
                  (len(have), len(gt_ids), len(missing)))
            if not have:
                print("No policy chunks visible for this tenant. Load them first (embed_and_load.py).")
                ok = False
                continue
            if missing:
                print("  not loaded (first 10): %s" % missing[:10])
                print("  Claims pointing at these ids can never score a hit, so Recall@5 would read low for a")
                print("  data reason, not a retrieval one. Check that the loaded clause_id format matches.")
                if not a.force:
                    print("  Skipping this tenant; re-run with --force to score anyway.")
                    ok = False
                    continue
            t = {"clauses_visible": len(have), "gt_not_loaded": missing, "variants": {}}
            for name, weights in variants:
                rows = evaluate(lambda q, k: retrieve(conn, tenant, q, k=k, **weights), claims, a.query)
                m = summarize(rows)
                by_cat = collections.defaultdict(list)
                for r in rows:
                    by_cat[r["category"]].append(r)
                m["by_category"] = {c: summarize(v) for c, v in sorted(by_cat.items(), key=lambda kv: str(kv[0]))}
                m["misses"] = [r for r in rows if not r["rank"] or r["rank"] > max(KS)]
                t["variants"][name] = m
            result["tenants"][tenant] = t

            print("\n%-14s %5s %9s %9s %9s %7s" % ("variant", "n", "R@1", "R@3", "R@5", "MRR"))
            for name, m in t["variants"].items():
                print("%-14s %5d %8.1f%% %8.1f%% %8.1f%% %7.3f" % (
                    name, m["n"], 100 * m["recall@1"], 100 * m["recall@3"], 100 * m["recall@5"], m["mrr"]))
            h = t["variants"]["hybrid"]
            verdict = "PASS" if h["recall@5"] >= TARGET_RECALL_AT_5 else "BELOW TARGET"
            print("\nhybrid Recall@5 = %.1f%% vs target %.0f%%: %s" % (100 * h["recall@5"], 100 * TARGET_RECALL_AT_5, verdict))
            print("by category (hybrid):")
            for c, m in h["by_category"].items():
                print("  %-28s n=%-4d R@5 %5.1f%%" % (c, m["n"], 100 * m["recall@5"]))
            print("misses (hybrid): %d" % len(h["misses"]))
            for r in h["misses"][:5]:
                print("  %s wanted %s, got %s" % (r["claim_id"], r["gt"], r["top"]))
            ok = ok and h["recall@5"] >= TARGET_RECALL_AT_5

    json.dump(result, open(a.out, "w"), indent=1)
    print("\nWrote", a.out)
    print("Caveat for the report: the query is built from the claim's denial text. If that text repeats the")
    print("clause wording, results overstate real retrieval; compare --query reason (harder) with --query both.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

