"""Week 4, Task 3: baseline retrieval evaluation (Recall@k against claim_labels).

For every denied claim of a tenant, build a query from the claim's denial text (claims.details), run
retrieve() as that tenant, and check whether claim_labels.ground_truth_clause_id is among the top-k chunks.
Claims and labels are read through RLS as the tenant, so only that tenant's rows are visible.

    PYTHONPATH=src/pipeline python src/pipeline/eval_retrieval.py \
        --tenant-id 00000000-0000-0000-0000-000000000001 \
        --tenant-id 00000000-0000-0000-0000-000000000002 --out eval_retrieval.json

    add --limit 20 --show-sample first, to smoke-test and see what query text is being built.

Connects as app_user through APP_DB_URI (never a superuser: RLS must apply). The policy chunks must already be
loaded for each tenant in vector_embeddings, with clause_id values equal to ground_truth_clause_id.

Reports Recall@1/2/3/5, MRR, a per-category breakdown, misses, and an ablation (vector only, keyword only, hybrid).
Runbook target: Recall@5 >= 85%. Read that number with care: when the corpus has only a few distinct clauses
(as of Nov 23, four: CLS-1001..1004), the top 5 chunks nearly always contain the right clause, so Recall@5 is close
to 100% whatever the retrieval quality. The script warns in that case; Recall@1, @2 and MRR are the informative ones.
Citation Grounding (target >= 90%) needs the draft stage and is not measured here.

A hit means: any of the top-k chunks has clause_id == ground_truth_clause_id.
"""
import argparse
import collections
import json
import os
import sys

TARGET_RECALL_AT_5 = 0.85
KS = (1, 2, 3, 5)
REASON_KEYS = ("reason", "denial_reason", "denial_reason_text", "reason_text")
TEXT_KEYS = ("text", "denial_text", "policy_detail", "description", "denial_description")


def denial_fields(details):
    """Pull (reason, text) out of claims.details, which may nest them under denial_details."""
    d = details.get("denial_details") if isinstance(details.get("denial_details"), dict) else details
    reason = next((str(d[k]) for k in REASON_KEYS if d.get(k)), "")
    text = next((str(d[k]) for k in TEXT_KEYS if d.get(k)), "")
    if not reason and not text:
        raise ValueError("no denial reason or text found in claims.details; keys seen: %s" % sorted(d.keys()))
    return reason, text


def build_query(claim, mode):
    reason, text = denial_fields(claim["details"])
    return {"reason": reason, "text": text, "both": (reason + ". " + text).strip(". ")}[mode]


def rank_of(clause_ids, gt):
    """1-based rank of the first chunk whose clause_id is gt, or None."""
    for i, cid in enumerate(clause_ids, start=1):
        if cid == gt:
            return i
    return None


def summarize(rows, ks=KS):
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
        ids = [h["clause_id"] for h in retrieve_fn(build_query(c, query_mode), k_max)]
        rows.append({"claim_id": c["claim_id"], "category": c.get("ground_truth_category"),
                     "gt": c["ground_truth_clause_id"], "top": ids, "rank": rank_of(ids, c["ground_truth_clause_id"])})
    return rows


def with_tenant(conn, tenant_id):
    """Open a transaction with the tenant set, for reads outside retrieve()."""
    cm = conn.transaction()
    cm.__enter__()
    cur = conn.cursor()
    cur.execute("SELECT set_config('app.current_tenant_id', %s, true)", (tenant_id,))
    return cm, cur


def load_claims(conn, tenant_id, denied_status):
    """Denied claims joined to their labels, as the tenant. Returns (claims, status_counts)."""
    cm, cur = with_tenant(conn, tenant_id)
    try:
        cur.execute("SELECT claim_status, count(*) FROM claims GROUP BY 1 ORDER BY 1")
        counts = {r[0]: r[1] for r in cur.fetchall()}
        wanted = [denied_status] if denied_status else [s for s in counts if "den" in (s or "").lower()]
        if not wanted:
            return [], counts
        cur.execute("""SELECT c.claim_id, c.claim_status, c.details, l.ground_truth_clause_id, l.ground_truth_category
                       FROM claims c JOIN claim_labels l USING (tenant_id, claim_id)
                       WHERE c.claim_status = ANY(%s) ORDER BY c.claim_id""", (wanted,))
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()], counts
    finally:
        cur.close()
        cm.__exit__(None, None, None)


def loaded_clause_ids(conn, tenant_id, model_name):
    cm, cur = with_tenant(conn, tenant_id)
    try:
        cur.execute("SELECT DISTINCT clause_id FROM vector_embeddings WHERE model_name = %s", (model_name,))
        return {r[0] for r in cur.fetchall()}
    finally:
        cur.close()
        cm.__exit__(None, None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tenant-id", action="append", required=True, help="repeat for several tenants")
    ap.add_argument("--denied-status", default=None,
                    help="claim_status value that means denied (default: any value containing 'den')")
    ap.add_argument("--query", choices=["both", "reason", "text"], default="both",
                    help="which denial fields form the query (default: both)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N denied claims per tenant (smoke test)")
    ap.add_argument("--show-sample", action="store_true", help="print one claim and the query built from it")
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

    variants = [("hybrid", dict(w_vec=1.0, w_kw=1.0))]
    if not a.no_ablation:
        variants += [("vector only", dict(w_vec=1.0, w_kw=0.0)), ("keyword only", dict(w_vec=0.0, w_kw=1.0))]

    result = {"query_mode": a.query, "k_values": list(KS), "target_recall_at_5": TARGET_RECALL_AT_5, "tenants": {}}
    ok = True
    with psycopg.connect(dsn) as conn:
        for tenant in a.tenant_id:
            tenant = check_tenant(tenant)
            print("\n=== tenant %s ===" % tenant)
            claims, counts = load_claims(conn, tenant, a.denied_status)
            print("claim_status values: %s" % counts)
            if a.limit:
                claims = claims[:a.limit]
            if not claims:
                print("No denied claims with labels visible. Pass --denied-status with the right value.")
                ok = False
                continue
            gt_ids = {c["ground_truth_clause_id"] for c in claims}
            print("denied claims scored: %d; distinct ground-truth clauses: %d %s" % (len(claims), len(gt_ids), sorted(gt_ids)))
            if a.show_sample:
                c = claims[0]
                print("sample claim %s -> %s\n  query (%s): %s" % (c["claim_id"], c["ground_truth_clause_id"], a.query, build_query(c, a.query)[:300]))

            have = loaded_clause_ids(conn, tenant, MODEL_NAME)
            missing = sorted(gt_ids - have)
            print("clause_ids visible in vector_embeddings: %d %s; ground-truth ids not loaded: %d" % (len(have), sorted(have)[:8], len(missing)))
            if not have:
                print("No policy chunks visible for this tenant. Load them first (embed_and_load.py).")
                ok = False
                continue
            if missing:
                print("  not loaded: %s" % missing[:10])
                print("  Claims pointing at these ids can never score a hit, so recall would read low for a data reason.")
                if not a.force:
                    print("  Skipping this tenant; re-run with --force to score anyway.")
                    ok = False
                    continue
            if len(have) <= max(KS):
                print("  WARNING: only %d distinct clauses are loaded, so the top %d chunks will nearly always contain the right" % (len(have), max(KS)))
                print("  one. Recall@5 will look near 100%% regardless of quality; read Recall@1, Recall@2 and MRR instead.")

            t = {"clauses_visible": sorted(have), "gt_not_loaded": missing, "status_counts": counts, "variants": {}}
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

            print("\n%-14s %5s %8s %8s %8s %8s %7s" % ("variant", "n", "R@1", "R@2", "R@3", "R@5", "MRR"))
            for name, m in t["variants"].items():
                print("%-14s %5d %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.3f" % (
                    name, m["n"], 100 * m["recall@1"], 100 * m["recall@2"], 100 * m["recall@3"], 100 * m["recall@5"], m["mrr"]))
            h = t["variants"]["hybrid"]
            print("\nhybrid Recall@5 = %.1f%% vs target %.0f%%: %s" % (
                100 * h["recall@5"], 100 * TARGET_RECALL_AT_5, "PASS" if h["recall@5"] >= TARGET_RECALL_AT_5 else "BELOW TARGET"))
            print("by category (hybrid):")
            for c, m in h["by_category"].items():
                print("  %-18s n=%-4d R@1 %5.1f%%  R@5 %5.1f%%" % (c, m["n"], 100 * m["recall@1"], 100 * m["recall@5"]))
            print("misses at k=5 (hybrid): %d" % len(h["misses"]))
            for r in h["misses"][:5]:
                print("  %s wanted %s, got %s" % (r["claim_id"], r["gt"], r["top"]))
            ok = ok and h["recall@5"] >= TARGET_RECALL_AT_5

    json.dump(result, open(a.out, "w"), indent=1)
    print("\nWrote", a.out)
    print("Caveat for the report: the query is built from the claim's denial text. If that text repeats the clause")
    print("wording, results overstate real retrieval; compare --query reason (harder) with --query both.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

