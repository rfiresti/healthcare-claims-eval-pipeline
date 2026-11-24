#!/usr/bin/env python3
"""Citation Grounding eval.

Definition (agreed with the team, Nov 23):
  For each denied claim, retrieve k=5 clauses as that claim's tenant, then ask the
  model to cite the clause(s) supporting the denial using ONLY those clauses, as
  JSON {"citations": [{"clause_id": ..., "quote": ...}]}.
  A citation is grounded if
    (a) its clause_id is one of the retrieved clauses, and
    (b) its quote appears word for word in that clause's text after normalizing
        case and whitespace.
  Citation Grounding = grounded citations / all citations.
  Secondary: gold-hit rate (cited clause is the claim's ground-truth clause) and
  the number of claims with zero citations (these never count as grounded).

Added guard (flag to the team): a quote shorter than --min-quote-words words is
ungrounded, otherwise a one-word quote like "the" would pass check (b) trivially.

Usage (run from the repo root on the DB server, APP_DB_URI and ADMIN_DB_URI exported):
  # 0. offline self-test of the scoring logic, no DB or LLM needed
  python3 src/pipeline/eval_grounding.py --self-test
  # 1. dry run: show query + retrieved clauses for 2 claims, no LLM calls
  PYTHONPATH=src/pipeline python src/pipeline/eval_grounding.py \
      --tenant-id 00000000-0000-0000-0000-000000000001 --limit 2 --dry-run
  # 2. dev subsample (about 100 claims), in tmux
  PYTHONPATH=src/pipeline python src/pipeline/eval_grounding.py \
      --tenant-id 00000000-0000-0000-0000-000000000001 \
      --tenant-id 00000000-0000-0000-0000-000000000002 \
      --ids-file eval/dev_ids.txt --sample 100 --split dev --workers 2
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LLM_URL = os.environ.get("LLM_URL", "http://10.4.36.58:30000")
CLAUSE_ID_RE = re.compile(r"\bCLS-\d+\b", re.IGNORECASE)
MAX_WORKERS = 4  # team rule: no load above 4 users

SCHEMA = {
    "type": "object",
    "properties": {
        "citations": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "clause_id": {"type": "string", "maxLength": 40},
                    "quote": {"type": "string", "maxLength": 400},
                },
                "required": ["clause_id", "quote"],
            },
        }
    },
    "required": ["citations"],
}
# Note: clause_id is deliberately NOT constrained to an enum of the retrieved IDs.
# Doing so would make check (a) pass by construction and inflate the score.

SYSTEM_PROMPT = (
    "You review denied health insurance claims. You are given a claim and a set of "
    "policy clauses, each labeled with its clause ID in square brackets. Cite the "
    "clause or clauses that support the denial, using ONLY the clauses provided. "
    "For each citation give the clause_id exactly as labeled and a quote of 5 to 30 "
    "words copied character for character from that clause's text. Do not paraphrase, "
    "shorten, or join separate sentences. If no clause supports the denial, return "
    '{"citations": []}.'
)


# ---------------------------------------------------------------- pure logic

_PUNCT_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})


def normalize(text):
    """Case- and whitespace-insensitive form used for the verbatim check."""
    text = unicodedata.normalize("NFKC", text or "").translate(_PUNCT_MAP)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.strip(" \"'")


def scrub_clause_ids(text):
    """Remove clause IDs so the claim text cannot name the answer."""
    return re.sub(r"\s{2,}", " ", CLAUSE_ID_RE.sub("", text or "")).strip()


def check_citation(citation, retrieved_by_id, min_quote_words):
    """Return (grounded, reason). retrieved_by_id: clause_id -> [chunk texts]."""
    cid = str(citation.get("clause_id", "")).strip()
    quote = normalize(citation.get("quote", ""))
    texts = retrieved_by_id.get(cid)
    if texts is None:
        # tolerate "[CLS-1001]" or lowercase IDs, nothing looser
        alt = cid.strip("[] ").upper()
        texts = retrieved_by_id.get(alt)
        if texts is None:
            return False, "clause_not_retrieved"
    if len(quote.split()) < min_quote_words:
        return False, "quote_too_short"
    if any(quote in normalize(t) for t in texts):
        return True, "ok"
    return False, "quote_not_in_clause"


def parse_reply(content):
    """Parse the model reply into a list of citation dicts. Raises ValueError."""
    data = json.loads(content)
    cits = data.get("citations") if isinstance(data, dict) else None
    if not isinstance(cits, list):
        raise ValueError("reply has no citations list")
    return [c for c in cits if isinstance(c, dict)]


def score(records):
    """Summarize per-claim records into the scorecard numbers."""
    total = grounded = gold_hits = 0
    zero_cit = errors = claims_all_grounded = 0
    reasons = {}
    for r in records:
        if r.get("error"):
            errors += 1
            continue
        cits = r["citations"]
        if not cits:
            zero_cit += 1
            continue
        ok_here = True
        for c in cits:
            total += 1
            reasons[c["reason"]] = reasons.get(c["reason"], 0) + 1
            if c["grounded"]:
                grounded += 1
            else:
                ok_here = False
            if c["gold_hit"]:
                gold_hits += 1
        claims_all_grounded += ok_here
    n = len(records)
    walls = sorted(r["llm_wall_s"] for r in records if r.get("llm_wall_s") is not None)
    return {
        "claims": n,
        "claims_with_citations": n - zero_cit - errors,
        "claims_zero_citations": zero_cit,
        "claims_error": errors,
        "citations_total": total,
        "citations_grounded": grounded,
        "citation_grounding": round(grounded / total, 4) if total else None,
        "claims_fully_grounded_rate": round(claims_all_grounded / n, 4) if n else None,
        "gold_hit_rate": round(gold_hits / total, 4) if total else None,
        "citation_outcomes": reasons,
        "llm_wall_p50_s": walls[len(walls) // 2] if walls else None,
    }


def build_prompt(query, chunks):
    blocks = "\n\n".join(f"[{c['clause_id']}]\n{c['content']}" for c in chunks)
    return f"Policy clauses:\n\n{blocks}\n\nClaim:\n{query}\n\nReturn the JSON now."


# ---------------------------------------------------------------- I/O

def call_llm(url, prompt, max_tokens, timeout):
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "citations", "schema": SCHEMA},
        },
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    wall = time.perf_counter() - t0
    choice = out["choices"][0]
    return choice["message"]["content"], choice.get("finish_reason"), out.get("usage", {}), wall


def as_chunk(row):
    """Adapt one row returned by retrieve() to {'clause_id', 'content'}."""
    if isinstance(row, dict):
        get = row.get
    elif hasattr(row, "_asdict"):
        get = row._asdict().get
    else:
        get = lambda k: getattr(row, k, None)  # noqa: E731
    cid = get("clause_id")
    content = get("text") if get("text") is not None else get("content")  # retrieve() uses "text"
    if cid is None or content is None:
        raise SystemExit(
            f"retrieve() row has no clause_id/text fields: {row!r}\n"
            "Paste this line to Claude so the adapter can be fixed."
        )
    return {"clause_id": str(cid), "content": content}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ---------------------------------------------------------------- main

def run(args):
    import psycopg  # noqa: F401  (retrieve.py uses psycopg 3)
    import eval_retrieval as er
    from retrieve import retrieve

    for var in ("APP_DB_URI", "ADMIN_DB_URI"):
        if not os.environ.get(var):
            raise SystemExit(f"{var} is not set in this shell (re-export it with read -rsp).")

    # Claims and labels come from eval_retrieval.load_claims (admin role: app_user
    # cannot read claim_labels), so both evals select claims the same way.
    admin = psycopg.connect(os.environ["ADMIN_DB_URI"])
    claims = []
    for t in args.tenant_id:
        rows, counts = er.load_claims(admin, t, args.denied_status)
        print(f"tenant {t}: status counts {counts}; using {len(rows)} {args.denied_status} claims")
        for r in rows:
            claims.append({"tenant_id": t, "claim_id": str(r["claim_id"]), "row": r,
                           "gold": r["ground_truth_clause_id"],
                           "category": r.get("ground_truth_category")})
    admin.close()

    if args.ids_file:
        keep = {l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()}
        claims = [c for c in claims if c["claim_id"] in keep]
        print(f"after --ids-file: {len(claims)} claims")
    if args.sample and args.sample < len(claims):
        claims = random.Random(args.seed).sample(claims, args.sample)
        claims.sort(key=lambda c: (c["tenant_id"], c["claim_id"]))
    if args.limit:
        claims = claims[: args.limit]
    if not claims:
        raise SystemExit("no claims selected")

    # Retrieval first, serially, on one app_user connection (fast, no LLM).
    app = psycopg.connect(os.environ["APP_DB_URI"], autocommit=True)
    prepared = []
    for c in claims:
        q = er.build_query(c["row"], args.query, scrub=True)  # same query as eval_retrieval
        q = scrub_clause_ids(q)  # belt and braces; build_query already scrubs
        chunks = [as_chunk(r) for r in retrieve(app, c["tenant_id"], q, k=args.k)]
        if not chunks:
            raise SystemExit(f"tenant {c['tenant_id']} returned no chunks; load its policy chunks first.")
        prepared.append({**{k: v for k, v in c.items() if k != "row"}, "query": q, "chunks": chunks})
    app.close()

    if args.dry_run:
        for p in prepared:
            print("\n=== claim", p["claim_id"], "tenant", p["tenant_id"], "gold", p["gold"], p.get("category"))
            print("query:", p["query"])
            for ch in p["chunks"]:
                print(f"  [{ch['clause_id']}] {ch['content'][:160]!r}")
            print("prompt chars:", len(build_prompt(p["query"], p["chunks"])))
        return

    lock = threading.Lock()
    done = [0]

    def one(p):
        by_id = {}
        for ch in p["chunks"]:
            by_id.setdefault(ch["clause_id"], []).append(ch["content"])
        rec = {
            "tenant_id": p["tenant_id"], "claim_id": p["claim_id"], "gold": p["gold"],
            "category": p.get("category"),
            "query": p["query"], "retrieved": [ch["clause_id"] for ch in p["chunks"]],
            "citations": [], "error": None, "llm_wall_s": None,
        }
        try:
            content, finish, usage, wall = call_llm(
                args.llm_url, build_prompt(p["query"], p["chunks"]), args.max_tokens, args.timeout)
            rec.update(raw=content, finish_reason=finish, usage=usage, llm_wall_s=round(wall, 2))
            for c in parse_reply(content):
                ok, why = check_citation(c, by_id, args.min_quote_words)
                rec["citations"].append({
                    "clause_id": c.get("clause_id"), "quote": c.get("quote"),
                    "grounded": ok, "reason": why,
                    "gold_hit": str(c.get("clause_id", "")).strip("[] ").upper() == str(p["gold"]).upper(),
                })
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        with lock:
            done[0] += 1
            flag = "ERR" if rec["error"] else ("ok" if rec["citations"] and all(
                c["grounded"] for c in rec["citations"]) else "FLAG")
            print(f"[{done[0]}/{len(prepared)}] {p['claim_id']} {flag} {rec['llm_wall_s']}s", flush=True)
        return rec

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(one, prepared))
    elapsed = time.perf_counter() - t0

    summary = score(records)
    run_doc = {
        "kind": "citation_grounding",
        "split": args.split,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "params": {k: v for k, v in vars(args).items() if k not in ("self_test", "dry_run")},
        "schema": SCHEMA,
        "system_prompt": SYSTEM_PROMPT,
        "elapsed_s": round(elapsed, 1),
        "summary": summary,
        "flagged": [r for r in records if r["error"] or not r["citations"]
                    or not all(c["grounded"] for c in r["citations"])],
        "records": records,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"grounding_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    run_doc["paraphrases"] = er.paraphrase_meta()
    out.write_text(json.dumps(run_doc, indent=2, default=str))

    print("\n" + json.dumps(summary, indent=2))
    print(f"elapsed {elapsed / 60:.1f} min, run file: {out}")


def self_test():
    retrieved = {
        "CLS-1001": ["Services not listed in Schedule A are excluded from coverage under this plan."],
        "CLS-1002": ["Prior authorization is required for all inpatient admissions\nand advanced imaging."],
    }
    cases = [
        ({"clause_id": "CLS-1001", "quote": "services not listed in Schedule A are excluded"}, True, "ok"),
        ({"clause_id": "CLS-1002", "quote": "Prior   authorization is required for all inpatient admissions and"}, True, "ok"),
        ({"clause_id": "[CLS-1002]", "quote": "required for all inpatient admissions and advanced imaging."}, True, "ok"),
        ({"clause_id": "CLS-1003", "quote": "services not listed in Schedule A are excluded"}, False, "clause_not_retrieved"),
        ({"clause_id": "CLS-1001", "quote": "excluded"}, False, "quote_too_short"),
        ({"clause_id": "CLS-1001", "quote": ""}, False, "quote_too_short"),
        ({"clause_id": "CLS-1001", "quote": "services that are not in Schedule A are excluded"}, False, "quote_not_in_clause"),
        ({"clause_id": "CLS-1001", "quote": "Prior authorization is required for all inpatient admissions"}, False, "quote_not_in_clause"),
        ({"clause_id": "CLS-1001", "quote": "“Services not listed in Schedule A are excluded”"}, True, "ok"),
    ]
    for cit, want_ok, want_reason in cases:
        got = check_citation(cit, retrieved, 5)
        assert got == (want_ok, want_reason), (cit, got)

    assert parse_reply('{"citations": []}') == []
    for bad in ('{"citations": [', '{"cites": []}', '[]'):
        try:
            parse_reply(bad)
            raise AssertionError(bad)
        except ValueError:
            pass

    assert scrub_clause_ids("Non-compliance with policy clause CLS-1004.") == "Non-compliance with policy clause ."

    recs = [
        {"citations": [{"grounded": True, "reason": "ok", "gold_hit": True}], "llm_wall_s": 10.0},
        {"citations": [{"grounded": True, "reason": "ok", "gold_hit": False},
                       {"grounded": False, "reason": "quote_not_in_clause", "gold_hit": False}], "llm_wall_s": 12.0},
        {"citations": [], "llm_wall_s": 8.0},
        {"citations": [], "error": "URLError: timeout", "llm_wall_s": None},
    ]
    s = score(recs)
    assert s["citations_total"] == 3 and s["citations_grounded"] == 2
    assert s["citation_grounding"] == round(2 / 3, 4)
    assert s["claims_zero_citations"] == 1 and s["claims_error"] == 1
    assert s["gold_hit_rate"] == round(1 / 3, 4)
    assert s["claims_fully_grounded_rate"] == 0.25
    assert s["citation_outcomes"] == {"ok": 2, "quote_not_in_clause": 1}

    p = build_prompt("Denied: no prior auth.", [{"clause_id": "CLS-1002", "content": "Prior authorization..."}])
    assert "[CLS-1002]" in p and "Denied: no prior auth." in p
    print("self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="test scoring logic offline and exit")
    ap.add_argument("--tenant-id", action="append", default=[], help="repeat for several tenants")
    ap.add_argument("--denied-status", default="DENIED")
    ap.add_argument("--query", default="both", choices=["reason", "text", "both", "paraphrase"],
                    help="query mode for eval_retrieval.build_query")
    ap.add_argument("--ids-file", help="restrict to claim IDs in this file (e.g. eval/dev_ids.txt)")
    ap.add_argument("--sample", type=int, help="random sample of N claims (fixed --seed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="first N claims after sampling (smoke tests)")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "smoke"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-quote-words", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    ap.add_argument("--out-dir", default="eval/runs")
    ap.add_argument("--dry-run", action="store_true", help="retrieve and print prompts; no LLM calls")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.tenant_id:
        ap.error("--tenant-id is required")
    if not 1 <= args.workers <= MAX_WORKERS:
        ap.error(f"--workers must be 1..{MAX_WORKERS} (team rule: no load above 4 users)")
    run(args)


if __name__ == "__main__":
    sys.exit(main())

