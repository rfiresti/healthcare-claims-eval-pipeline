#!/usr/bin/env python3
"""Generate one paraphrased denial note per denied claim for --query paraphrase.

Each note restates the claim's denial reason in new words, keeping the
specifics (service, numbers, time limits, conditions). A note is rejected and
retried if it:
  * shares a run of MIN_RUN (5) or more consecutive words with the claim's
    denial text, which contains the gold clause sentence verbatim. Five words
    is also the minimum verbatim quote that counts in Citation Grounding, so a
    query can never hand the model a ready-made quote;
  * names a clause, policy or claim ID (CLS-, POL-, CLM-);
  * drops a number from the original (as digits or a word, e.g. 5 / five),
    since distractor clauses differ from real ones mostly by a number;
  * is shorter than 6 or longer than 60 words.

Output: eval/paraphrased_queries.json
    {"meta": {...}, "queries": {claim_id: text}, "stats": {...}, "failed": {...}}
The file is rewritten after every claim, so a run stopped mid-way resumes
where it left off. Kept out of the claims table on purpose.

Run on the DB server in tmux, repo root, venv active, ADMIN_DB_URI exported:

    python3 src/pipeline/gen_paraphrases.py --limit 5       # smoke test
    python3 src/pipeline/gen_paraphrases.py --workers 2     # full run
    python3 src/pipeline/gen_paraphrases.py --check 20      # hand-check sample
"""
import argparse
import datetime as dt
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from eval_retrieval import build_query, load_claims

DEFAULT_TENANTS = [
    "00000000-0000-0000-0000-000000000001",
    "00000000-0000-0000-0000-000000000002",
]
DEFAULT_ENDPOINT = "http://10.4.36.58:30000"
PROMPT_VERSION = "p2"
MIN_RUN = 5
MIN_WORDS, MAX_WORDS = 6, 60
ID_RE = re.compile(r"\b(?:CLS|POL|CLM)[-\s_]?\d", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
ID_STRIP_RE = re.compile(r"\b(?:CLS|POL|CLM)[-\s_]?\d+", re.IGNORECASE)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
NUM_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
             "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()

SYSTEM = "You write short, plain notes in the voice of a health insurance claims reviewer."
USER = (
    "A claim was denied for this reason:\n\n\"{denial}\"\n\n"
    "Rewrite it as a 1-2 sentence denial note a claims reviewer might send to the member. "
    "Keep every specific fact (the service, any numbers, time limits and conditions), "
    "but use your own wording: do not copy any run of {min_run} or more words from the "
    "reason above. Do not mention clause, policy or claim numbers. "
    "Reply with the note only."
)
RETRY = ("That note copied \"{run}\" from the original. Write it again with different "
         "wording, keeping the same facts. Reply with the note only.")


# ---------- text checks ----------

def words(text):
    return WORD_RE.findall(text.lower())


def longest_shared_run(a, b):
    """Longest run of consecutive words shared by a and b -> (length, words)."""
    wa, wb = words(a), words(b)
    best, end = 0, 0
    prev = [0] * (len(wb) + 1)
    for i in range(1, len(wa) + 1):
        cur = [0] * (len(wb) + 1)
        for j in range(1, len(wb) + 1):
            if wa[i - 1] == wb[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, end = cur[j], i
        prev = cur
    return best, " ".join(wa[end - best:end])


def clean(reply):
    text = reply.strip().strip('"').strip()
    text = re.sub(r"^(denial note|note)\s*:\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def problem(text, source):
    """Return None if the paraphrase is acceptable, else (reason, shared_run)."""
    n = len(words(text))
    if n < MIN_WORDS or n > MAX_WORDS:
        return f"length {n} words", None
    if ID_RE.search(text):
        return "names an ID", None
    have = set(NUM_RE.findall(text)) | set(words(text))
    for num in NUM_RE.findall(ID_STRIP_RE.sub("", source)):
        word = NUM_WORDS[int(num)] if num.isdigit() and int(num) <= 20 else None
        if num not in have and word not in have:
            return f"drops the number {num}", None
    run_len, run = longest_shared_run(text, source)
    if run_len >= MIN_RUN:
        return f"copies {run_len} words", run
    return None


# ---------- model ----------

def chat(endpoint, messages, seed, temperature, max_tokens=120, timeout=180):
    body = json.dumps({
        "model": "local", "messages": messages, "temperature": temperature,
        "seed": seed, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            if attempt == 2:
                raise RuntimeError(f"endpoint failed: {e}") from e
            time.sleep(5 * (attempt + 1))


def model_name(endpoint):
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/v1/models", timeout=10) as r:
            return json.load(r)["data"][0]["id"]
    except Exception:
        return "unknown"


def paraphrase(endpoint, claim_id, source, base_seed, temperature, max_attempts):
    """Return (text or None, attempts, longest_shared_run, last_reason)."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(denial=source, min_run=MIN_RUN)}]
    seed0 = base_seed + zlib.crc32(claim_id.encode()) % 100000
    reason = None
    for attempt in range(1, max_attempts + 1):
        reply = chat(endpoint, messages, seed0 + attempt, temperature)
        text = clean(reply)
        bad = problem(text, source)
        if bad is None:
            return text, attempt, longest_shared_run(text, source)[0], None
        reason, run = bad
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": RETRY.format(run=run) if run else
                      f"That note was rejected ({reason}). Write it again: 1-2 sentences, keep "
                      "every number from the original, no ID numbers. Reply with the note only."}]
    return None, max_attempts, None, reason


# ---------- data ----------

def field(row, key):
    return row[key] if isinstance(row, dict) else getattr(row, key)


def load_sources(tenants, denied_status):
    """Return {claim_id: (tenant_id, denial_text)} using the evals' own query builder."""
    uri = os.environ.get("ADMIN_DB_URI")
    if not uri:
        sys.exit("ADMIN_DB_URI is not set (see the session setup block).")
    import psycopg
    out = {}
    with psycopg.connect(uri) as conn:
        for tenant in tenants:
            claims, status_counts = load_claims(conn, tenant, denied_status)
            if not claims:
                sys.exit(f"No '{denied_status}' claims for tenant {tenant}. "
                         f"Statuses seen: {status_counts}. Check --denied-status.")
            for c in claims:
                cid = field(c, "claim_id")
                if cid in out:
                    sys.exit(f"Claim ID {cid} appears in more than one tenant; "
                             "a claim-ID-keyed file would be ambiguous. Stopping.")
                text = build_query(c, "text", scrub=True).strip()
                if not text:
                    sys.exit(f"{cid}: build_query(mode='text') is empty; check TEXT_KEYS.")
                out[cid] = (tenant, text)
    return out


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def save(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


# ---------- commands ----------

def check(path, n, seed):
    with open(path) as f:
        doc = json.load(f)
    ids = sorted(doc["queries"])
    random.Random(seed).shuffle(ids)
    sources = load_sources(doc["meta"]["tenants"], doc["meta"]["denied_status"])
    for cid in ids[:n]:
        print(f"--- {cid}  (shared run {doc['stats'][cid]['longest_shared_run']} words)")
        print(f"ORIGINAL:   {sources[cid][1]}")
        print(f"PARAPHRASE: {doc['queries'][cid]}\n")
    print("For each: same denial reason? specifics kept? reads like a payer note?")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tenant-id", action="append", dest="tenants")
    ap.add_argument("--denied-status", default="DENIED")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--out", default="eval/paraphrased_queries.json")
    ap.add_argument("--workers", type=int, default=2, help="max 4 (llama.cpp has 4 slots)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=516)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N pending claims (smoke test)")
    ap.add_argument("--check", type=int, metavar="N", help="print N random pairs to hand-check")
    args = ap.parse_args()

    if args.check:
        return check(args.out, args.check, args.seed)
    if not 1 <= args.workers <= 4:
        sys.exit("--workers must be between 1 and 4.")

    tenants = args.tenants or DEFAULT_TENANTS
    sources = load_sources(tenants, args.denied_status)

    meta = {"prompt_version": PROMPT_VERSION, "min_run": MIN_RUN, "endpoint": args.endpoint,
            "model": model_name(args.endpoint), "temperature": args.temperature,
            "seed": args.seed, "tenants": tenants, "denied_status": args.denied_status}
    if os.path.exists(args.out):
        with open(args.out) as f:
            doc = json.load(f)
        old = {k: doc["meta"].get(k) for k in ("prompt_version", "min_run", "model", "temperature", "seed")}
        new = {k: meta[k] for k in old}
        if old != new:
            sys.exit(f"{args.out} was made with different settings:\n  {old}\nvs\n  {new}\n"
                     "Move it aside to start a fresh run.")
        doc["failed"] = {}  # retry earlier failures
        print(f"Resuming: {len(doc['queries'])} already done.")
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        doc = {"meta": {**meta, "started": dt.datetime.now().isoformat(timespec="seconds"),
                        "git_commit": git_commit()},
               "queries": {}, "stats": {}, "failed": {}}

    pending = [cid for cid in sorted(sources) if cid not in doc["queries"]]
    if args.limit:
        pending = pending[:args.limit]
    print(f"{len(sources)} denied claims, {len(pending)} to generate, "
          f"{args.workers} worker(s), model {meta['model']}")

    lock = threading.Lock()
    t0, done = time.time(), 0

    def work(cid):
        tenant, src = sources[cid]
        return cid, tenant, paraphrase(args.endpoint, cid, src, args.seed,
                                       args.temperature, args.max_attempts)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, cid) for cid in pending]
        for fut in as_completed(futures):
            cid, tenant, (text, attempts, run, reason) = fut.result()
            with lock:
                done += 1
                if text:
                    doc["queries"][cid] = text
                    doc["stats"][cid] = {"tenant_id": tenant, "attempts": attempts,
                                         "longest_shared_run": run}
                else:
                    doc["failed"][cid] = {"tenant_id": tenant, "reason": reason}
                doc["meta"]["finished"] = dt.datetime.now().isoformat(timespec="seconds")
                save(args.out, doc)
                if done % 25 == 0 or done == len(pending):
                    rate = (time.time() - t0) / done
                    print(f"{done}/{len(pending)}  {rate:.1f} s/claim  "
                          f"eta {rate * (len(pending) - done) / 60:.0f} min  "
                          f"failed {len(doc['failed'])}", flush=True)

    retries = sum(1 for s in doc["stats"].values() if s["attempts"] > 1)
    print(f"\nDone: {len(doc['queries'])}/{len(sources)} paraphrased, "
          f"{retries} needed a retry, {len(doc['failed'])} failed.")
    if doc["failed"]:
        print("Failed claims are left out of --query paraphrase runs; rerun to retry them.")
    print(f"Next: python3 {sys.argv[0]} --check 20")


if __name__ == "__main__":
    main()

