#!/usr/bin/env python3
"""Week 4, Task 1, Step 4: concurrency probe for the llama.cpp server.

Fires the draft profile with 1, 2 and 4 simultaneous requests (3 rounds each) and
records errors, slow calls and throughput. Run it from benchmarks/inference_latency/
on VM 3, next to run_benchmark.py and bench_prompts.json:

    python3 run_concurrency.py --base http://10.4.36.58:30000 \
        --model "bartowski/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M"

Before sending any load it checks the server's context setting and tells you whether
N simultaneous drafts can fit in the KV cache. Results are saved after every burst, so
an interrupted run keeps what it has (bench_concurrency.json).

Do not run anything else against VM 5 while this runs, and do not go above 4 users;
the plan uses the stub for larger loads.
"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from run_benchmark import one_call, pct

DRAFT_CAP = 350          # same output cap as the Step 3 draft profile
TEMPLATE_TOKENS = 20     # chat-template tokens the server adds on top of /tokenize counts


def get_json(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def token_count(base, text):
    r = requests.post(base + "/tokenize", json={"content": text}, timeout=60)
    r.raise_for_status()
    return len(r.json()["tokens"])


def context_check(base, prompts, users, assume_yes):
    """Print the server's context setting and whether N drafts fit at once. Returns info dict."""
    props = get_json(base + "/props") or {}
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
    slots = props.get("total_slots")
    per_slot = None
    slot_list = get_json(base + "/slots")
    if isinstance(slot_list, list) and slot_list:
        per_slot = [s.get("n_ctx") for s in slot_list]

    try:
        longest = max(token_count(base, p) for p in prompts)
        counted = "measured"
    except Exception as e:
        longest, counted = 1770, "assumed (/tokenize failed: %s)" % e
    per_req = longest + TEMPLATE_TOKENS + DRAFT_CAP

    print("=== Context check ===")
    print("total slots:                %s" % slots)
    print("n_ctx (from /props):        %s" % n_ctx)
    if per_slot:
        print("n_ctx per slot (/slots):    %s" % per_slot)
    print("longest draft prompt:       %d tokens (%s)" % (longest, counted))
    print("one draft needs up to:      ~%d tokens (prompt + template + %d output cap)" % (per_req, DRAFT_CAP))
    print("The Step 1 startup log said kv_unified = true, which means the n_ctx above is probably one pool")
    print("shared by all slots, even though each slot reports the full size. The burst results settle it.\n")
    print("%-6s %-22s %s" % ("users", "tokens needed at once", "fits in n_ctx?"))
    misfit = False
    for n in users:
        need = n * per_req
        fits = (n_ctx is not None) and need <= n_ctx
        misfit = misfit or not fits
        print("%-6d %-22d %s" % (n, need, "yes" if fits else "NO" if n_ctx else "unknown"))
    print()

    if misfit:
        print("Some bursts need more tokens than n_ctx. Depending on the build, the extra requests may")
        print("wait for a slot to free up, be truncated, or fail. That is a context limit, not CPU")
        print("contention, so do not report those numbers as the endpoint's capacity. If you see it,")
        print("ask the team about relaunching with a larger -c (for example 16384) and re-run Steps 3 and 4.\n")
        if not assume_yes:
            if not sys.stdin.isatty():
                print("Not an interactive terminal; re-run with --yes to go ahead.")
                sys.exit(1)
            if input("Run the bursts anyway? [y/N] ").strip().lower() != "y":
                print("Stopped before sending load.")
                sys.exit(0)
    return {"n_ctx": n_ctx, "total_slots": slots, "per_slot_n_ctx": per_slot,
            "longest_draft_prompt_tokens": longest, "per_request_tokens": per_req}


def run_burst(base, model, prompts, n, rounds, cursor):
    reqs, t0 = [], time.perf_counter()
    for r in range(rounds):
        with ThreadPoolExecutor(n) as ex:
            futs = [ex.submit(one_call, base, model, prompts[(cursor + r * n + i) % len(prompts)], DRAFT_CAP)
                    for i in range(n)]
            for i, f in enumerate(futs):
                res = f.result()
                res["round"], res["idx"] = r, i
                reqs.append(res)
    elapsed = time.perf_counter() - t0
    return reqs, elapsed


def summarize(n, reqs, elapsed, baseline_p50, expected_in, slow_factor):
    ok = [r for r in reqs if r["ok"]]
    walls = [r["wall"] for r in ok]
    s = {"users": n, "calls": len(reqs), "ok": len(ok), "errors": len(reqs) - len(ok),
         "elapsed_s": round(elapsed, 1),
         "draft_calls_per_min": round(len(ok) / (elapsed / 60), 2) if elapsed else None,
         "p50": round(pct(walls, .5), 1) if walls else None,
         "p95": round(pct(walls, .95), 1) if walls else None,
         "max": round(max(walls), 1) if walls else None}
    s["slow"] = (sum(1 for w in walls if w > slow_factor * baseline_p50) if baseline_p50 else None)
    s["short_prompt"] = (sum(1 for r in ok if expected_in and r["in_tokens"] < 0.9 * expected_in)
                         if expected_in else None)
    s["early_stop_at_length"] = sum(1 for r in ok if r["finish_reason"] == "length" and r["out_tokens"] < DRAFT_CAP)
    s["walls_by_round"] = [[round(r["wall"], 1) for r in reqs if r["round"] == k and r["ok"]]
                           for k in range(max((r["round"] for r in reqs), default=-1) + 1)]
    errs = [r for r in reqs if not r["ok"]]
    s["first_error"] = (errs[0].get("status"), (errs[0].get("error") or "")[:200]) if errs else None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. http://10.4.36.58:30000")
    ap.add_argument("--model", required=True, help="the id from /v1/models")
    ap.add_argument("--prompts", default="bench_prompts.json")
    ap.add_argument("--users", type=int, nargs="+", default=[1, 2, 4], help="concurrency levels, max 4")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--slow-factor", type=float, default=2.0,
                    help="a call is 'slow' if its wall time exceeds this multiple of the 1-user p50 from this run")
    ap.add_argument("--cooldown", type=int, default=10, help="seconds of idle between bursts")
    ap.add_argument("--yes", action="store_true", help="skip the prompt when a burst does not fit in n_ctx")
    ap.add_argument("--out", default="bench_concurrency.json")
    a = ap.parse_args()
    base = a.base.rstrip("/")

    if max(a.users) > 4:
        sys.exit("Refusing to go above 4 users; the plan uses the stub for larger loads.")
    users = sorted(set(a.users))
    prompts = json.load(open(a.prompts))["draft"]

    ctx = context_check(base, prompts, users, a.yes)

    print("warm-up (not recorded)...")
    one_call(base, a.model, prompts[0], 50)

    out = {"context": ctx, "draft_cap": DRAFT_CAP, "rounds": a.rounds, "bursts": []}
    baseline_p50, expected_in, cursor = None, None, 0
    for n in users:
        print("burst: %d simultaneous x %d rounds..." % (n, a.rounds))
        reqs, elapsed = run_burst(base, a.model, prompts, n, a.rounds, cursor)
        cursor += n * a.rounds
        if n == 1:
            good = [r for r in reqs if r["ok"]]
            if len(good) != len(reqs):
                out["bursts"].append({"users": 1, "requests": reqs})
                json.dump(out, open(a.out, "w"), indent=1)
                sys.exit("The 1-user baseline had failed calls: %s. Fix that before going higher." %
                         (reqs[0].get("error") or [r for r in reqs if not r["ok"]][0].get("error")))
            baseline_p50 = pct([r["wall"] for r in good], .5)
            expected_in = statistics.median(r["in_tokens"] for r in good)
        summary = summarize(n, reqs, elapsed, baseline_p50, expected_in, a.slow_factor)
        summary["requests"] = reqs
        out["bursts"].append(summary)
        json.dump(out, open(a.out, "w"), indent=1)       # saved after every burst
        if n != users[-1] and a.cooldown:
            print("idle %ds before the next burst..." % a.cooldown)
            time.sleep(a.cooldown)

    print("\n%-6s %5s %3s %4s %7s %7s %7s %10s %5s %7s %10s" % (
        "users", "calls", "ok", "err", "p50 s", "p95 s", "max s", "calls/min", "slow", "short", "early stop"))
    for b in out["bursts"]:
        print("%-6d %5d %3d %4d %7s %7s %7s %10s %5s %7s %10s" % (
            b["users"], b["calls"], b["ok"], b["errors"], b["p50"], b["p95"], b["max"],
            b["draft_calls_per_min"], b["slow"], b["short_prompt"], b["early_stop_at_length"]))
    print("\nwall times by round (s):")
    for b in out["bursts"]:
        print("  %d users: %s" % (b["users"], "  |  ".join(" ".join("%.0f" % w for w in rnd) for rnd in b["walls_by_round"])))
    print("\ncolumns: slow = calls over %.1fx the 1-user p50 (%s s); short = prompt shorter than 90%% of the" %
          (a.slow_factor, round(baseline_p50, 1) if baseline_p50 else "n/a"))
    print("1-user prompt size (possible truncation); early stop = stopped at 'length' before the %d-token cap." % DRAFT_CAP)

    print("\nHow to read it (hints, not verdicts):")
    for b in out["bursts"]:
        if b["users"] == 1:
            continue
        if b["errors"]:
            print("  %d users: %d failed calls, first: %s" % (b["users"], b["errors"], b["first_error"]))
            print("    -> likely a context or slot limit, not CPU contention. Do not report this as capacity.")
            continue
        rounds_uneven = [rnd for rnd in b["walls_by_round"] if len(rnd) > 1 and max(rnd) > 1.6 * min(rnd)]
        if rounds_uneven:
            print("  %d users: wall times within a round differ a lot (some calls waited for others)." % b["users"])
            print("    -> requests are queueing: one slot at a time, or the KV cache is too small for all of them.")
        elif baseline_p50 and b["p50"] > 1.3 * baseline_p50:
            print("  %d users: all calls slowed about evenly (p50 %.1fx the 1-user p50)." % (b["users"], b["p50"] / baseline_p50))
            print("    -> the slots are running together and sharing the 32 CPUs.")
        else:
            print("  %d users: close to the 1-user speed. Check that the server really ran them in parallel." % b["users"])
        if b["short_prompt"] or b["early_stop_at_length"]:
            print("    -> %d short prompts, %d early stops: possible truncation from the context limit." %
                  (b["short_prompt"], b["early_stop_at_length"]))
    print("\nWrote", a.out)
    print("For the report: draft_calls_per_min at 1, 2 and 4 users sets the stub's low capacity,")
    print("and say which --parallel / -c setting this was run with.")


if __name__ == "__main__":
    main()

