#!/usr/bin/env python3
"""Week 4 inference latency: single-request latency benchmark for the llama.cpp server.

Runs the reference prompt, then every prompt in bench_prompts.json one at a time
(no concurrency; Step 4 handles that). Prompt caching is turned off so repeated
prompts do not make prefill look faster than it is.

    python3 run_benchmark.py --base http://10.4.36.58:30000 \
        --model "bartowski/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M"

Writes bench_results.json and prints a per-profile summary.
"""
import argparse
import json
import statistics
import time

import requests

CAPS = {"extraction": 120, "draft": 350, "critique": 100}
SCHEMAS = {"extraction": {
    "type": "object", "required": ["denial_code", "provider", "dates", "amounts"],
    "properties": {"denial_code": {"type": "string"}, "provider": {"type": "string"},
                   "dates": {"type": "array", "items": {"type": "string"}},
                   "amounts": {"type": "array", "items": {"type": "number"}}}}}


def one_call(base, model, prompt, max_tokens, schema=None):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "cache_prompt": False,
            "messages": [{"role": "user", "content": prompt}]}
    if schema:   # if your build rejects this form, use {"type": "json_object", "schema": schema}
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    t0 = time.perf_counter()
    try:
        r = requests.post(base + "/v1/chat/completions", json=body, timeout=600)
    except requests.RequestException as e:
        return {"ok": False, "wall": time.perf_counter() - t0, "error": str(e)}
    wall = time.perf_counter() - t0
    if r.status_code != 200:
        return {"ok": False, "wall": wall, "status": r.status_code, "error": r.text[:300]}
    j = r.json()
    t = j.get("timings", {})
    text = j["choices"][0]["message"]["content"]
    res = {"ok": True, "wall": wall, "in_tokens": j["usage"]["prompt_tokens"],
           "out_tokens": j["usage"]["completion_tokens"],
           "finish_reason": j["choices"][0].get("finish_reason"),
           "cache_n": t.get("cache_n"), "prefill_ms": t.get("prompt_ms"),
           "prefill_tps": t.get("prompt_per_second"), "decode_ms": t.get("predicted_ms"),
           "decode_tps": t.get("predicted_per_second"), "text": text}
    if schema:
        try:
            parsed = json.loads(text)
            res["valid_json"] = all(k in parsed for k in schema["required"])
        except ValueError:
            res["valid_json"] = False
    return res


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def med(rs, key):
    vals = [r[key] for r in rs if r.get(key) is not None]
    return round(statistics.median(vals), 1) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. http://10.4.36.58:30000")
    ap.add_argument("--model", required=True, help="the id from /v1/models")
    ap.add_argument("--prompts", default="bench_prompts.json")
    ap.add_argument("--reference", default="reference_prompt.txt")
    ap.add_argument("--ref-runs", type=int, default=5)
    ap.add_argument("--ref-max-tokens", type=int, default=64,
                    help="set this to the output cap the Week 2 reference run used")
    ap.add_argument("--out", default="bench_results.json")
    a = ap.parse_args()
    base = a.base.rstrip("/")

    profiles = json.load(open(a.prompts))
    ref_prompt = open(a.reference).read()

    print("warm-up (not recorded)...")
    one_call(base, a.model, profiles["draft"][0], 50)

    print("reference prompt x%d (compare with the deck's 9.2 s warm)..." % a.ref_runs)
    ref = [one_call(base, a.model, ref_prompt, a.ref_max_tokens) for _ in range(a.ref_runs)]

    out = {"reference": ref}
    for name, prompts in profiles.items():
        print("%s: %d prompts..." % (name, len(prompts)))
        out[name] = [one_call(base, a.model, p, CAPS[name], SCHEMAS.get(name)) for p in prompts]
    json.dump(out, open(a.out, "w"), indent=1)

    print("\n%-11s %4s %5s %7s %7s %9s %9s %9s %10s %9s" % (
        "profile", "ok", "err", "p50 s", "p95 s", "in tok", "out tok", "prefill ms", "decode t/s", "cached"))
    for name, rs in out.items():
        good = [r for r in rs if r["ok"]]
        if not good:
            print("%-11s all %d calls failed; first error: %s" % (name, len(rs), rs[0].get("error")))
            continue
        w = [r["wall"] for r in good]
        cached = sum(1 for r in good if r.get("cache_n"))
        print("%-11s %4d %5d %7.1f %7.1f %9s %9s %9s %10s %9d" % (
            name, len(good), len(rs) - len(good), pct(w, .5), pct(w, .95), med(good, "in_tokens"),
            med(good, "out_tokens"), med(good, "prefill_ms"), med(good, "decode_tps"), cached))
    ext = [r for r in out.get("extraction", []) if r["ok"]]
    if ext:
        print("\nextraction valid JSON with required keys: %d/%d" % (sum(r["valid_json"] for r in ext), len(ext)))
    capped = {n: sum(1 for r in rs if r["ok"] and r["finish_reason"] == "length") for n, rs in out.items()}
    print("calls stopped at the output cap (finish_reason=length):", capped)
    print("'cached' should be 0 everywhere; if not, the server ignored cache_prompt=false.")
    print("Wrote", a.out)


if __name__ == "__main__":
    main()
