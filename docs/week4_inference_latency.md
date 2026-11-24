# Week 4: Inference latency on the CPU endpoint

Status: Steps 1 to 4 done (Nov 16 to 19, 2026), at `-c 4096` and again at `-c 16384`. The model decision (Step 6) is not made yet; criteria and freeze date are still to be agreed with the team.

## Setup

| Item | Value |
| --- | --- |
| VM | VM 5 (`cpu-inference-1-appeals`), 32 vCPUs, 122 GiB RAM, no swap, CPU only |
| Server | llama.cpp build `b12316-5d4c8eeb2`, Docker container `llama` |
| Model | `bartowski/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M` (8.03 B parameters, 4.9 GB file) |
| Endpoint | `http://10.4.36.58:30000` (private network, called from VM 3) |
| Idle baseline | 6.8 GiB RAM used, CPU 99.5% idle, load average 0.22 |

Launch command (container arguments):

```
/app/llama-server -hf bartowski/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M --host 0.0.0.0 --port 30000 -c 4096
```

**Relaunched Nov 19 with `-c 16384` and `--restart unless-stopped`** (same image, model cache bind mount and host networking; only those two things changed). Startup log: `n_slots = 4`, `kv_unified = true`, slot context capped at 8192 (the model's trained limit), shared pool 16384. `/props` therefore reports `n_ctx` 8192. Everything else is server default, as resolved at startup: 32 threads, 4 slots, `n_ctx` 4096 with a unified KV cache (`kv_unified = true`), no batch flags. The container also sets `LLAMA_ARG_PORT=8080`, but the `--port 30000` argument overrides it. Model file: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`, snapshot `4ebc4aa83d60a5d6f9e1e1e9272a4d6306d770c1`. The model's trained context is 8192 tokens.

## Method

- `make_bench_prompts.py` builds 20 prompts for each of three Path A call profiles from the denied claims in `data/synthetic_output/labeled_claims.json`. Token counts come from the server's `/tokenize` endpoint.
- Each claim record holds only a one-sentence denial text, so the denial letter and policy clauses are padded with generic filler to reach the target sizes. The four clauses that are not the claim's own are placeholders. Latency depends on token count, not wording.
- `run_benchmark.py` sends one request at a time from VM 3, with temperature 0 and prompt caching off (`cache_prompt: false`). One warm-up call is not recorded.
- Extraction requests use a JSON schema (`response_format`). Output caps: extraction 120, draft 350, critique 100 tokens.

## Results: single request (Nov 17)

65 recorded calls, 0 failures, 0 calls served from cache.

| Profile | Calls | Input tokens (median) | Output tokens (median) | Prefill (ms) | Decode (tokens/s) | p50 wall (s) | p95 wall (s) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Extraction | 20 | 834 | 73 | 5,387 | 9.8 | 12.7 | 13.2 | none hit the cap |
| Draft | 20 | 1,769 | 350 | 11,232 | 10.1 | 45.6 | 48.3 | 12 of 20 hit the 350-token cap |
| Critique | 20 | 2,138 | 100 | 13,847 | 9.6 | 24.6 | 25.2 | all 20 hit the 100-token cap |
| Reference (short prompt) | 5 | 30 | 78 | 234 | 13.5 | 5.9 | 6.2 | all 5 hit the 78-token cap |

Input tokens include about 20 tokens of chat template. Prefill ran at about 155 tokens/s in every profile, and decode at about 10 tokens/s with 800 to 2,100 tokens of context (13.5 tokens/s on a 30-token prompt). Wall time is roughly input tokens ÷ 155 plus output tokens ÷ 10.

One Path A pass (extraction, draft, critique in sequence, no revision loop) takes about 83 s at p50 (12.7 + 45.6 + 24.6).

Structured output: 20 of 20 extraction replies parsed as JSON with all required keys. Field correctness (are the values right?) and the cost of the schema constraint (a run without it) were not measured.

## Results: single request at `-c 16384` (Nov 19)

Same prompts, same script, new container. 65 recorded calls, 0 failures, 0 cache hits, 20 of 20 extraction replies valid JSON.

| Profile | Input tokens (median) | Output tokens (median) | Prefill (ms) | Decode (tokens/s) | p50 wall (s) | p95 wall (s) | Hit output cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Extraction | 834 | 73 | 5,155 | 18.4 | 9.2 | 9.8 | 0 of 20 |
| Draft | 1,769 | 350 | 10,979 | 17.3 | 31.2 | 31.8 | 12 of 20 |
| Critique | 2,138 | 100 | 13,596 | 16.3 | 20.0 | 20.3 | 20 of 20 |

One Path A pass (no revision loop) is about 60 s at p50 (9.2 + 31.2 + 20.0), versus about 83 s in the `-c 4096` run.

Prefill is unchanged (about 155 tokens/s). Decode is about 75% faster than in the first run. The cause is not confirmed: a larger context should not speed up decoding, so likely explanations are other load on the shared host during the first run, or a quieter host after the restart. The 1-user baseline in the concurrency run below (31.4 s) matches this run, so the faster speed held for at least the later part of the evening. Treat the two runs as a range until it is seen again.

## Caveats

- **Output length drives latency.** Most draft and critique calls stopped at their caps, so these times are near the upper bound for the capped length. Each extra 100 output tokens adds about 10 s. Real output lengths are not known until the real prompts run.
- **The reference run does not reproduce the Week 2 figure.** The deck reports a 78-token answer in 9.2 s warm (about 8.5 tokens/s). Here the same output length took 5.9 s. The deck's rate equals 78 ÷ 9.2, so it looks like an end-to-end figure, and the Week 2 prompt may have been a few hundred tokens. The Week 2 method has not been confirmed.
- **Prompts are synthetic**, with filler text and placeholder clauses.

## Concurrency (Step 4)

Method: the draft profile with 1, 2 and 4 simultaneous requests, 3 rounds each (3, 6 and 12 calls), 10 s idle between bursts, `run_concurrency.py`.

**At `-c 4096` the 4 slots shared one 4096-token pool** (`kv_unified = true`), and concurrent drafts failed with HTTP 500 `Context size has been exceeded`. That was a context limit, not CPU capacity, so those numbers are not reported. Relaunching with `-c 16384` fixed it.

At `-c 16384` (Nov 19), 0 errors, 0 short prompts, 0 early stops:

| Users | Calls | Errors | p50 (s) | p95 (s) | Max (s) | Draft calls/min | p50 vs 1 user |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 0 | 31.4 | 31.8 | 31.8 | 1.91 | 1.0x |
| 2 | 6 | 0 | 50.3 | 51.5 | 51.5 | 2.38 | 1.6x |
| 4 | 12 | 0 | 88.2 | 88.8 | 88.8 | 2.73 | 2.8x |

Reading it: all requests in a burst slow down about equally, so the slots do run in parallel and share the 32 CPUs. Adding users raises total throughput only a little (1.25x at 2 users, 1.43x at 4) because generation is limited by the CPU, not by free slots. The endpoint's capacity is roughly 2.7 drafts per minute. Three rounds per level is a small sample, though the rounds agree closely (4 users: 80 to 89 s).

Estimate, not measured: scaling the 60 s Path A pass by the draft slowdown gives about 97 s with 2 users and about 170 s with 4 users at once. Extraction and critique were not run under load.

## Decision: TBD (measurements in, criteria not yet agreed)

Criteria still to be agreed with the team. The proposed starting point:

| Criterion | Proposed threshold | Measured so far |
| --- | --- | --- |
| Estimated time for one Path A pass | 3 minutes or less | about 60 s single user at `-c 16384` (83 s in the first run); about 170 s estimated with 4 at once |
| Crashes, out-of-memory errors or timeouts | 0 | 0 in 130 single-request calls and 21 concurrent calls at `-c 16384`; concurrent drafts failed at `-c 4096` (fixed by the relaunch) |
| Extraction replies that validate against the schema | 10 of 10 | 20 of 20 in both runs |

Model freeze date: TBD (proposed Fri Nov 20).

## Reproduce

From `benchmarks/inference_latency/` on VM 3:

```bash
python3 make_bench_prompts.py \
  --claims ../../data/synthetic_output/labeled_claims.json \
  --base http://10.4.36.58:30000 --n 20

python3 run_benchmark.py --base http://10.4.36.58:30000 \
  --model "bartowski/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M" \
  --reference reference_prompt_short.txt --ref-max-tokens 78
```

Concurrency, same folder:

```bash
python3 run_concurrency.py --base http://10.4.36.58:30000 \
  --model "bartowski/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M" \
  --out bench_concurrency_c16384.json --yes
```

`--yes` skips the script's context prompt. Its check reads `n_ctx` 8192 from `/props` and wrongly flags the 4-user row, because the real shared pool is 16384.

Files: `make_bench_prompts.py`, `run_benchmark.py`, `bench_prompts.json`, `bench_results.json` (`-c 4096` run), `bench_results_c16384.json`, `bench_concurrency_c16384.json`, `reference_prompt_short.txt`. The `reference_78_out` key in `bench_results.json` holds the reference runs (added after the main run by a separate snippet).

The VM costs 32 service units per hour, so run the whole benchmark in one sitting and tell the team before the VM is shelved.
