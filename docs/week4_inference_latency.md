# Week 4, Task 1: Inference latency on the CPU endpoint

Status: Steps 1 to 3 done (Nov 16 to 17, 2026). Step 4 (concurrency) and the model decision (Step 6) are not done yet and are marked TBD below.

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

Everything else is server default, as resolved at startup: 32 threads, 4 slots, `n_ctx` 4096 with a unified KV cache (`kv_unified = true`), no batch flags. The container also sets `LLAMA_ARG_PORT=8080`, but the `--port 30000` argument overrides it. Model file: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`, snapshot `4ebc4aa83d60a5d6f9e1e1e9272a4d6306d770c1`. The model's trained context is 8192 tokens.

## Method

- `make_bench_prompts.py` builds 20 prompts for each of three Path A call profiles from the denied claims in `data/synthetic_output/labeled_claims.json`. Token counts come from the server's `/tokenize` endpoint.
- Each claim record holds only a one-sentence denial text, so the denial letter and policy clauses are padded with generic filler to reach the runbook's sizes. The four clauses that are not the claim's own are placeholders. Latency depends on token count, not wording.
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

## Caveats

- **Output length drives latency.** Most draft and critique calls stopped at their caps, so these times are near the upper bound for the capped length. Each extra 100 output tokens adds about 10 s. Real output lengths are not known until the real prompts run.
- **The reference run does not reproduce the Week 2 figure.** The deck reports a 78-token answer in 9.2 s warm (about 8.5 tokens/s). Here the same output length took 5.9 s. The deck's rate equals 78 ÷ 9.2, so it looks like an end-to-end figure, and the Week 2 prompt may have been a few hundred tokens. The Week 2 method has not been confirmed.
- **Prompts are synthetic**, with filler text and placeholder clauses.

## Concurrency (Step 4): TBD

Planned: the draft profile with 1, 2 and 4 simultaneous requests, 3 rounds each.

Known risk: with `kv_unified = true`, the 4096-token context is likely one pool shared by all 4 slots, not 4096 per slot. Two simultaneous drafts need about 4,300 tokens, and four need about 8,600. Requests may queue, be truncated or fail, in which case the result reflects the context limit and not CPU contention. If that happens, relaunch with a larger `-c` (for example 16384) after the team agrees, and rerun Steps 3 and 4.

| Users | p50 (s) | p95 (s) | Draft calls/min | Errors |
| --- | --- | --- | --- | --- |
| 1 | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD |

## Decision: TBD

Criteria still to be agreed with the team. The proposed starting point from the Week 4 runbook:

| Criterion | Proposed threshold | Measured so far |
| --- | --- | --- |
| Estimated time for one Path A pass | 3 minutes or less | about 83 s at p50 (single user, caps as above) |
| Crashes, out-of-memory errors or timeouts | 0 | 0 in 65 recorded calls (concurrency not yet tested) |
| Extraction replies that validate against the schema | 10 of 10 | 20 of 20 |

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

Files: `make_bench_prompts.py`, `run_benchmark.py`, `bench_prompts.json`, `bench_results.json`, `reference_prompt_short.txt`. The `reference_78_out` key in `bench_results.json` holds the reference runs (added after the main run by a separate snippet).

The VM costs 32 service units per hour, so run the whole benchmark in one sitting and tell the team before the VM is shelved.

