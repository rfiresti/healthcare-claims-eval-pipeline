# Retrieval and Citation Grounding Scorecard
 
**Date:** 2026-11-24 · **Commit:** `3a186ba` (HEAD at the held-out run) · **Owner:** Steve
 
**Result:** On the 153 held-out claims, hybrid Recall@5 is 100% (target ≥ 85%) and Citation Grounding is 100% (target ≥ 90%). Both targets are met. The test set was run once, on Nov 24, with the configuration frozen on Nov 23.
 
## Held-out results (test split, 153 claims, run once with the frozen configuration)
 
Queries are paraphrased denial notes that do not copy the clause wording (see Method).
 
| Variant | R@1 …0001 (79) | R@1 …0002 (74) | Pooled R@1 | Pooled R@5 | Pooled MRR |
| --- | --- | --- | --- | --- | --- |
| Hybrid, w_kw 0.25 (frozen) | 97.5% | 98.7% | 98.0% | 100% | 0.990 |
| Vector only | 96.2% | 98.7% | 97.4% | 100% | 0.985 |
| Keyword only | 87.3% | 85.1% | 86.3% | 100% | 0.918 |
 
All three variants reach R@5 100% on held-out, so the comparison rests on R@1 and MRR. There, hybrid is ahead of vector-only (98.0% vs 97.4% R@1, 0.990 vs 0.985 MRR), and both are well ahead of keyword-only. The gap between hybrid and vector-only is one claim out of 153, so treat them as tied. Hybrid's only R@1 misses are Exclusion claims (R@1 92.0% and 95.0%); every other category is at 100%.
 
| Grounding metric | Held-out |
| --- | --- |
| Claims with ≥ 1 citation | 153 / 153 |
| Zero-citation claims | 0 |
| Errors | 0 |
| Grounded (grounded citations / all citations) | 100% (153 / 153) |
| Gold hit | 100% |
| LLM wall time per claim, p50 | 7.5 s (19.4 min total) |
 
Run files: `eval/runs/retrieval_test_paraphrase_20261124-192022.json`, `retrieval_test_both_20261124-192041.json`, `retrieval_test_reason_20261124-192053.json`, `eval/runs/retrieval_test_console.txt`, `eval/runs/grounding_test_console.txt`, `eval/runs/grounding_test_20261124_194034.json`
 
## Dev results (dev split, 354 claims; all tuning happened here)
 
Retrieval, paraphrased queries, 28 clauses per tenant (4 real clauses + 24 near-miss distractors):
 
| Variant | R@1 …0001 (182) | R@1 …0002 (172) | Pooled R@1 | Pooled R@5 | Pooled MRR |
| --- | --- | --- | --- | --- | --- |
| Hybrid, w_kw 0.25 (frozen) | 96.7% | 96.5% | 96.6% | 100% | 0.980 |
| Vector only | 95.6% | 95.9% | 95.8% | 100% | 0.976 |
| Keyword only | 86.3% | 84.9% | 85.6% | 98.0% | 0.907 |
 
Run file: `eval/runs/retrieval_dev_paraphrase_20261124-040157.json` (commit 3a186ba). On dev, R@5 can't tell hybrid and vector-only apart: both reach 100%. R@1 and MRR separate them. Keyword-only fails mainly on MedicalNecessity (R@1 53.5% / 47.1%). Of its 7 R@5 misses, 6 are MedicalNecessity claims whose gold clause is CLS-1004; in each miss, the top 5 are all distractors.
 
w_kw 0.25 was chosen on dev by a rule set before the sweep (best pooled MRR), out of 0, 0.25, 0.5, 0.75 and 1.0. The old default, 1.0, gave pooled R@1 93.5% and MRR 0.962. Run file: `eval/runs/fusion_sweep_dev_paraphrase_20261124-035838.json`.
 
Grounding, same 100 dev claims each run:
 
| Prompt | Zero-citation claims | Grounded | Gold hit | Run file |
| --- | --- | --- | --- | --- |
| v1 original | 31 | 100% | 100% | `grounding_dev_20261124_033406` |
| v2 support rule | 25 | 100% | 100% | `grounding_dev_20261124_034316` |
| v3 v2 + worked example | 0 | 100% | 99% | `grounding_dev_20261124_035025` |
| v3, w_kw 0.25 (frozen) | 0 | 100% | 100% | `grounding_dev_20261124_040725` |
 
The single v3 gold miss (CLM-000840) shows why gold hit is reported beside grounding. Retrieval ranked distractor CLS-2023 first, and the model quoted it exactly. The citation was grounded, but it pointed to the wrong clause. Moving to w_kw 0.25 fixed it.
 
## Method
 
- **Split:** 507 denied claims split 70/30 by seed 516, stratified by tenant and denial category: dev 354, test 153 (`eval/dev_ids.txt`, `eval/test_ids.txt`). All tuning used dev only. The test set was run once, with no changes afterward.
- **Queries:** For each claim, the frozen model wrote one paraphrased denial note (prompt p3, temperature 0.7, seed 516, commit 0ddc830; 507 notes) into `eval/paraphrased_queries.json`, kept out of the claims table. The model that writes the paraphrases also drafts the appeals.
- **Why not the raw claim text:** In `both`/`text` mode, the denial text copies the clause sentence word for word, so every method scores 100%. In `reason` mode with clause IDs removed, every query is the same string, so R@5 just equals a label frequency. The held-out `both` and `reason` numbers are reported below only as an upper bound and a floor.
- **Citation Grounding:** A citation is grounded if the cited clause is in the retrieved set and its quote appears verbatim in that clause (case and whitespace normalized). Quotes under 5 words fail. Score = grounded citations / all citations. Clause IDs are not restricted to the retrieved set. Claims with zero citations are reported separately; they do not lower the score.
| Held-out, other query modes | Hybrid R@5 …0001 | Hybrid R@5 …0002 |
| --- | --- | --- |
| both (upper bound, copy leakage) | 100% | 100% |
| reason (floor, identical queries) | 31.6% | 27.0% |
 
In `both` mode, all three variants score 100% R@1, which is the copy leakage. In `reason` mode, every claim gets the same top 5 (CLS-2006, CLS-2005, CLS-1001, CLS-2002, CLS-2001). R@5 then equals the share of Exclusion claims (25/79 and 20/74), and the other three categories are at 0%. The same pattern showed up on all claims before the split (31.4% and 27.2%).
 
## Frozen configuration (frozen 2026-11-23)
 
| Setting | Value |
| --- | --- |
| Model | Llama 3 8B Q4_K_M, llama.cpp b12316, `-c 16384`, 4 slots (CPU, cpu-inference-1-appeals) |
| Grounding prompt | v3 |
| Retrieval | Hybrid RRF, w_vec 1.0 / w_kw 0.25, k 5, 20 candidates, runs as `app_user` under RLS |
| Embeddings | all-MiniLM-L6-v2 |
| Chunking | MAX_CHARS 1000 (as committed) |
| Corpus | 28 clauses per tenant: POL-01001 (CLS-1001–1004) + POL-01002 distractors (CLS-2001–2024) |
 
**Not tested:** Whether prompt v3 still declines to cite when no retrieved clause applies (the gold clause is removed from the retrieved set).
 

## Week 5 Orchestration, RLS Checkpointing & Concurrency Verification (2026-11-28)

Appended after the Nov 24 frozen held-out scorecard without modifying the held-out test baseline.

- **7-Node LangGraph Workflow & Concurrency:** `appeal_graph.py`, `extract.py`, and `draft_critique.py` execute structured extraction, deterministic grounding critique (1-revision cap, verbatim quote check, fail-closed `"ungrounded"` status), HITL pause/resume (`interrupt_review` and `Command(resume=...)`), and a 4-slot semaphore (`LLM_SEMAPHORE = threading.BoundedSemaphore(4)`) against `http://10.4.36.58:30000`.
- **4-Worker `extract` Benchmark:** 8 denied claims across 4 workers with `BoundedSemaphore(4)` completed in `36.97s` wall time (`0.22 req/s`), `p50` latency `18,483.5 ms`, `p95` latency `22,467.2 ms` (range `14,319.8 ms` to `22,467.2 ms`).
- **RLS Persistence & Tenancy Checks:** `rls_checkpointer.py` (`PostgresSaver` with `autocommit=True`, `prepare_threshold=0`, `dict_row`) and transaction-scoped `retrieve()` (`set_config('app.current_tenant_id', ..., true)`) passed 41/41 isolation tests (`eval/week5_retrieval_isolation.txt`) and 6/6 checkpoint tenancy checks (`eval/week5_checkpoint_tenancy.txt`); synthetic test threads were purged afterward.
- **Nov 28 Dev Verification Runs:**
  - Retrieval (`eval/runs/retrieval_dev_paraphrase_20261128-013816.json`, `--ids-file eval/dev_ids.txt`, `--query paraphrase`): `100.0%` hybrid `Recall@5` on both tenants (`96.7%` / `96.5%` `R@1`, `0.981` / `0.979` `MRR`, `0` misses at `k=5`).
  - Grounding (`eval/runs/grounding_dev_20261128_014017.json`, `eval/week5_eval_grounding_s20.log`, `--sample 20 --workers 2`): `100.0%` citation grounding (`20/20`), `100.0%` claims fully grounded, `100.0%` gold-hit rate, `0` zero-citation claims, `0` errors (`llm_wall_p50_s = 6.98s`, `1.2 min` total).
