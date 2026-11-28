#!/usr/bin/env python3
"""Build bench_prompts.json for Week 4, Task 1, Step 2.

Run on VM 3 (it needs to reach the llama.cpp server for exact token counts):

    python3 make_bench_prompts.py --claims claims.json --base http://10.4.36.58:30000
    python3 make_bench_prompts.py --claims claims.json --clauses clauses.json --n 20

Output:
    bench_prompts.json    {"extraction": [...], "draft": [...], "critique": [...]}
    reference_prompt.txt  ~78-token reference prompt (kept out of the JSON on purpose,
                          because the Step 3 script runs every key in the JSON and
                          has no output cap for a "reference" key)

The claim records only carry a one-sentence denial text, so the letter, clauses and
draft text are padded with generic filler to reach the runbook's size targets
(extraction ~800, draft ~1,800, critique ~2,200 input tokens). Latency depends on
token counts, not on what the words say. Real token counts are measured with the
server's /tokenize endpoint and printed at the end; adjust TARGETS if they are off.
"""
import argparse
import json
import random
import statistics

import requests

TARGETS = {"extraction": 800, "draft": 1800, "critique": 2200}
LETTER_TOKENS = 800        # size of the denial letter inside every profile
DRAFT_TOKENS = 380         # size of the draft appeal pasted into the critique prompt
N_CLAUSES = 5

PAYERS = ["Northline Health Plan", "Meridian Mutual Insurance", "Lakeshore Benefits Group"]
PROVIDERS = ["Riverside Physical Therapy", "Eastgate Orthopedic Clinic", "Harborview Medical Associates",
             "Summit Rehabilitation Center", "Westfield Family Practice"]

FILLER = [
    "This notice explains how the plan reached its decision on the claim identified above.",
    "The determination was made by comparing the services billed with the benefits described in the member's policy.",
    "Coverage for any service depends on the plan in effect on the date the service was provided.",
    "Some services are covered only up to a set limit, and charges above that limit are the member's responsibility.",
    "If the provider submitted additional records after this notice was prepared, they were not part of this review.",
    "A member may ask for a copy of every document, record and guideline that was relied on in making this decision, free of charge.",
    "The plan reviews each appeal using staff who were not involved in the original decision.",
    "Appeals must be submitted in writing within one hundred eighty days of the date on this notice.",
    "An appeal should include the claim number, the member's name, and a short explanation of why the decision should change.",
    "Supporting documents such as medical records, provider letters and receipts may be attached to the appeal.",
    "The plan will send a written decision within thirty days of receiving a complete appeal.",
    "Members who need the decision sooner for medical reasons may request an expedited review by telephone.",
    "If the denial is upheld, the member may have the right to an independent external review at no cost.",
    "Contact member services at the number on the back of the member card for help with any part of this process.",
    "Nothing in this notice changes the terms of the policy, and the policy language controls if there is any difference.",
    "Benefit limits are counted from the first day of the calendar year and reset on January first.",
    "Services that exceed a stated visit or dollar limit are denied even when they are medically appropriate.",
    "Clinical notes supporting medical necessity do not change a limit that applies by the terms of the policy.",
]

DRAFT_TEMPLATE = (
    "Dear Appeals Committee,\n\nI am writing to appeal the denial of claim {claim_id}. The notice states that the claim was "
    "denied under clause {clause_id}. Under clause {clause_id}, {clause_text} I ask that the plan review the visit count "
    "recorded for the member and confirm that it was calculated correctly for the calendar year in question. "
)


class Tok:
    """Exact token counts from the server's /tokenize endpoint, with a chars-per-token fallback."""

    def __init__(self, base):
        self.base = base.rstrip("/")
        self.ok = True
        sample = " ".join(FILLER)
        n = self.count(sample)
        self.cpt = len(sample) / n if n else 4.0

    def count(self, text):
        if self.ok:
            try:
                r = requests.post(self.base + "/tokenize", json={"content": text}, timeout=60)
                r.raise_for_status()
                return len(r.json()["tokens"])
            except Exception as e:
                print("WARNING: /tokenize failed (%s); using chars/4 estimate" % e)
                self.ok = False
        return max(1, round(len(text) / 4))

    def est(self, text):
        return len(text) / self.cpt * 1.10


def filler(tok, n_tokens, rng):
    """Roughly n_tokens of generic policy text, cut on a sentence boundary."""
    out, i = [], rng.randrange(len(FILLER))
    while tok.est(" ".join(out)) < n_tokens:
        out.append(FILLER[i % len(FILLER)])
        i += 1
    return " ".join(out)


def make_letter(tok, c, rng, n_tokens):
    d = c["denial_details"]
    start = "2026-%02d-%02d" % (rng.randint(1, 9), rng.randint(1, 20))
    end = "2026-%02d-%02d" % (int(start[5:7]), int(start[8:10]) + rng.randint(0, 6))
    top = (
        "%s\nClaims Determination Notice\n\nDate of notice: 2026-%02d-%02d\nMember ID: %s\nClaim number: %s\n"
        "Provider: %s\nDates of service: %s to %s\nBilled amount: $%.2f\nAmount allowed: $0.00\nAmount denied: $%.2f\n"
        "Claim status: %s\n\nReason for denial: %s\nPolicy detail: %s\nDenial code: %s (%s)\n\nAbout this decision and your appeal rights:\n"
        % (rng.choice(PAYERS), rng.randint(10, 12), rng.randint(1, 28), c["patient_id"], c["claim_id"],
           rng.choice(PROVIDERS), start, end, c["billed_amount"], c["billed_amount"], c["status"],
           d["reason"], d["text"], c["ground_truth_clause_id"], c["ground_truth_category"])
    )
    return top + filler(tok, max(0, n_tokens - tok.est(top)), rng)


def load_clauses(path):
    """Accept a policy file ({"clauses": [...]}), a list of policy files, or a bare list of clauses.
    Each clause needs clause_id plus clause_text (or text). Returns [{"clause_id", "text"}]."""
    data = json.load(open(path))
    if isinstance(data, dict):
        data = [data]
    flat = []
    for item in data:
        flat.extend(item["clauses"] if "clauses" in item else [item])
    out = {}
    for x in flat:
        out[x["clause_id"]] = {"clause_id": x["clause_id"], "text": x.get("clause_text") or x["text"]}
    return list(out.values())


def pick_clauses(c, clauses, rng):
    """Ground-truth clause first, then 4 others. Returns [(id, text)]."""
    gt = c["ground_truth_clause_id"]
    gt_text = c["denial_details"]["text"]
    if clauses:
        by_id = {x["clause_id"]: x["text"] for x in clauses}
        gt_text = by_id.get(gt, gt_text)
        others = [x for x in clauses if x["clause_id"] != gt]
    else:
        others = [{"clause_id": "CLS-%d" % (2000 + k),
                   "text": "Placeholder clause %d: benefits under this section are subject to the limits stated in the policy schedule." % k}
                  for k in range(30)]
    chosen = rng.sample(others, N_CLAUSES - 1)
    return [(gt, gt_text)] + [(x["clause_id"], x["text"]) for x in chosen]


def clause_block(tok, pairs, per_clause_tokens, rng):
    blocks = []
    for cid, text in pairs:
        head = "[%s] %s " % (cid, text)
        blocks.append(head + filler(tok, max(0, per_clause_tokens - tok.est(head)), rng))
    return "\n\n".join(blocks)


def build(tok, c, clauses, rng, profile):
    letter = make_letter(tok, c, rng, LETTER_TOKENS)
    if profile == "extraction":
        instr = ("Extract the denial code, the provider name, every date, and every dollar amount from the "
                 "denial letter below. Reply with JSON only.\n\nDENIAL LETTER:\n")
        return instr + letter
    pairs = pick_clauses(c, clauses, rng)
    ctx_head = "DENIAL LETTER:\n" + letter + "\n\nPOLICY CLAUSES:\n"
    if profile == "draft":
        instr = ("You are drafting a formal appeal for a patient whose claim was denied. Use only the policy clauses "
                 "provided, and cite a clause ID for every statement you make.\n\n")
        tail = "\n\nWrite the appeal letter."
        rest = tok.est(instr + ctx_head + tail)
        per = (TARGETS["draft"] - rest) / N_CLAUSES
        return instr + ctx_head + clause_block(tok, pairs, per, rng) + tail
    # critique
    instr = ("Review the draft appeal below. For each sentence, say whether it cites one of the policy clauses provided. "
             "List every sentence that has no valid citation. Be brief.\n\n")
    draft_head = DRAFT_TEMPLATE.format(claim_id=c["claim_id"], clause_id=pairs[0][0], clause_text=pairs[0][1])
    draft = draft_head + filler(tok, max(0, DRAFT_TOKENS - tok.est(draft_head)), rng)
    tail = "\n\nDRAFT APPEAL:\n" + draft
    rest = tok.est(instr + ctx_head + tail)
    per = (TARGETS["critique"] - rest) / N_CLAUSES
    return instr + ctx_head + clause_block(tok, pairs, per, rng) + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True, help="JSON list of claim records")
    ap.add_argument("--clauses", help="optional policy JSON ({\"clauses\": [...]}), list of policies, or list of clauses")
    ap.add_argument("--base", default="http://10.4.36.58:30000", help="llama.cpp server (for /tokenize)")
    ap.add_argument("--n", type=int, default=20, help="prompts per profile")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="bench_prompts.json")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    claims = [c for c in json.load(open(a.claims)) if c.get("status") == "DENIED"]
    clauses = load_clauses(a.clauses) if a.clauses else None
    if clauses and len(clauses) < N_CLAUSES:
        print("WARNING: only %d clauses in %s; need at least %d. Using placeholders instead."
              % (len(clauses), a.clauses, N_CLAUSES))
        clauses = None
    tok = Tok(a.base)

    out = {}
    for profile in TARGETS:
        picks = rng.sample(claims, min(a.n, len(claims)))
        out[profile] = [build(tok, c, clauses, rng, profile) for c in picks]
    json.dump(out, open(a.out, "w"), indent=1)

    ref = "Summarize the following in two sentences. " + filler(tok, 62, rng)
    open("reference_prompt.txt", "w").write(ref)

    print("%-11s %6s %6s %6s %6s %7s" % ("profile", "target", "min", "median", "max", "prompts"))
    for profile, prompts in out.items():
        n = [tok.count(p) for p in prompts]
        print("%-11s %6d %6d %6d %6d %7d" % (profile, TARGETS[profile], min(n), statistics.median(n), max(n), len(n)))
    print("reference   target     78 -> actual %d tokens" % tok.count(ref))
    print("\nIf a median is more than ~10%% off target, change TARGETS and re-run. Wrote %s and reference_prompt.txt" % a.out)


if __name__ == "__main__":
    main()
