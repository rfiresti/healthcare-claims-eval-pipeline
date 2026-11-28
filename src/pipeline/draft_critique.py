import json
import os
import re
import urllib.request

DEFAULT_LLM_URL = os.environ.get("LLM_CHAT_URL", "http://10.4.36.58:30000/v1/chat/completions")
MAX_DRAFT_TOKENS = 350
MAX_REVISIONS = 1

CLAUSE_ID_RE = re.compile(r"\bCLS-\d+\b")
QUOTED_SPAN_RE = re.compile(r'"([^"]+)"|\'([^\']{12,})\'')


def _normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def has_five_word_verbatim_match(span: str, source_texts: list[str]) -> bool:
    span_words = _normalize_words(span)
    if len(span_words) < 5:
        return False
    windows = [" ".join(span_words[i : i + 5]) for i in range(len(span_words) - 4)]
    normalized_sources = [" ".join(_normalize_words(src)) for src in source_texts]
    return all(any(w in src for src in normalized_sources) for w in windows)


def draft_appeal(
    claim: dict,
    extracted: dict,
    clauses: list[dict],
    critique_feedback: str | None = None,
    llm_url: str = DEFAULT_LLM_URL,
    timeout_s: float = 90.0,
) -> str:
    clauses_block = "\n".join(
        f"- [{c.get('clause_id')}] {c.get('text', c.get('clause_text', ''))}"
        for c in clauses
    )
    revision_note = (
        f"\nRevision instructions from critique (fix every item):\n{critique_feedback}"
        if critique_feedback
        else ""
    )
    prompt = (
        f"Claim summary:\n{json.dumps(claim, default=str)}\n\n"
        f"Extracted fields:\n{json.dumps(extracted, default=str)}\n\n"
        f"Retrieved policy clauses (cite ONLY from these):\n{clauses_block}\n"
        f"{revision_note}\n\n"
        "Write a concise insurance claim appeal letter grounding every policy claim in "
        "the retrieved clauses above. Include exact clause IDs (CLS-XXXX) and quote "
        "exact wording from the retrieved clauses when citing policy language."
    )
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a healthcare appeals specialist. Stay strictly within 350 tokens. "
                    "Only cite clause IDs present in the provided policy clauses, and only use "
                    "verbatim quotes that appear word-for-word in the provided policy clauses."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": MAX_DRAFT_TOKENS,
    }
    req = urllib.request.Request(
        llm_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def critique_draft(draft_text: str, clauses: list[dict], revision_count: int) -> dict:
    reasons: list[str] = []
    allowed_ids = {c.get("clause_id") for c in clauses if c.get("clause_id")}
    source_texts = [c.get("text", c.get("clause_text", "")) for c in clauses]

    approx_tokens = len(re.findall(r"\S+", draft_text))
    if approx_tokens > MAX_DRAFT_TOKENS:
        reasons.append(f"Draft length ({approx_tokens} words/tokens) exceeds {MAX_DRAFT_TOKENS} cap.")

    cited_ids = set(CLAUSE_ID_RE.findall(draft_text))
    if not cited_ids:
        reasons.append("No policy clause ID (CLS-XXXX) cited in draft.")
    invalid_ids = sorted(cited_ids - allowed_ids)
    if invalid_ids:
        reasons.append(f"Cited clause IDs not in retrieved clauses: {', '.join(invalid_ids)}.")

    quoted_spans = [
        (m.group(1) or m.group(2)).strip()
        for m in QUOTED_SPAN_RE.finditer(draft_text)
        if (m.group(1) or m.group(2)) and len(_normalize_words(m.group(1) or m.group(2))) >= 5
    ]
    for span in quoted_spans:
        if not has_five_word_verbatim_match(span, source_texts):
            reasons.append(f"Quote lacks 5-word verbatim match in retrieved clauses: \"{span[:80]}\"")

    if not reasons:
        return {
            "passed": True,
            "grounding_status": "grounded",
            "critique_feedback": "",
        }

    if revision_count >= MAX_REVISIONS:
        return {
            "passed": False,
            "grounding_status": "ungrounded",
            "critique_feedback": "; ".join(reasons),
        }

    return {
        "passed": False,
        "grounding_status": "needs_revision",
        "critique_feedback": "; ".join(reasons),
    }
