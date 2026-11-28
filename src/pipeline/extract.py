import json
import os
import urllib.request

DEFAULT_LLM_URL = os.environ.get("LLM_CHAT_URL", "http://10.4.36.58:30000/v1/chat/completions")

EXTRACT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "claim_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "denial_code": {"type": ["string", "null"]},
                "provider": {"type": ["string", "null"]},
                "service_dates": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "billed_amount": {"type": ["number", "null"]},
                "denied_amount": {"type": ["number", "null"]},
                "denial_category": {
                    "type": "string",
                    "enum": ["Exclusion", "PreAuth", "Limitation", "MedicalNecessity"],
                },
            },
            "required": [
                "denial_code",
                "provider",
                "service_dates",
                "billed_amount",
                "denied_amount",
                "denial_category",
            ],
        },
    },
}

SYSTEM_PROMPT = (
    "Extract denial_code, provider, service_dates, billed_amount, denied_amount, "
    "and classify denial_category into exactly one of: "
    "Exclusion, PreAuth, Limitation, MedicalNecessity.\n"
    "Use null when a field is not present in the claim.\n\n"
    "Category rules:\n"
    "- Exclusion: experimental/investigational procedures or services strictly excluded from coverage (CLS-1001).\n"
    "- PreAuth: missing prior authorization or pre-certification within required timeframe (CLS-1002).\n"
    "- Limitation: benefit caps, visit limits, dollar limits, or coverage maximums (CLS-1003).\n"
    "- MedicalNecessity: clinical necessity requirements, medical criteria, or proof of clinical deterioration (CLS-1004)."
)


def extract(claim: dict, llm_url: str = DEFAULT_LLM_URL, timeout_s: float = 90.0) -> dict:
    sanitized = {k: v for k, v in claim.items() if not k.startswith("ground_truth_")}
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(sanitized, default=str)},
        ],
        "response_format": EXTRACT_SCHEMA,
        "temperature": 0.0,
        "max_tokens": 300,
    }
    req = urllib.request.Request(
        llm_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return json.loads(body["choices"][0]["message"]["content"])
