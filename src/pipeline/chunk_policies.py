# This file will turn a single policy document into chunks for embedding and storage. 
# Each chunk is a single clause, or a piece of a clause if the clause is too long.
import json
import re
from jsonschema import validate

# Roughly how much text the embedding model can read at once
MAX_CHARS = 1000


def load_policies(path, schema_path="data/schemas/policy_schema.json"):
    """Read a JSON file holding one policy or a list of policies, and check the format."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    with open(schema_path) as f:
        schema = json.load(f)
    for policy in data:
        validate(instance=policy, schema=schema)  # raises an error if a policy is malformed
    return data


def split_long_text(text, max_chars=MAX_CHARS):
    """Short text stays whole and as is. If we have too much text, split it at sentence ends."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    pieces.append(current)
    return pieces


def chunk_policy(policy):
    """Return a list of chunks for one policy."""
    chunks = []
    for clause in policy["clauses"]:
        pieces = split_long_text(clause["clause_text"])
        for number, piece in enumerate(pieces):
            chunks.append({
                "clause_id": clause["clause_id"],
                "chunk_index": number,
                "text": piece,
            })
    return chunks


# can run this file directly to preview the chunks without touching the database
#   python src/pipeline/chunk_policies.py tests/fixtures/sample_policy.json
if __name__ == "__main__":
    import sys
    for policy in load_policies(sys.argv[1]):
        for chunk in chunk_policy(policy):
            print(policy["policy_id"], chunk["clause_id"], chunk["chunk_index"], chunk["text"][:60])