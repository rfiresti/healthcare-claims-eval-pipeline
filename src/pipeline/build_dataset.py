import json
import random
from pathlib import Path

CLAUSE_POOL = [
    {"id": "CLS-1001", "cat": "Exclusion", "text": "Experimental procedures are strictly excluded from coverage."},
    {"id": "CLS-1002", "cat": "PreAuth", "text": "Advanced imaging requires prior authorization within 48 hours."},
    {"id": "CLS-1003", "cat": "Limitation", "text": "Physical therapy capped at 20 visits per calendar year."},
    {"id": "CLS-1004", "cat": "MedicalNecessity", "text": "Inpatient admission requires proof of clinical deterioration."}
]

def generate_labeled_claim(idx: int) -> dict:
    selected_clause = random.choice(CLAUSE_POOL)
    is_denied = random.choice([True, False])
    
    return {
        "claim_id": f"CLM-{idx:06d}",
        "patient_id": f"PAT-{random.randint(10000, 99999)}",
        "billed_amount": round(random.uniform(150.0, 12000.0), 2),
        "status": "DENIED" if is_denied else "APPROVED",
        "ground_truth_clause_id": selected_clause["id"],
        "ground_truth_category": selected_clause["cat"],
        "denial_details": {
            "reason": f"Non-compliance with policy clause {selected_clause['id']}",
            "text": selected_clause["text"]
        } if is_denied else None
    }

def run_pipeline(target_count: int = 1050, output_path: str = 'data/synthetic_output/labeled_claims.json'):
    dataset = [generate_labeled_claim(i) for i in range(1, target_count + 1)]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Successfully generated {len(dataset)} synthetic claim records at '{output_path}'.")

if __name__ == '__main__':
    run_pipeline()

