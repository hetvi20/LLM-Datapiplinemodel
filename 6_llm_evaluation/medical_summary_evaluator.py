"""
6_llm_evaluation/medical_summary_evaluator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL-WORLD SCENARIO: QA Pipeline for AI Medical Summaries
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem : A health-tech company uses AI to summarize patient
          records for doctors. Wrong summaries can harm patients.
          They need automated evaluation before summaries go live.
Solution: LLM-as-judge evaluation pipeline scoring:
          accuracy, completeness, safety, readability, hallucination.

Pattern: Judge model evaluates generator model outputs.

Models:
  - claude-haiku-4-5-20251001   → fast summary generation
  - claude-sonnet-4-20250514    → high-quality evaluation judge
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import get_client, print_header
from config.settings import SONNET, HAIKU

# ── Medical case data ─────────────────────────────────────────────────────────
PATIENT_RECORDS = [
    {
        "id": "patient_001",
        "record": """
        Patient: Male, 67 years old. Chief complaint: chest pain and shortness of breath.
        Labs: Troponin elevated at 2.3 ng/mL (normal <0.04). EKG shows ST elevation in leads V1-V4.
        Diagnosis: STEMI (ST-elevation myocardial infarction).
        Treatment: Immediate PCI performed. Aspirin 325mg, Heparin, Clopidogrel administered.
        Allergies: Penicillin (rash). No known drug interactions identified.
        Outcome: Hemodynamically stable post-procedure. ICU monitoring ordered.
        Follow-up: Cardiology consult in 48 hours, echo scheduled.
        """,
    },
    {
        "id": "patient_002",
        "record": """
        Patient: Female, 34 years old. Visit for prenatal check, 28 weeks gestation.
        BP: 158/102 (elevated). Protein in urine: 2+ (300mg/24hr). 
        Symptoms: Headache, visual disturbances, right upper quadrant pain.
        Diagnosis: Preeclampsia with severe features.
        Plan: Admitted for monitoring. Magnesium sulfate started for seizure prophylaxis.
        Betamethasone given for fetal lung maturity. OB consult ordered immediately.
        Risks discussed with patient. Delivery may be required if condition worsens.
        """,
    },
]

# ── Evaluation criteria ───────────────────────────────────────────────────────
EVALUATION_SYSTEM = """You are a senior medical AI evaluator at a health-tech company.
Evaluate AI-generated medical summaries on these criteria. 
Respond ONLY with valid JSON.

Criteria:
1. accuracy (1-10): Are key medical facts correctly captured?
2. completeness (1-10): Are critical details (diagnosis, treatment, allergies, follow-up) included?
3. safety (1-10): Are safety-critical details (allergies, contraindications) preserved?
4. hallucination_risk (1-10): 10 = no hallucinations detected, 1 = clear fabrications
5. readability (1-10): Is it clear and useful for a busy physician?
6. overall (1-10): Overall quality score

Also provide:
- "flags": list of specific concerns (e.g., "missing allergy info", "incorrect dosage")
- "verdict": "approved" | "needs_revision" | "rejected"

Format:
{
  "accuracy": N, "completeness": N, "safety": N,
  "hallucination_risk": N, "readability": N, "overall": N,
  "flags": ["..."],
  "verdict": "..."
}"""


def generate_summary(record: str) -> str:
    """Generate summary using Haiku (fast, cost-efficient)."""
    client = get_client()
    response = client.messages.create(
        model=HAIKU,
        max_tokens=300,
        system="You are a medical summarization AI. Create a concise physician-ready summary of this patient record. Include: chief complaint, diagnosis, key labs, treatment, allergies, and follow-up plan.",
        messages=[{"role": "user", "content": f"Summarize this patient record:\n{record}"}],
    )
    return response.content[0].text


def evaluate_summary(original: str, summary: str) -> dict:
    """Evaluate summary using Sonnet as judge."""
    client = get_client()
    response = client.messages.create(
        model=SONNET,
        max_tokens=512,
        system=EVALUATION_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"ORIGINAL RECORD:\n{original}\n\nAI-GENERATED SUMMARY:\n{summary}\n\nEvaluate this summary."
        }],
    )
    try:
        return json.loads(response.content[0].text)
    except:
        return {"error": "parse_failed", "raw": response.content[0].text}


def main():
    print_header("MODULE 6: LLM Evaluation — Medical Summary QA Pipeline")

    all_verdicts = []

    for patient in PATIENT_RECORDS:
        print(f"\n📋 Processing {patient['id']}")
        print(f"Original record length: {len(patient['record'].split())} words")

        # Generate
        print("⚙️  Generating summary (Haiku)...")
        summary = generate_summary(patient["record"])
        print(f"\n📝 Generated Summary:\n{summary}")

        # Evaluate
        print("\n🔍 Evaluating summary (Sonnet judge)...")
        evaluation = evaluate_summary(patient["record"], summary)

        if "error" not in evaluation:
            print(f"\n📊 Evaluation Results:")
            print(f"   Accuracy:        {evaluation.get('accuracy', 'N/A')}/10")
            print(f"   Completeness:    {evaluation.get('completeness', 'N/A')}/10")
            print(f"   Safety:          {evaluation.get('safety', 'N/A')}/10")
            print(f"   Anti-Halluc.:    {evaluation.get('hallucination_risk', 'N/A')}/10")
            print(f"   Readability:     {evaluation.get('readability', 'N/A')}/10")
            print(f"   ─────────────────────────")
            print(f"   Overall:         {evaluation.get('overall', 'N/A')}/10")
            print(f"   Verdict:         {evaluation.get('verdict', 'N/A').upper()}")
            flags = evaluation.get("flags", [])
            if flags:
                print(f"   ⚠️  Flags: {', '.join(flags)}")
            all_verdicts.append(evaluation.get("verdict", "unknown"))
        else:
            print(f"   Error: {evaluation}")

        print("─" * 60)

    # Pipeline summary
    approved = all_verdicts.count("approved")
    print(f"\n📈 Pipeline Summary: {approved}/{len(all_verdicts)} summaries approved")


if __name__ == "__main__":
    main()
