"""
3_prompt_engineering/customer_support.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL-WORLD SCENARIO: SaaS Customer Support — 95% Resolution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem : A SaaS company's support tickets take 4 hours average
          to resolve. 60% are repetitive billing/tech issues.
Solution: Prompt engineering techniques to handle tickets:
          - Chain-of-thought for issue diagnosis
          - Few-shot for tone calibration
          - Structured output for routing
          - Persona for brand consistency

Models used:
  - claude-haiku-4-5-20251001  → triage/classification (fast)
  - claude-sonnet-4-20250514   → full resolution drafting
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import get_client, print_header
from config.settings import SONNET, HAIKU

# ── TECHNIQUE 1: Structured output for ticket triage ────────────────────────
TRIAGE_SYSTEM = """You are a support ticket classifier. Respond ONLY with valid JSON.
Classify the ticket into:
{
  "category": "billing" | "technical" | "account" | "feature_request" | "other",
  "priority": "low" | "medium" | "high" | "urgent",
  "sentiment": "positive" | "neutral" | "frustrated" | "angry",
  "can_self_serve": true | false,
  "estimated_complexity": 1-5
}"""


def triage_ticket(ticket: str) -> dict:
    client = get_client()
    response = client.messages.create(
        model=HAIKU,
        max_tokens=128,
        system=TRIAGE_SYSTEM,
        messages=[{"role": "user", "content": f"Ticket: {ticket}"}],
    )
    try:
        return json.loads(response.content[0].text)
    except json.JSONDecodeError:
        return {"error": "parse_failed", "raw": response.content[0].text}


# ── TECHNIQUE 2: Chain-of-thought for issue resolution ──────────────────────
RESOLUTION_SYSTEM = """You are Alex, a senior support engineer at CloudSync (a project management SaaS).

PERSONALITY: Calm, empathetic, technically precise. Never blame the customer.

RESOLUTION PROCESS (think step by step internally):
1. Acknowledge the customer's frustration/question
2. Diagnose the root cause based on the ticket
3. Provide a clear, numbered solution
4. Offer a follow-up if the solution doesn't work
5. End with a warm close

CONSTRAINTS:
- Keep responses under 150 words
- Use simple language, no jargon unless necessary
- If it's a billing issue, always say "I'll make this right"
- Never say "Unfortunately" — say "Here's what we can do"

FEW-SHOT EXAMPLES:
Customer: I can't log in, keeps saying wrong password
Alex: Hi! I can help you get back in right away.
Your account may have triggered our security lock after multiple attempts.
Here's what to do:
1. Go to cloudsync.com/reset
2. Enter your email and click "Send Reset Link"
3. Check spam if you don't see it in 2 minutes
If that doesn't work, reply here and I'll manually unlock your account.
You'll be back in within minutes! — Alex"""


def resolve_ticket(ticket: str, triage: dict) -> str:
    client = get_client()

    context = f"""Triage info: Category={triage.get('category')}, 
Priority={triage.get('priority')}, Sentiment={triage.get('sentiment')}

Customer ticket: {ticket}"""

    response = client.messages.create(
        model=SONNET,
        max_tokens=300,
        system=RESOLUTION_SYSTEM,
        messages=[{"role": "user", "content": context}],
    )
    return response.content[0].text


# ── Demo tickets ─────────────────────────────────────────────────────────────
TICKETS = [
    "I was charged twice this month!! This is ridiculous, fix this NOW.",
    "How do I export my project data to CSV? I've looked everywhere.",
    "Your app crashes every time I try to upload files larger than 10MB. Very annoying.",
    "Can you add a dark mode? It would really help me work at night.",
]


def main():
    print_header("MODULE 3: Prompt Engineering — SaaS Customer Support")

    for i, ticket in enumerate(TICKETS, 1):
        print(f"\n🎫 Ticket #{i}: {ticket}")

        # Fast Haiku triage
        triage = triage_ticket(ticket)
        print(f"🏷️  Triage (Haiku): {json.dumps(triage, indent=2)}")

        # Sonnet resolution
        reply = resolve_ticket(ticket, triage)
        print(f"💬 Response (Sonnet):\n{reply}")
        print("─" * 60)


if __name__ == "__main__":
    main()
