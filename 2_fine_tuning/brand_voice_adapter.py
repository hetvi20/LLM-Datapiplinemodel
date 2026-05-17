"""
2_fine_tuning/brand_voice_adapter.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL-WORLD SCENARIO: E-Commerce Brand Voice Adaptation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem : An e-commerce brand (think Glossier meets Patagonia)
          needs all AI-generated content to sound "on-brand" —
          not like generic ChatGPT output.
Solution: Use few-shot examples in the system prompt to simulate
          fine-tuning behavior. Demonstrate before/after quality.

Models used:
  - claude-haiku-4-5-20251001  → generic (no brand voice)
  - claude-sonnet-4-20250514   → with brand voice system prompt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import get_client, print_header
from config.settings import SONNET, HAIKU

# ── Brand: "EcoBloom" — sustainable skincare brand ───────────────────────────
BRAND_SYSTEM_PROMPT = """You are the content writer for EcoBloom, a sustainable skincare brand.

BRAND VOICE GUIDELINES:
- Warm, honest, and science-grounded (never hype, never exaggeration)
- Always mention sustainability or environmental values naturally
- Short punchy sentences mixed with occasional longer ones
- Use "you" and "your skin" — never clinical or detached
- Avoid: "revolutionary", "game-changing", "holy grail", "must-have"
- Tone: like a knowledgeable friend who genuinely cares

FEW-SHOT EXAMPLES:

Input: Write product description for a face wash
Output: This gentle cleanser works with your skin, not against it.
Crafted with cold-pressed botanicals sourced from regenerative farms,
it removes the day without stripping your barrier. Your skin feels clean.
Balanced. Ready. Our packaging? 100% ocean-bound recycled plastic.
Because good skin and a healthy planet aren't mutually exclusive.

Input: Write an email subject line for a 20% off sale
Output: A little something for you (and the planet)

Input: Write an Instagram caption for a moisturizer
Output: Your skin is working hard. Give it what it actually needs.
No fillers. No fluff. Just 12 barrier-supporting botanicals that
your skin recognizes as nutrition. Tap to learn what's inside →
#CleanBeauty #EcoBloom #SustainableSkincare"""


def generate_generic(prompt: str) -> str:
    """Generate with Haiku, no brand voice."""
    client = get_client()
    response = client.messages.create(
        model=HAIKU,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_branded(prompt: str) -> str:
    """Generate with Sonnet + brand voice system prompt."""
    client = get_client()
    response = client.messages.create(
        model=SONNET,
        max_tokens=256,
        system=BRAND_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main():
    print_header("MODULE 2: Fine-Tuning Simulation — EcoBloom Brand Voice")

    tasks = [
        "Write a product description for a vitamin C serum.",
        "Write a subject line for a new product launch email.",
        "Write an Instagram caption for a sunscreen product.",
        "Write a 2-sentence response to a customer who says the product is too expensive.",
    ]

    for task in tasks:
        print(f"\n📝 Task: {task}")
        print("\n🤖 Generic (Haiku, no brand voice):")
        generic = generate_generic(task)
        print(f"   {generic}")

        print("\n✨ Branded (Sonnet + EcoBloom voice):")
        branded = generate_branded(task)
        print(f"   {branded}")
        print("\n" + "─" * 60)


if __name__ == "__main__":
    main()
