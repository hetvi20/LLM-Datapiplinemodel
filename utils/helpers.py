"""
utils/helpers.py — Shared utility functions used across all modules
"""
import anthropic
import sys
import os
import math

# ── Allow imports from project root ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import ANTHROPIC_API_KEY, SONNET, HAIKU


def get_client() -> anthropic.Anthropic:
    """Return an Anthropic client, raising clearly if key is missing."""
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to your .env file or environment."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def call_model(
    prompt: str,
    system: str = "",
    model: str = SONNET,
    max_tokens: int = 1024,
) -> str:
    """Single-turn call — returns the text response."""
    client = get_client()
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a ** 2 for a in vec_a))
    norm_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def word_count(text: str) -> int:
    return len(text.split())


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(label: str, value: str) -> None:
    print(f"\n[{label}]\n{value}\n")
