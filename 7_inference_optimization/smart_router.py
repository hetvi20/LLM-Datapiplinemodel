"""
7_inference_optimization/smart_router.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL-WORLD SCENARIO: Inference Cost Optimization — 60% Savings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem : A startup is spending $50k/month on Sonnet for ALL
          queries — including simple ones like "what's 2+2".
Solution: Smart routing: classify query complexity, use Haiku
          for simple/medium tasks, Sonnet only for complex.
          Add caching for repeated queries.

Techniques: complexity routing, response caching, batching.

Models:
  - claude-haiku-4-5-20251001  → simple + medium queries
  - claude-sonnet-4-20250514   → complex + high-stakes queries
"""

import sys, os, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import get_client, print_header
from config.settings import SONNET, HAIKU

# ── Cost simulation (per 1M tokens, approximate) ────────────────────────────
COSTS = {
    HAIKU:  {"input": 0.80,  "output": 4.00},   # $/1M tokens
    SONNET: {"input": 3.00, "output": 15.00},
}

# ── Response cache (production: use Redis) ───────────────────────────────────
_cache: dict[str, str] = {}


def cache_key(prompt: str, model: str) -> str:
    return hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()


def get_cached(prompt: str, model: str) -> str | None:
    return _cache.get(cache_key(prompt, model))


def set_cache(prompt: str, model: str, response: str) -> None:
    _cache[cache_key(prompt, model)] = response


# ── Complexity classifier ────────────────────────────────────────────────────
def classify_complexity(query: str) -> str:
    """
    Rule-based + heuristic classifier.
    Production: train a small classifier or use embeddings.
    Returns: 'simple' | 'medium' | 'complex'
    """
    q = query.lower()
    word_count = len(query.split())

    # Simple: short factual, math, definitions
    simple_signals = ["what is", "define", "how many", "calculate", "convert",
                      "what does", "who is", "when was", "spell", "translate"]
    if word_count < 15 and any(s in q for s in simple_signals):
        return "simple"

    # Complex: analysis, comparison, strategy, code generation, multi-step
    complex_signals = ["analyze", "compare", "strategy", "architecture", "design",
                       "debug", "explain why", "pros and cons", "trade-off",
                       "write code", "implement", "synthesize", "recommend"]
    if word_count > 30 or any(s in q for s in complex_signals):
        return "complex"

    return "medium"


def route_model(complexity: str) -> str:
    """Route to appropriate model based on complexity."""
    return {
        "simple":  HAIKU,
        "medium":  HAIKU,
        "complex": SONNET,
    }[complexity]


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate cost in USD."""
    rates = COSTS[model]
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


# ── Smart query handler ──────────────────────────────────────────────────────
def smart_query(query: str, system: str = "") -> dict:
    """Route query to best model, check cache, track costs."""
    client = get_client()

    # 1. Classify
    complexity = classify_complexity(query)
    model = route_model(complexity)

    # 2. Check cache
    cached = get_cached(query, model)
    if cached:
        return {
            "response": cached,
            "model": model,
            "complexity": complexity,
            "source": "cache",
            "cost_usd": 0.0,
            "latency_ms": 0,
        }

    # 3. Call model
    start = time.time()
    kwargs = {
        "model": model,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": query}],
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    latency_ms = int((time.time() - start) * 1000)

    text = response.content[0].text
    cost = estimate_cost(
        response.usage.input_tokens,
        response.usage.output_tokens,
        model,
    )

    # 4. Cache result
    set_cache(query, model, text)

    return {
        "response": text,
        "model": model,
        "complexity": complexity,
        "source": "api",
        "cost_usd": cost,
        "latency_ms": latency_ms,
        "tokens_in": response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
    }


# ── Batch processing ─────────────────────────────────────────────────────────
def batch_process(queries: list[str]) -> list[dict]:
    """Process multiple queries with routing stats."""
    results = []
    total_cost = 0.0
    sonnet_cost_if_all = 0.0

    for q in queries:
        result = smart_query(q)
        results.append({"query": q, **result})
        total_cost += result["cost_usd"]

        # Estimate what Sonnet would've cost
        tokens_in = result.get("tokens_in", len(q.split()) + 10)
        tokens_out = result.get("tokens_out", 100)
        sonnet_cost_if_all += estimate_cost(tokens_in, tokens_out, SONNET)

    return results, total_cost, sonnet_cost_if_all


def main():
    print_header("MODULE 7: Inference Optimization — Smart Model Router")

    queries = [
        "What is 15% of $240?",
        "Define machine learning in one sentence.",
        "Analyze the trade-offs between microservices and monolithic architecture for a fintech startup with 10 engineers.",
        "Convert 100 Celsius to Fahrenheit.",
        "Compare the pros and cons of React vs Vue for a large enterprise application with 50 developers.",
        "Who invented the telephone?",
        "Write a Python function to implement binary search with proper error handling and type hints.",
        "What does API stand for?",
    ]

    print(f"Processing {len(queries)} queries with smart routing...\n")
    results, total_cost, sonnet_all_cost = batch_process(queries)

    for r in results:
        model_short = "Haiku" if r["model"] == HAIKU else "Sonnet"
        source_icon = "💾" if r["source"] == "cache" else "🌐"
        print(f"{source_icon} [{r['complexity'].upper():7s}→{model_short:6s}] {r['query'][:60]}")
        print(f"   Cost: ${r['cost_usd']:.6f} | Latency: {r['latency_ms']}ms")
        print(f"   Response preview: {r['response'][:80]}...")
        print()

    savings = sonnet_all_cost - total_cost
    savings_pct = (savings / sonnet_all_cost * 100) if sonnet_all_cost > 0 else 0

    print(f"\n{'─'*50}")
    print(f"💰 Cost Summary:")
    print(f"   If all Sonnet:  ${sonnet_all_cost:.6f}")
    print(f"   Smart routing:  ${total_cost:.6f}")
    print(f"   💵 Savings:     ${savings:.6f} ({savings_pct:.1f}%)")
    print(f"   Cache hits:     {sum(1 for r in results if r['source'] == 'cache')}/{len(results)}")


if __name__ == "__main__":
    main()
