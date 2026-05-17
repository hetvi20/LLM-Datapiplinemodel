"""
config/settings.py — Central configuration for all modules
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Models ──────────────────────────────────────────────────────────────────
SONNET = "claude-sonnet-4-20250514"   # Deep reasoning, agents, evaluation
HAIKU  = "claude-haiku-4-5-20251001"  # Fast, cheap, classification

# ── API ─────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Inference defaults ───────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 1024
AGENT_MAX_TOKENS   = 2048

# ── Routing thresholds ───────────────────────────────────────────────────────
COMPLEXITY_THRESHOLD = 50   # words; above → Sonnet, below → Haiku
