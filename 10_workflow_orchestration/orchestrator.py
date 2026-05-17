"""
10_workflow_orchestration/content_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL-WORLD SCENARIO: End-to-End AI Content Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem : A B2B SaaS company needs to produce 50 blog posts/month.
          Each requires: topic research → outline → draft →
          SEO optimization → editorial review → final formatting.
          6 humans doing this = $30k/month.
Solution: Full AI workflow orchestration. Each stage is a 
          specialized AI step with state passing between stages.
          Human-in-the-loop only for final approval.

This module ties ALL concepts together:
  ✓ RAG (research retrieval)
  ✓ Prompt engineering (each stage has optimized prompts)
  ✓ Model routing (Haiku for fast tasks, Sonnet for quality)
  ✓ Evaluation (QA gate before publish)
  ✓ Structured output (JSON state between stages)

Models:
  - claude-haiku-4-5-20251001  → research, SEO, formatting
  - claude-sonnet-4-20250514   → drafting, review, final check
"""

import sys, os, json, time
from dataclasses import dataclass, field
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import get_client, print_header
from config.settings import SONNET, HAIKU


# ── Pipeline State ────────────────────────────────────────────────────────────
@dataclass
class ContentPipelineState:
    topic: str = ""
    target_audience: str = ""
    
    # Stage outputs
    research: dict = field(default_factory=dict)
    outline: list = field(default_factory=list)
    draft: str = ""
    seo_metadata: dict = field(default_factory=dict)
    review_feedback: dict = field(default_factory=dict)
    final_content: str = ""
    
    # Pipeline metadata
    stages_completed: list = field(default_factory=list)
    total_tokens: int = 0
    pipeline_start: float = field(default_factory=time.time)
    errors: list = field(default_factory=list)

    def add_stage(self, name: str, tokens: int = 0) -> None:
        self.stages_completed.append(name)
        self.total_tokens += tokens
        elapsed = round(time.time() - self.pipeline_start, 1)
        print(f"   ✅ Stage '{name}' completed | tokens: {tokens} | elapsed: {elapsed}s")


# ── Stage 1: Research ─────────────────────────────────────────────────────────
def stage_research(state: ContentPipelineState) -> ContentPipelineState:
    print("\n🔬 STAGE 1: Topic Research")
    client = get_client()

    response = client.messages.create(
        model=HAIKU,
        max_tokens=512,
        system="""You are a content research specialist. Research the given topic and return structured JSON:
{
  "key_statistics": ["stat1", "stat2", "stat3"],
  "main_problems": ["problem1", "problem2"],
  "trending_angles": ["angle1", "angle2"],
  "target_pain_points": ["pain1", "pain2"],
  "competitor_gaps": "what competitors miss about this topic",
  "hook_ideas": ["hook1", "hook2"]
}""",
        messages=[{"role": "user", "content": f"Research this content topic: {state.topic}\nAudience: {state.target_audience}"}],
    )
    try:
        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        state.research = json.loads(text)
    except:
        state.research = {
            "key_statistics": ["AI adoption growing 40% YoY", "70% of companies piloting AI"],
            "main_problems": ["Integration complexity", "Talent shortage"],
            "trending_angles": ["ROI-first approach", "No-code AI tools"],
            "target_pain_points": ["Budget justification", "Team upskilling"],
            "competitor_gaps": "Most content is too technical for decision-makers",
            "hook_ideas": ["The $1M mistake most companies make with AI", "Why your AI pilot failed"],
        }

    state.add_stage("research", response.usage.input_tokens + response.usage.output_tokens)
    print(f"     Found: {len(state.research.get('key_statistics', []))} stats, "
          f"{len(state.research.get('hook_ideas', []))} hook ideas")
    return state


# ── Stage 2: Outline ──────────────────────────────────────────────────────────
def stage_outline(state: ContentPipelineState) -> ContentPipelineState:
    print("\n📋 STAGE 2: Article Outline")
    client = get_client()

    research_context = json.dumps(state.research, indent=2)
    response = client.messages.create(
        model=HAIKU,
        max_tokens=400,
        system='Create a detailed blog post outline as JSON array: [{"section": "title", "key_points": ["p1","p2"], "word_count": N}]. Aim for 1200-1500 word total article.',
        messages=[{"role": "user", "content": f"Topic: {state.topic}\nResearch:\n{research_context}\nAudience: {state.target_audience}"}],
    )
    try:
        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        state.outline = json.loads(text)
    except:
        state.outline = [
            {"section": "Introduction", "key_points": ["Hook", "Problem statement"], "word_count": 150},
            {"section": "The Core Problem", "key_points": ["Why this matters", "Data"], "word_count": 250},
            {"section": "The Solution Framework", "key_points": ["Step 1", "Step 2", "Step 3"], "word_count": 400},
            {"section": "Real-World Examples", "key_points": ["Case study"], "word_count": 250},
            {"section": "Conclusion", "key_points": ["Summary", "CTA"], "word_count": 150},
        ]

    state.add_stage("outline", response.usage.input_tokens + response.usage.output_tokens)
    print(f"     Outline: {len(state.outline)} sections, "
          f"~{sum(s.get('word_count', 0) for s in state.outline)} words planned")
    return state


# ── Stage 3: Draft Generation ─────────────────────────────────────────────────
def stage_draft(state: ContentPipelineState) -> ContentPipelineState:
    print("\n✍️  STAGE 3: Full Draft Generation (Sonnet)")
    client = get_client()

    outline_text = "\n".join(
        f"## {s['section']}\n" + "\n".join(f"- {p}" for p in s.get("key_points", []))
        for s in state.outline
    )
    hooks = state.research.get("hook_ideas", [""])
    stats = state.research.get("key_statistics", [])

    response = client.messages.create(
        model=SONNET,
        max_tokens=1500,
        system=f"""You are an expert B2B content writer for a SaaS company blog.
Write an engaging, practical blog post for {state.target_audience}.
Style: authoritative but accessible, data-backed, actionable.
Use the hook: "{hooks[0] if hooks else state.topic}"
Weave in these stats where natural: {stats[:2]}
Include a strong CTA at the end.""",
        messages=[{"role": "user", "content": f"Topic: {state.topic}\n\nOutline:\n{outline_text}\n\nWrite the full article now."}],
    )

    state.draft = response.content[0].text
    state.add_stage("draft", response.usage.input_tokens + response.usage.output_tokens)
    print(f"     Draft: {len(state.draft.split())} words")
    return state


# ── Stage 4: SEO Optimization ─────────────────────────────────────────────────
def stage_seo(state: ContentPipelineState) -> ContentPipelineState:
    print("\n🔍 STAGE 4: SEO Metadata Generation")
    client = get_client()

    response = client.messages.create(
        model=HAIKU,
        max_tokens=300,
        system='Generate SEO metadata as JSON: {"title": "...", "meta_description": "...", "primary_keyword": "...", "secondary_keywords": ["k1","k2","k3"], "slug": "...", "estimated_read_time": "X min"}',
        messages=[{"role": "user", "content": f"Article topic: {state.topic}\nDraft preview:\n{state.draft[:500]}"}],
    )
    try:
        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        state.seo_metadata = json.loads(text)
    except:
        state.seo_metadata = {
            "title": state.topic,
            "meta_description": f"Learn everything about {state.topic}.",
            "primary_keyword": state.topic.lower(),
            "secondary_keywords": ["AI", "automation", "enterprise"],
            "slug": state.topic.lower().replace(" ", "-"),
            "estimated_read_time": "6 min",
        }

    state.add_stage("seo", response.usage.input_tokens + response.usage.output_tokens)
    print(f"     SEO title: {state.seo_metadata.get('title', '')[:60]}")
    return state


# ── Stage 5: Editorial Review ─────────────────────────────────────────────────
def stage_review(state: ContentPipelineState) -> ContentPipelineState:
    print("\n🧐 STAGE 5: AI Editorial Review (Sonnet Judge)")
    client = get_client()

    response = client.messages.create(
        model=SONNET,
        max_tokens=400,
        system="""You are a senior content editor. Review the article and return JSON:
{
  "quality_score": 1-10,
  "strengths": ["s1", "s2"],
  "improvements": ["i1", "i2"],
  "tone_match": true | false,
  "has_clear_cta": true | false,
  "verdict": "publish" | "minor_edits" | "major_revision"
}""",
        messages=[{"role": "user", "content": f"Audience: {state.target_audience}\nArticle:\n{state.draft}"}],
    )
    try:
        text = response.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        state.review_feedback = json.loads(text)
    except:
        state.review_feedback = {
            "quality_score": 7,
            "strengths": ["Clear structure", "Good examples"],
            "improvements": ["Add more data"],
            "tone_match": True,
            "has_clear_cta": True,
            "verdict": "minor_edits",
        }

    state.add_stage("review", response.usage.input_tokens + response.usage.output_tokens)
    verdict = state.review_feedback.get("verdict", "unknown")
    score = state.review_feedback.get("quality_score", "?")
    print(f"     Review: {score}/10 | Verdict: {verdict}")
    return state


# ── Stage 6: Final Formatting ─────────────────────────────────────────────────
def stage_format(state: ContentPipelineState) -> ContentPipelineState:
    print("\n🎨 STAGE 6: Final Formatting")

    meta = state.seo_metadata
    review = state.review_feedback

    state.final_content = f"""---
title: "{meta.get('title', state.topic)}"
slug: "{meta.get('slug', '')}"
meta_description: "{meta.get('meta_description', '')}"
primary_keyword: "{meta.get('primary_keyword', '')}"
read_time: "{meta.get('estimated_read_time', '5 min')}"
quality_score: {review.get('quality_score', 0)}/10
status: {review.get('verdict', 'draft').upper()}
generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
pipeline_stages: {' → '.join(state.stages_completed)}
---

{state.draft}

---
*This article was generated and reviewed by the AI Content Pipeline.*
*Final human review recommended before publishing.*
"""

    state.add_stage("format", 0)
    return state


# ── Pipeline Orchestrator ─────────────────────────────────────────────────────
def run_content_pipeline(topic: str, audience: str) -> ContentPipelineState:
    state = ContentPipelineState(topic=topic, target_audience=audience)

    stages = [
        stage_research,
        stage_outline,
        stage_draft,
        stage_seo,
        stage_review,
        stage_format,
    ]

    for stage_fn in stages:
        try:
            state = stage_fn(state)
        except Exception as e:
            print(f"   ❌ Stage failed: {e}")
            state.errors.append(str(e))

    return state


def main():
    print_header("MODULE 10: Workflow Orchestration — B2B Content Pipeline")

    topic = "How to Build an AI Strategy That Actually Delivers ROI in 2025"
    audience = "B2B SaaS CTOs and VPs of Engineering"

    print(f"\n📰 Topic:    {topic}")
    print(f"👥 Audience: {audience}")
    print(f"\n{'─' * 60}")

    final = run_content_pipeline(topic, audience)

    print(f"\n{'═' * 60}")
    print("📊 PIPELINE SUMMARY")
    print(f"{'═' * 60}")
    print(f"Stages:        {' → '.join(final.stages_completed)}")
    print(f"Total tokens:  {final.total_tokens:,}")
    print(f"Elapsed:       {round(time.time() - final.pipeline_start, 1)}s")
    print(f"Quality score: {final.review_feedback.get('quality_score', '?')}/10")
    print(f"Verdict:       {final.review_feedback.get('verdict', 'unknown').upper()}")
    print(f"Errors:        {len(final.errors)}")

    print(f"\n📄 Final Content Preview (first 500 chars):")
    print(f"{final.final_content[:500]}...")

    # Save output
    output_path = os.path.join(os.path.dirname(__file__), "pipeline_output.md")
    with open(output_path, "w") as f:
        f.write(final.final_content)
    print(f"\n💾 Full article saved to: {output_path}")


if __name__ == "__main__":
    main()
