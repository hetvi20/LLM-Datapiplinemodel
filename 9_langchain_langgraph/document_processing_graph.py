"""
9_langchain_langgraph/document_processing_graph.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL-WORLD SCENARIO: Multi-Node Document Processing Graph
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem : A publishing company needs to process research papers:
          classify → extract key findings → generate abstract →
          suggest peer reviewers → quality check.
Solution: LangGraph-style directed acyclic graph (DAG) where
          each node is an AI step, edges are conditional routes.

Pattern: State machine graph — each node receives state, 
         modifies it, returns updated state. Edges route based
         on node output.

Models:
  - claude-haiku-4-5-20251001  → classification, extraction nodes
  - claude-sonnet-4-20250514   → abstract generation, QA nodes
"""

import sys, os, json
from dataclasses import dataclass, field
from typing import Callable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import get_client, print_header
from config.settings import SONNET, HAIKU

# ── Graph State ───────────────────────────────────────────────────────────────
@dataclass
class DocumentState:
    """Shared state flowing through the graph."""
    raw_text: str = ""
    doc_type: str = ""
    domain: str = ""
    key_findings: list[str] = field(default_factory=list)
    abstract: str = ""
    suggested_reviewers: list[str] = field(default_factory=list)
    quality_score: int = 0
    quality_issues: list[str] = field(default_factory=list)
    final_status: str = ""
    node_trace: list[str] = field(default_factory=list)

    def log(self, node: str) -> None:
        self.node_trace.append(node)
        print(f"   → Node executed: {node}")


# ── Graph Nodes ───────────────────────────────────────────────────────────────
def node_classify(state: DocumentState) -> DocumentState:
    """Node 1: Classify document type and domain."""
    state.log("classify")
    client = get_client()

    response = client.messages.create(
        model=HAIKU,
        max_tokens=128,
        system='Classify this academic document. Respond ONLY with JSON: {"doc_type": "...", "domain": "..."}. doc_type options: "research_paper" | "review_article" | "case_study" | "meta_analysis"',
        messages=[{"role": "user", "content": f"Document excerpt:\n{state.raw_text[:500]}"}],
    )
    try:
        result = json.loads(response.content[0].text)
        state.doc_type = result.get("doc_type", "research_paper")
        state.domain = result.get("domain", "unknown")
        print(f"     Classified: {state.doc_type} | Domain: {state.domain}")
    except:
        state.doc_type = "research_paper"
        state.domain = "general"
    return state


def node_extract_findings(state: DocumentState) -> DocumentState:
    """Node 2: Extract key findings from the document."""
    state.log("extract_findings")
    client = get_client()

    response = client.messages.create(
        model=HAIKU,
        max_tokens=300,
        system='Extract the 3-5 key findings from this research document. Return ONLY a JSON array: ["finding 1", "finding 2", ...]',
        messages=[{"role": "user", "content": state.raw_text}],
    )
    try:
        findings_text = response.content[0].text.strip()
        # Clean JSON if wrapped in markdown
        if "```" in findings_text:
            findings_text = findings_text.split("```")[1].replace("json", "").strip()
        state.key_findings = json.loads(findings_text)
        print(f"     Extracted {len(state.key_findings)} key findings")
    except:
        state.key_findings = ["Key finding extraction failed — manual review needed"]
    return state


def node_generate_abstract(state: DocumentState) -> DocumentState:
    """Node 3: Generate a publication-quality abstract."""
    state.log("generate_abstract")
    client = get_client()

    findings_text = "\n".join(f"- {f}" for f in state.key_findings)
    response = client.messages.create(
        model=SONNET,
        max_tokens=350,
        system=f"You are a scientific editor. Generate a structured academic abstract (Background, Methods, Results, Conclusions) for a {state.doc_type} in {state.domain}. Max 250 words.",
        messages=[{"role": "user", "content": f"Document:\n{state.raw_text}\n\nKey findings:\n{findings_text}"}],
    )
    state.abstract = response.content[0].text
    print(f"     Abstract generated: {len(state.abstract.split())} words")
    return state


def node_suggest_reviewers(state: DocumentState) -> DocumentState:
    """Node 4: Suggest appropriate peer reviewers by profile."""
    state.log("suggest_reviewers")
    client = get_client()

    response = client.messages.create(
        model=HAIKU,
        max_tokens=200,
        system='Suggest 3 peer reviewer profiles (not real names) for this paper. Return JSON array: [{"expertise": "...", "institution_type": "..."}, ...]',
        messages=[{"role": "user", "content": f"Domain: {state.domain}\nType: {state.doc_type}\nFindings: {state.key_findings[:2]}"}],
    )
    try:
        reviewers_text = response.content[0].text.strip()
        if "```" in reviewers_text:
            reviewers_text = reviewers_text.split("```")[1].replace("json", "").strip()
        reviewers = json.loads(reviewers_text)
        state.suggested_reviewers = [f"{r.get('expertise')} @ {r.get('institution_type')}" for r in reviewers]
        print(f"     Suggested {len(state.suggested_reviewers)} reviewer profiles")
    except:
        state.suggested_reviewers = ["Domain expert at research university"]
    return state


def node_quality_check(state: DocumentState) -> DocumentState:
    """Node 5: Quality gate — score abstract and decide publish/revise."""
    state.log("quality_check")
    client = get_client()

    response = client.messages.create(
        model=SONNET,
        max_tokens=256,
        system='Quality check this abstract. Return ONLY JSON: {"score": 1-10, "issues": ["issue1", ...], "verdict": "approve" | "revise" | "reject"}',
        messages=[{"role": "user", "content": f"Abstract:\n{state.abstract}"}],
    )
    try:
        result_text = response.content[0].text.strip()
        if "```" in result_text:
            result_text = result_text.split("```")[1].replace("json", "").strip()
        result = json.loads(result_text)
        state.quality_score = result.get("score", 5)
        state.quality_issues = result.get("issues", [])
        state.final_status = result.get("verdict", "revise")
        print(f"     Quality score: {state.quality_score}/10 | Verdict: {state.final_status}")
    except:
        state.quality_score = 5
        state.final_status = "revise"
    return state


def node_revision_request(state: DocumentState) -> DocumentState:
    """Node 6 (conditional): Generate revision instructions if quality fails."""
    state.log("revision_request")
    client = get_client()

    issues = "\n".join(f"- {i}" for i in state.quality_issues)
    response = client.messages.create(
        model=HAIKU,
        max_tokens=200,
        system="Generate specific, actionable revision instructions for the author. Be constructive.",
        messages=[{"role": "user", "content": f"Issues found:\n{issues}\n\nAbstract:\n{state.abstract}"}],
    )
    print(f"     Revision instructions generated")
    state.final_status = f"revise: {response.content[0].text[:100]}..."
    return state


# ── Graph Engine ──────────────────────────────────────────────────────────────
class DocumentGraph:
    def __init__(self):
        self.nodes: dict[str, Callable] = {}
        self.edges: dict[str, str | dict] = {}

    def add_node(self, name: str, fn: Callable) -> None:
        self.nodes[name] = fn

    def add_edge(self, from_node: str, to_node: str) -> None:
        self.edges[from_node] = to_node

    def add_conditional_edge(self, from_node: str, condition: Callable, mapping: dict) -> None:
        self.edges[from_node] = {"condition": condition, "mapping": mapping}

    def run(self, start_node: str, state: DocumentState) -> DocumentState:
        current = start_node
        visited = set()

        while current and current != "END" and current not in visited:
            visited.add(current)
            print(f"\n📍 Running node: [{current}]")
            state = self.nodes[current](state)

            edge = self.edges.get(current)
            if edge is None:
                break
            elif isinstance(edge, str):
                current = edge
            elif isinstance(edge, dict):
                result = edge["condition"](state)
                current = edge["mapping"].get(result, "END")

        return state


def build_document_graph() -> DocumentGraph:
    graph = DocumentGraph()

    # Register nodes
    graph.add_node("classify",          node_classify)
    graph.add_node("extract_findings",  node_extract_findings)
    graph.add_node("generate_abstract", node_generate_abstract)
    graph.add_node("suggest_reviewers", node_suggest_reviewers)
    graph.add_node("quality_check",     node_quality_check)
    graph.add_node("revision_request",  node_revision_request)

    # Linear edges
    graph.add_edge("classify",          "extract_findings")
    graph.add_edge("extract_findings",  "generate_abstract")
    graph.add_edge("generate_abstract", "suggest_reviewers")
    graph.add_edge("suggest_reviewers", "quality_check")

    # Conditional edge: quality check routes to approve or revise
    graph.add_conditional_edge(
        "quality_check",
        condition=lambda s: "approve" if s.quality_score >= 7 else "revise",
        mapping={"approve": "END", "revise": "revision_request"},
    )
    graph.add_edge("revision_request", "END")

    return graph


# ── Sample research paper ─────────────────────────────────────────────────────
SAMPLE_PAPER = """
Title: Transformer-Based LLMs for Clinical Decision Support in Emergency Medicine

Abstract (draft): This study examines the application of large language models 
in emergency department triage. We evaluated GPT-4 and Claude-3 across 500 
patient cases from 3 hospitals.

Methods: Retrospective analysis of ED visits 2022-2024. Models were prompted 
with chief complaints, vitals, and initial labs. Primary outcome: triage 
accuracy vs. attending physician decisions.

Results: LLM triage accuracy reached 87.3% (95% CI: 84.1-90.5%) vs. 91.2% 
for physicians. LLMs reduced triage time by 4.2 minutes on average. 
False-negative rate for high-acuity cases: 2.1% for LLMs vs. 1.8% for physicians.

Conclusions: LLMs show promise as triage support tools but should not replace 
physician judgment in high-acuity presentations. Further prospective studies needed.

Keywords: artificial intelligence, emergency medicine, triage, clinical decision support
"""


def main():
    print_header("MODULE 9: LangGraph-Style Document Processing Graph")

    graph = build_document_graph()
    state = DocumentState(raw_text=SAMPLE_PAPER)

    print("\n📄 Processing: Clinical LLM Research Paper")
    print("─" * 60)

    final_state = graph.run("classify", state)

    print("\n" + "═" * 60)
    print("📊 FINAL RESULTS")
    print("═" * 60)
    print(f"Document Type:   {final_state.doc_type}")
    print(f"Domain:          {final_state.domain}")
    print(f"Key Findings:    {len(final_state.key_findings)}")
    for f in final_state.key_findings[:3]:
        print(f"  • {f[:80]}")
    print(f"\nAbstract ({len(final_state.abstract.split())} words):\n{final_state.abstract[:300]}...")
    print(f"\nReviewers:       {len(final_state.suggested_reviewers)}")
    for r in final_state.suggested_reviewers:
        print(f"  • {r}")
    print(f"\nQuality Score:   {final_state.quality_score}/10")
    print(f"Final Status:    {final_state.final_status}")
    print(f"Node Trace:      {' → '.join(final_state.node_trace)}")


if __name__ == "__main__":
    main()
