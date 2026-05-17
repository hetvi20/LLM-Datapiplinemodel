"""
4_ai_agents/research_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL-WORLD SCENARIO: Autonomous Market Research Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem : A VC firm needs to research startup markets before
          investment meetings. Manual research takes 2 days.
Solution: A ReAct agent with tools: search, summarize, compare,
          and compile final report. Agent self-directs the
          research workflow.

Pattern: ReAct (Reason + Act) — agent reasons about next step,
         calls a tool, observes output, repeats until done.

Model used: claude-sonnet-4-20250514 (complex multi-step reasoning)
"""

import sys, os, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import get_client, print_header
from config.settings import SONNET

# ── Tool definitions for the agent ──────────────────────────────────────────
TOOLS = [
    {
        "name": "search_market_data",
        "description": "Search for market size, growth rate, and key players in a specific industry or sector.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The market or industry to research"},
                "focus": {"type": "string", "description": "What to focus on: 'size' | 'players' | 'trends' | 'all'"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "analyze_competitor",
        "description": "Get detailed analysis of a specific company including funding, product, and positioning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Name of the company to analyze"},
            },
            "required": ["company_name"],
        },
    },
    {
        "name": "calculate_market_score",
        "description": "Calculate an investment attractiveness score for a market based on size, growth, and competition.",
        "input_schema": {
            "type": "object",
            "properties": {
                "market_size_bn": {"type": "number", "description": "Market size in billions USD"},
                "growth_rate_pct": {"type": "number", "description": "Annual growth rate percentage"},
                "competitor_count": {"type": "integer", "description": "Number of major competitors"},
            },
            "required": ["market_size_bn", "growth_rate_pct", "competitor_count"],
        },
    },
    {
        "name": "compile_report",
        "description": "Compile all gathered research into a final investment memo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "market": {"type": "string"},
                "findings": {"type": "string", "description": "All research findings gathered so far"},
                "recommendation": {"type": "string", "enum": ["invest", "monitor", "pass"]},
            },
            "required": ["market", "findings", "recommendation"],
        },
    },
]

# ── Simulated tool execution ─────────────────────────────────────────────────
def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Simulate tool responses (in production: call real APIs)."""

    if tool_name == "search_market_data":
        query = tool_input["query"]
        return json.dumps({
            "market": query,
            "size_2024": "$12.4B",
            "projected_2029": "$38.7B",
            "cagr": "25.6%",
            "key_players": ["Anthropic", "OpenAI", "Cohere", "Mistral AI"],
            "key_trends": ["Enterprise adoption", "Edge AI", "Multimodal models"],
            "barriers_to_entry": "High compute costs, talent scarcity",
        })

    elif tool_name == "analyze_competitor":
        company = tool_input["company_name"]
        return json.dumps({
            "company": company,
            "funding": "$7.3B total raised",
            "valuation": "$18B (Series E)",
            "product": "Foundation models + API + consumer apps",
            "moat": "RLHF safety leadership, enterprise trust",
            "weakness": "High inference costs, competition pressure",
            "employees": "~900",
        })

    elif tool_name == "calculate_market_score":
        size = tool_input["market_size_bn"]
        growth = tool_input["growth_rate_pct"]
        competitors = tool_input["competitor_count"]
        # Simple scoring formula
        score = min(100, (size * 2) + (growth * 1.5) - (competitors * 3))
        tier = "A" if score > 70 else "B" if score > 40 else "C"
        return json.dumps({
            "score": round(score, 1),
            "tier": tier,
            "interpretation": f"Market is {'highly' if tier=='A' else 'moderately'} attractive for investment",
        })

    elif tool_name == "compile_report":
        return f"""
╔══════════════════════════════════════════════════════════╗
║           VC INVESTMENT MEMO — {tool_input['market'].upper()[:20]}
╚══════════════════════════════════════════════════════════╝

RECOMMENDATION: {tool_input['recommendation'].upper()}
Generated: {datetime.now().strftime('%Y-%m-%d')}

FINDINGS:
{tool_input['findings']}

This memo was compiled by the AI Research Agent.
All data should be verified with primary sources before investment decisions.
"""

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ── ReAct Agent Loop ─────────────────────────────────────────────────────────
def run_research_agent(research_goal: str, max_steps: int = 10) -> str:
    client = get_client()
    messages = [{"role": "user", "content": research_goal}]

    system = """You are an autonomous market research agent for a VC firm.
Your goal: thoroughly research the given market and produce an investment memo.
Use your tools strategically: search first, analyze competitors, score the market, then compile.
Be systematic. Each tool call should advance toward a complete research report."""

    print(f"\n🤖 Agent starting research: {research_goal}\n")

    for step in range(max_steps):
        response = client.messages.create(
            model=SONNET,
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        # Check stop condition
        if response.stop_reason == "end_turn":
            final_text = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            print(f"\n✅ Agent completed after {step + 1} steps.")
            return final_text

        # Process tool calls
        tool_calls_made = False
        tool_results = []

        for block in response.content:
            if block.type == "text" and block.text:
                print(f"💭 Agent thinking: {block.text[:120]}...")
            elif block.type == "tool_use":
                tool_calls_made = True
                print(f"🔧 Calling tool: {block.name}({json.dumps(block.input)[:80]}...)")
                result = execute_tool(block.name, block.input)
                print(f"   → Result: {result[:100]}...")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if not tool_calls_made:
            break

        # Update message history with assistant response + tool results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "Agent reached max steps without completing."


def main():
    print_header("MODULE 4: AI Agents — VC Market Research Agent")
    result = run_research_agent(
        "Research the AI/LLM API market for a potential $50M investment. "
        "Analyze key competitors, calculate market attractiveness, and produce a memo."
    )
    if result:
        print(f"\n📋 Final output:\n{result}")


if __name__ == "__main__":
    main()
