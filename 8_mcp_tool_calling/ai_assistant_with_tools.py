"""
8_mcp_tool_calling/ai_assistant_with_tools.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL-WORLD SCENARIO: AI Business Assistant with Live Tools
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem : A financial services firm wants an AI assistant that
          can actually DO things: look up stock prices, calculate
          portfolio metrics, query client databases, send alerts.
Solution: Tool calling (function calling) — define tools the
          model can invoke. Handle multi-step tool chains.

Demonstrates:
  - Tool definition schema
  - Multi-tool conversations
  - Tool result handling
  - Chained tool calls (tool result → next tool)

Model: claude-sonnet-4-20250514 (best tool use reasoning)
"""

import sys, os, json, math
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import get_client, print_header
from config.settings import SONNET

# ── Tool definitions ──────────────────────────────────────────────────────────
FINANCIAL_TOOLS = [
    {
        "name": "get_stock_price",
        "description": "Get current stock price and daily change for a given ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL, MSFT)"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_portfolio_value",
        "description": "Calculate total portfolio value and P&L given holdings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "shares": {"type": "number"},
                            "avg_cost": {"type": "number"},
                        },
                    },
                    "description": "List of stock holdings with shares and average cost",
                },
            },
            "required": ["holdings"],
        },
    },
    {
        "name": "get_client_profile",
        "description": "Retrieve client profile from the CRM database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "client_id": {"type": "string", "description": "Client ID in the system"},
            },
            "required": ["client_id"],
        },
    },
    {
        "name": "send_alert",
        "description": "Send an alert or notification to a client or advisor.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "message": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
            },
            "required": ["recipient", "message", "priority"],
        },
    },
    {
        "name": "calculate_risk_metrics",
        "description": "Calculate portfolio risk metrics: volatility, Sharpe ratio, max drawdown.",
        "input_schema": {
            "type": "object",
            "properties": {
                "portfolio_value": {"type": "number"},
                "risk_tolerance": {"type": "string", "enum": ["conservative", "moderate", "aggressive"]},
            },
            "required": ["portfolio_value", "risk_tolerance"],
        },
    },
]


# ── Simulated tool execution ──────────────────────────────────────────────────
SIMULATED_PRICES = {
    "AAPL": {"price": 189.45, "change": +2.3, "change_pct": +1.23},
    "MSFT": {"price": 378.90, "change": -1.5, "change_pct": -0.39},
    "GOOGL": {"price": 152.70, "change": +4.1, "change_pct": +2.76},
    "TSLA": {"price": 245.30, "change": -8.2, "change_pct": -3.23},
    "NVDA": {"price": 487.65, "change": +12.4, "change_pct": +2.61},
}

SIMULATED_CLIENTS = {
    "C001": {
        "name": "Margaret Chen",
        "risk_profile": "moderate",
        "aum": 485000,
        "advisor": "john.smith@wealth.com",
        "holdings": [
            {"ticker": "AAPL", "shares": 100, "avg_cost": 155.00},
            {"ticker": "MSFT", "shares": 50,  "avg_cost": 310.00},
            {"ticker": "NVDA", "shares": 25,  "avg_cost": 380.00},
        ],
    }
}


def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "get_stock_price":
        ticker = tool_input["ticker"].upper()
        if ticker in SIMULATED_PRICES:
            data = SIMULATED_PRICES[ticker]
            return json.dumps({
                "ticker": ticker,
                "price": data["price"],
                "change": data["change"],
                "change_pct": f"{data['change_pct']:+.2f}%",
                "timestamp": datetime.now().isoformat(),
            })
        return json.dumps({"error": f"Ticker {ticker} not found"})

    elif tool_name == "calculate_portfolio_value":
        holdings = tool_input["holdings"]
        total_value = 0
        total_cost = 0
        details = []
        for h in holdings:
            ticker = h["ticker"].upper()
            price = SIMULATED_PRICES.get(ticker, {}).get("price", h.get("avg_cost", 100))
            value = price * h["shares"]
            cost = h["avg_cost"] * h["shares"]
            total_value += value
            total_cost += cost
            details.append({"ticker": ticker, "value": round(value, 2), "pnl": round(value - cost, 2)})

        return json.dumps({
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_pnl": round(total_value - total_cost, 2),
            "pnl_pct": round((total_value - total_cost) / total_cost * 100, 2),
            "holdings_detail": details,
        })

    elif tool_name == "get_client_profile":
        client_id = tool_input["client_id"]
        if client_id in SIMULATED_CLIENTS:
            return json.dumps(SIMULATED_CLIENTS[client_id])
        return json.dumps({"error": f"Client {client_id} not found"})

    elif tool_name == "send_alert":
        return json.dumps({
            "status": "sent",
            "recipient": tool_input["recipient"],
            "message_preview": tool_input["message"][:50],
            "priority": tool_input["priority"],
            "timestamp": datetime.now().isoformat(),
        })

    elif tool_name == "calculate_risk_metrics":
        value = tool_input["portfolio_value"]
        risk = tool_input["risk_tolerance"]
        volatility = {"conservative": 8.5, "moderate": 14.2, "aggressive": 23.8}[risk]
        sharpe = {"conservative": 1.8, "moderate": 1.3, "aggressive": 0.9}[risk]
        return json.dumps({
            "portfolio_value": value,
            "annualized_volatility": f"{volatility}%",
            "sharpe_ratio": sharpe,
            "max_drawdown": f"{volatility * 1.5:.1f}%",
            "var_95": round(value * volatility / 100 * 1.65, 2),
            "risk_level": risk,
        })

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ── Agentic tool-calling loop ─────────────────────────────────────────────────
def run_assistant(user_request: str) -> str:
    client = get_client()
    messages = [{"role": "user", "content": user_request}]

    system = """You are an AI financial advisor assistant for WealthTech Pro.
You have access to real-time market data, client portfolios, and communication tools.
Answer client and advisor requests by using your tools systematically.
Always be precise with numbers and professional in tone."""

    for _ in range(8):  # Max tool rounds
        response = client.messages.create(
            model=SONNET,
            max_tokens=1024,
            system=system,
            tools=FINANCIAL_TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if hasattr(b, "text")), "")

        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text:
                print(f"   🤔 {block.text[:100]}...")
            elif block.type == "tool_use":
                print(f"   🔧 Tool: {block.name}({list(block.input.keys())})")
                result = execute_tool(block.name, block.input)
                print(f"   ✅ Got: {result[:80]}...")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if not tool_results:
            break

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "Request completed."


def main():
    print_header("MODULE 8: MCP / Tool Calling — Financial AI Assistant")

    scenarios = [
        "What is the current price of NVIDIA stock?",
        "Pull up client C001's profile and calculate their current portfolio value. Let me know the total P&L.",
        "Client C001's portfolio has dropped. Calculate their risk metrics and send an urgent alert to their advisor.",
    ]

    for i, request in enumerate(scenarios, 1):
        print(f"\n📨 Request #{i}: {request}")
        print("─" * 60)
        response = run_assistant(request)
        print(f"\n💬 Final Response:\n{response}")
        print("═" * 60)


if __name__ == "__main__":
    main()
