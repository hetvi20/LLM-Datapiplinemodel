# Architecture Overview

## System Design

```
ai-enterprise-project/
│
├── Shared Infrastructure
│   ├── config/settings.py       Model constants, API config
│   └── utils/helpers.py         Client factory, cosine similarity, logging
│
├── Module 1: RAG Pipeline
│   Query → Chunk → Embed → Retrieve → Generate (grounded)
│   Model: Sonnet (accurate citations)
│
├── Module 2: Fine-Tuning Simulation  
│   System prompt + few-shot examples → brand-consistent output
│   Model: Haiku (generic) vs Sonnet+prompt (branded)
│
├── Module 3: Prompt Engineering
│   Triage (structured JSON) → Routing → Resolution (chain-of-thought)
│   Model: Haiku (classify) + Sonnet (resolve)
│
├── Module 4: AI Agents
│   ReAct loop: Reason → Act (tool) → Observe → Repeat → Report
│   Model: Sonnet (complex multi-step reasoning)
│
├── Module 5: Vector Database
│   Index (embed) → Query (embed) → Rank (cosine) → Explain
│   Model: Haiku (explanation)
│
├── Module 6: LLM Evaluation
│   Generate (Haiku) → Evaluate (Sonnet judge) → Verdict → Approve/Reject
│   Pattern: Generator + Judge separation
│
├── Module 7: Inference Optimization
│   Classify complexity → Route (Haiku/Sonnet) → Cache → Track cost
│   ~60% cost reduction vs all-Sonnet
│
├── Module 8: Tool Calling
│   Define tools → Model reasons → Call tool → Handle result → Respond
│   Multi-step tool chains, agentic loops
│
├── Module 9: LangGraph-Style Graph
│   Nodes (AI steps) + Edges (conditional routing) + Shared state
│   Classify → Extract → Generate → Review → Approve/Revise
│
└── Module 10: Workflow Orchestration
    Chains all patterns: Research → Outline → Draft → SEO → Review → Format
    Full pipeline with state management and error handling
```

## Model Selection Strategy

| Complexity | Model | Use Case |
|---|---|---|
| Low | Haiku | Classification, extraction, short answers, triage |
| Medium | Haiku | Summarization, SEO, formatting, outlines |
| High | Sonnet | Drafting, agent reasoning, evaluation, complex analysis |

## Key Patterns

### ReAct (Module 4, 8)
```
Reason: "I need to check the stock price first"
Act: get_stock_price(ticker="AAPL")
Observe: {"price": 189.45, ...}
Reason: "Now I can calculate portfolio value"
Act: calculate_portfolio_value(holdings=[...])
```

### LLM-as-Judge (Module 6)
```
Generator (Haiku) → output → Judge (Sonnet) → score + verdict
```

### Smart Routing (Module 7)
```
Query → complexity_classifier → [simple/medium → Haiku] | [complex → Sonnet]
                              → cache_check → [hit → return] | [miss → API call]
```

### Graph State Machine (Module 9)
```
State → Node_A → (modify state) → conditional_edge → Node_B | Node_C
```
