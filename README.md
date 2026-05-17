# 🚀 AI Enterprise Mastery Project

> **Real-world AI engineering scenarios using Claude Sonnet & Claude Haiku**  
> Covering every skill companies actually hire for — not just model names.

---

## 📌 Project Overview

This project demonstrates **10 production-grade AI engineering concepts** through real-world scenarios using the **Anthropic API** (Claude Sonnet 4 + Claude Haiku). Each module is self-contained and solves a genuine business problem.

### Models Used
| Model | Use Case |
|---|---|
| `claude-sonnet-4-20250514` | Complex reasoning, agent orchestration, evaluation |
| `claude-haiku-4-5-20251001` | Fast inference, classification, lightweight tasks |

---

## 🗂️ Project Structure

```
ai-enterprise-project/
├── 1_rag_pipeline/              # Legal document Q&A with RAG
├── 2_fine_tuning/               # Prompt-based fine-tuning simulation
├── 3_prompt_engineering/        # Customer support prompt templates
├── 4_ai_agents/                 # Multi-step research agent
├── 5_vector_database/           # In-memory vector store + semantic search
├── 6_llm_evaluation/            # Automated LLM output evaluation
├── 7_inference_optimization/    # Haiku vs Sonnet cost/speed routing
├── 8_mcp_tool_calling/          # Tool calling: calculator, weather, DB
├── 9_langchain_langgraph/       # LangGraph-style agent graph (pure Python)
├── 10_workflow_orchestration/   # End-to-end AI pipeline orchestrator
├── utils/                       # Shared utilities
├── config/                      # Configuration and environment setup
├── docs/                        # Architecture diagrams and notes
└── requirements.txt             # Python dependencies
```

---

## 🎯 Real-World Scenarios

| # | Module | Business Scenario |
|---|--------|------------------|
| 1 | RAG Pipeline | Law firm needs to query 1000s of legal contracts |
| 2 | Fine-Tuning Sim | E-commerce brand voice adaptation |
| 3 | Prompt Engineering | SaaS customer support with 95% resolution rate |
| 4 | AI Agents | Autonomous market research agent |
| 5 | Vector Database | Semantic job matching platform |
| 6 | LLM Evaluation | QA pipeline for AI-generated medical summaries |
| 7 | Inference Optimization | Route queries by complexity to save 60% cost |
| 8 | MCP / Tool Calling | AI assistant with live tools (calculator, DB, API) |
| 9 | LangChain / LangGraph | Multi-node document processing graph |
| 10 | Workflow Orchestration | Full content pipeline: research → write → review |

---

## ⚡ Quick Start

```bash
# 1. Clone and navigate
cd ai-enterprise-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
export ANTHROPIC_API_KEY="your-key-here"
# OR create a .env file:
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# 4. Run any module
python 1_rag_pipeline/rag_pipeline.py
python 4_ai_agents/research_agent.py
python 10_workflow_orchestration/orchestrator.py
```

---

## 🧠 Key Concepts Demonstrated

- **RAG**: Chunking, embedding simulation, retrieval, grounded generation
- **Fine-Tuning**: System prompt adaptation to brand voice
- **Prompt Engineering**: Chain-of-thought, few-shot, structured output
- **AI Agents**: ReAct pattern, tool use, memory, planning
- **Vector DBs**: Cosine similarity search, document indexing
- **LLM Evaluation**: Criteria-based automated scoring
- **Inference Optimization**: Smart routing, caching, batching
- **Tool Calling**: Function definitions, tool result handling
- **LangGraph Pattern**: Node-edge graph, conditional routing
- **Orchestration**: Multi-stage pipelines, state management

---

## 📋 Requirements

- Python 3.9+
- Anthropic API Key
- See `requirements.txt` for packages

---

## 📄 License

MIT License — free for educational and commercial use.
