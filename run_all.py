"""
run_all.py — Run all 10 AI engineering modules in sequence
Usage: python run_all.py
       python run_all.py --module 4    (run specific module)
       python run_all.py --list        (list modules)
"""

import sys
import os
import importlib.util
import argparse

MODULES = [
    (1,  "1_rag_pipeline/rag_pipeline.py",                   "RAG Pipeline — Law Firm Contract Q&A"),
    (2,  "2_fine_tuning/brand_voice_adapter.py",              "Fine-Tuning — Brand Voice Adaptation"),
    (3,  "3_prompt_engineering/customer_support.py",          "Prompt Engineering — Customer Support"),
    (4,  "4_ai_agents/research_agent.py",                     "AI Agents — Market Research Agent"),
    (5,  "5_vector_database/semantic_job_matcher.py",          "Vector Database — Job Matching"),
    (6,  "6_llm_evaluation/medical_summary_evaluator.py",     "LLM Evaluation — Medical QA Pipeline"),
    (7,  "7_inference_optimization/smart_router.py",          "Inference Optimization — Smart Router"),
    (8,  "8_mcp_tool_calling/ai_assistant_with_tools.py",     "Tool Calling — Financial AI Assistant"),
    (9,  "9_langchain_langgraph/document_processing_graph.py","LangGraph — Document Processing Graph"),
    (10, "10_workflow_orchestration/orchestrator.py",         "Workflow Orchestration — Content Pipeline"),
]


def load_and_run(path: str) -> None:
    abs_path = os.path.join(os.path.dirname(__file__), path)
    spec = importlib.util.spec_from_file_location("module", abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "main"):
        mod.main()


def main():
    parser = argparse.ArgumentParser(description="AI Enterprise Project Runner")
    parser.add_argument("--module", type=int, help="Run specific module number (1-10)")
    parser.add_argument("--list", action="store_true", help="List all modules")
    args = parser.parse_args()

    if args.list:
        print("\n📦 Available Modules:")
        for num, path, desc in MODULES:
            print(f"  {num:2d}. {desc}")
        return

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set.")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    if args.module:
        module_map = {num: (path, desc) for num, path, desc in MODULES}
        if args.module not in module_map:
            print(f"❌ Module {args.module} not found. Use --list to see options.")
            sys.exit(1)
        path, desc = module_map[args.module]
        print(f"\n🚀 Running Module {args.module}: {desc}\n")
        load_and_run(path)
    else:
        print("\n🚀 Running all 10 AI Engineering Modules\n")
        for num, path, desc in MODULES:
            print(f"\n{'=' * 70}")
            print(f"MODULE {num}: {desc}")
            print(f"{'=' * 70}")
            try:
                load_and_run(path)
            except KeyboardInterrupt:
                print("\n⏹ Interrupted by user.")
                break
            except Exception as e:
                print(f"❌ Module {num} failed: {e}")
                print("   Continuing to next module...")

        print("\n\n✅ All modules completed!")


if __name__ == "__main__":
    main()
