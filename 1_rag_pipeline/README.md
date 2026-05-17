# Module 1: RAG Pipeline

## Scenario
A law firm needs associates to instantly answer client questions about contracts without reading thousands of pages manually.

## What This Demonstrates
- Document chunking with overlap
- Semantic retrieval (keyword-based; swap for real embeddings in production)
- Grounded answer generation with citations
- Hallucination prevention via context-only instructions

## How to Run
```bash
python 1_rag_pipeline/rag_pipeline.py
```

## Production Upgrade Path
- Replace `pseudo_embed` with `voyage-3` or `text-embedding-3-large`
- Store chunks in Pinecone / Weaviate / pgvector
- Add re-ranking with Cohere Rerank
