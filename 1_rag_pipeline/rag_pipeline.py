"""
1_rag_pipeline/rag_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL-WORLD SCENARIO: Law Firm Contract Q&A System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem : Associates spend hours reading contracts to answer
          client questions like "What's the termination clause?"
Solution: RAG pipeline that chunks contracts, retrieves relevant
          sections, and answers grounded in actual document text.

Model used: claude-sonnet-4-20250514 (accurate, citation-aware)
"""

import sys, os, math, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import get_client, print_header, print_result
from config.settings import SONNET

# ── Simulated contract corpus ────────────────────────────────────────────────
CONTRACTS = [
    {
        "id": "contract_001",
        "title": "Acme Corp Software License Agreement",
        "text": """
        TERMINATION CLAUSE: Either party may terminate this Agreement upon 30 days
        written notice. Acme Corp may terminate immediately if the licensee breaches
        any payment obligation or intellectual property clause.

        PAYMENT TERMS: Licensee shall pay $50,000 annually, due on the first day of
        each contract year. Late payments incur 1.5% monthly interest.

        INTELLECTUAL PROPERTY: All software, code, and documentation remain the
        exclusive property of Acme Corp. Licensee receives a non-exclusive,
        non-transferable license.

        LIMITATION OF LIABILITY: In no event shall Acme Corp's liability exceed the
        fees paid in the preceding 12 months.
        """,
    },
    {
        "id": "contract_002",
        "title": "GlobalTech SaaS Service Agreement",
        "text": """
        DATA PRIVACY: GlobalTech shall process personal data only per GDPR and CCPA
        requirements. Data will not be sold or shared with third parties without
        explicit consent.

        UPTIME GUARANTEE: GlobalTech guarantees 99.9% monthly uptime. Downtime
        exceeding this threshold entitles customer to service credits equal to
        10x the downtime duration in service fees.

        TERMINATION: Customer may cancel with 60 days notice. No refunds are issued
        for the remaining contract period unless GlobalTech breaches the SLA.

        RENEWAL: Contract auto-renews annually unless cancelled 90 days before
        expiry.
        """,
    },
    {
        "id": "contract_003",
        "title": "StartupXYZ Employment Agreement",
        "text": """
        COMPENSATION: Employee shall receive $120,000 base salary, paid bi-weekly,
        plus equity vesting over 4 years with a 1-year cliff.

        NON-COMPETE: Employee agrees not to work for direct competitors within
        the United States for 12 months post-termination.

        CONFIDENTIALITY: Employee must protect all trade secrets and proprietary
        information indefinitely, even after employment ends.

        TERMINATION: StartupXYZ may terminate at-will. Employee may resign with
        2 weeks notice. Severance of 1 month salary applies if terminated without cause.
        """,
    },
]


# ── Step 1: Chunking ─────────────────────────────────────────────────────────
def chunk_document(doc: dict, chunk_size: int = 150) -> list[dict]:
    """Split document text into overlapping chunks."""
    words = doc["text"].split()
    chunks = []
    step = chunk_size - 30  # 30-word overlap

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) < 20:
            break
        chunks.append({
            "doc_id":   doc["id"],
            "doc_title": doc["title"],
            "chunk_id": f"{doc['id']}_chunk_{len(chunks)}",
            "text":     " ".join(chunk_words),
        })
    return chunks


# ── Step 2: Pseudo-embedding via keyword overlap (no external API needed) ────
def pseudo_embed(text: str) -> dict[str, int]:
    """
    Simple TF representation for retrieval demo.
    In production: use a real embedding model.
    """
    words = re.findall(r"\w+", text.lower())
    return {w: words.count(w) for w in set(words)}


def pseudo_similarity(q_vec: dict, d_vec: dict) -> float:
    """Cosine-like similarity between two keyword dicts."""
    common = set(q_vec) & set(d_vec)
    if not common:
        return 0.0
    dot = sum(q_vec[w] * d_vec[w] for w in common)
    norm_q = math.sqrt(sum(v**2 for v in q_vec.values()))
    norm_d = math.sqrt(sum(v**2 for v in d_vec.values()))
    return dot / (norm_q * norm_d) if norm_q and norm_d else 0.0


# ── Step 3: Retrieval ────────────────────────────────────────────────────────
def retrieve(query: str, index: list[dict], top_k: int = 3) -> list[dict]:
    q_vec = pseudo_embed(query)
    scored = [
        (chunk, pseudo_similarity(q_vec, pseudo_embed(chunk["text"])))
        for chunk in index
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored[:top_k] if score > 0]


# ── Step 4: Augmented Generation ────────────────────────────────────────────
def answer_question(query: str, retrieved_chunks: list[dict]) -> str:
    client = get_client()

    context = "\n\n---\n\n".join(
        f"Source: {c['doc_title']}\n{c['text']}" for c in retrieved_chunks
    )

    system = """You are a legal AI assistant for a law firm. Answer questions using ONLY
the contract excerpts provided. Always cite the source document. If the answer
is not in the excerpts, say 'This information is not covered in the retrieved contracts.'"""

    prompt = f"""Contract Excerpts:
{context}

Question: {query}

Provide a concise, accurate answer citing the relevant contract."""

    response = client.messages.create(
        model=SONNET,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ── Main demo ────────────────────────────────────────────────────────────────
def main():
    print_header("MODULE 1: RAG Pipeline — Law Firm Contract Q&A")

    # Build index
    print("📄 Indexing contract corpus...")
    index = []
    for doc in CONTRACTS:
        index.extend(chunk_document(doc))
    print(f"   ✅ {len(CONTRACTS)} contracts → {len(index)} chunks indexed\n")

    queries = [
        "What are the termination conditions and notice period?",
        "What happens if there is a data breach under GDPR?",
        "What is the non-compete duration for employees?",
        "What payment terms apply and what are late payment penalties?",
    ]

    for q in queries:
        print(f"❓ Question: {q}")
        chunks = retrieve(q, index)
        answer = answer_question(q, chunks)
        print(f"💼 Answer:\n{answer}")
        print("-" * 50)


if __name__ == "__main__":
    main()
