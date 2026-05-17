"""
5_vector_database/semantic_job_matcher.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL-WORLD SCENARIO: Semantic Job Matching Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Problem : LinkedIn-style platforms use keyword matching — a
          "ML Engineer" with Python misses jobs titled "AI
          Developer" requiring the same skills.
Solution: Semantic vector search — embed job descriptions and
          resumes, find matches by meaning not keywords.

This module: in-memory vector store (no external DB needed).
Production: plug in Pinecone / Weaviate / pgvector.

Model used: claude-haiku-4-5-20251001 (fast embedding generation)
"""

import sys, os, math, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import get_client, print_header
from config.settings import HAIKU, SONNET

# ── Sample job database ───────────────────────────────────────────────────────
JOBS = [
    {
        "id": "job_001",
        "title": "Senior ML Engineer",
        "description": "Build and deploy machine learning models at scale. Python, PyTorch, MLflow. Design data pipelines. Work with cross-functional teams on AI products.",
    },
    {
        "id": "job_002",
        "title": "AI Product Developer",
        "description": "Develop intelligent applications using LLMs and generative AI. Experience with OpenAI, Claude APIs. Python. Integrate AI into user-facing products.",
    },
    {
        "id": "job_003",
        "title": "Data Science Lead",
        "description": "Lead a team of data scientists. Statistical modeling, A/B testing, business insights. Python, R, SQL. Communicate findings to executives.",
    },
    {
        "id": "job_004",
        "title": "Backend Software Engineer",
        "description": "Build scalable REST APIs with Python and FastAPI. PostgreSQL, Redis, Docker. Microservices architecture. No ML experience required.",
    },
    {
        "id": "job_005",
        "title": "MLOps Engineer",
        "description": "Build ML infrastructure, CI/CD for models, monitoring and observability. Kubernetes, Docker, MLflow, Kubeflow. Python. Bridge ML and DevOps.",
    },
]

# ── Resume to match ───────────────────────────────────────────────────────────
CANDIDATE_RESUME = """
John Doe — Machine Learning Engineer
5 years experience building production AI systems.
Skills: Python, TensorFlow, PyTorch, Docker, Kubernetes, MLflow
Built recommendation systems serving 10M users.
Experience deploying models with FastAPI and monitoring with Prometheus.
Interested in LLM applications and generative AI products.
"""


# ── In-memory vector store ────────────────────────────────────────────────────
class VectorStore:
    def __init__(self):
        self.documents: list[dict] = []
        self.vectors: list[dict] = []  # TF vectors

    def _vectorize(self, text: str) -> dict[str, float]:
        """TF-IDF-like sparse vector (production: use real embeddings)."""
        words = re.findall(r"\w+", text.lower())
        freq: dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        total = len(words)
        return {w: count / total for w, count in freq.items()}

    def add(self, doc: dict) -> None:
        self.documents.append(doc)
        combined = f"{doc.get('title', '')} {doc.get('description', '')}"
        self.vectors.append(self._vectorize(combined))

    def search(self, query: str, top_k: int = 3) -> list[tuple[dict, float]]:
        q_vec = self._vectorize(query)
        results = []
        for doc, d_vec in zip(self.documents, self.vectors):
            common = set(q_vec) & set(d_vec)
            if not common:
                results.append((doc, 0.0))
                continue
            dot = sum(q_vec[w] * d_vec[w] for w in common)
            norm_q = math.sqrt(sum(v**2 for v in q_vec.values()))
            norm_d = math.sqrt(sum(v**2 for v in d_vec.values()))
            score = dot / (norm_q * norm_d) if norm_q and norm_d else 0.0
            results.append((doc, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ── AI-enhanced match explanation ────────────────────────────────────────────
def explain_match(resume: str, job: dict, score: float) -> str:
    client = get_client()
    response = client.messages.create(
        model=HAIKU,
        max_tokens=150,
        system="You are a recruitment AI. Explain in 2-3 sentences why this candidate matches this job. Be specific about skills alignment.",
        messages=[{
            "role": "user",
            "content": f"Resume: {resume}\n\nJob: {job['title']}: {job['description']}\n\nSimilarity score: {score:.2f}"
        }],
    )
    return response.content[0].text


def main():
    print_header("MODULE 5: Vector Database — Semantic Job Matcher")

    # Build vector store
    store = VectorStore()
    for job in JOBS:
        store.add(job)
    print(f"📦 Indexed {len(JOBS)} job postings\n")

    print(f"👤 Candidate Resume:\n{CANDIDATE_RESUME}")
    print("\n🔍 Finding top matches...\n")

    matches = store.search(CANDIDATE_RESUME, top_k=3)

    for rank, (job, score) in enumerate(matches, 1):
        print(f"#{rank} {job['title']} (score: {score:.3f})")
        explanation = explain_match(CANDIDATE_RESUME, job, score)
        print(f"   💡 {explanation}")
        print()


if __name__ == "__main__":
    main()
