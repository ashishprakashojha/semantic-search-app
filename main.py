from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import time
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------

DOC_FOLDER = "docs"
LLM_MODEL = "gpt-4o-mini"  # Used only if rerank=True

app = FastAPI()
client = OpenAI()

# ----------------------------
# LOAD LOCAL EMBEDDING MODEL
# ----------------------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# LOAD DOCUMENTS
# ----------------------------

def load_documents():
    docs = []

    if not os.path.exists(DOC_FOLDER):
        print("⚠ 'docs' folder not found")
        return docs

    for filename in os.listdir(DOC_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(DOC_FOLDER, filename), "r", encoding="utf-8") as f:
                docs.append({
                    "id": len(docs),
                    "content": f.read(),
                    "metadata": {"source": filename}
                })

    print(f"Loaded {len(docs)} documents")
    return docs


documents = load_documents()
doc_embeddings = None  # Lazy initialization


# ----------------------------
# EMBEDDING FUNCTION (LOCAL)
# ----------------------------

def embed_texts(texts):
    return embedder.encode(texts)


# ----------------------------
# REQUEST MODEL
# ----------------------------

class SearchRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    rerankK: int = 5


# ----------------------------
# COSINE SIMILARITY
# ----------------------------

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ----------------------------
# OPTIONAL LLM RERANKING
# ----------------------------

def rerank_results(query, results):
    reranked = []

    for r in results:
        prompt = f"""
Query: "{query}"

Document: "{r['content']}"

Rate relevance from 0-10.
Respond only with the number.
"""

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            score = float(response.choices[0].message.content.strip())
            normalized = max(0.0, min(score / 10.0, 1.0))

        except Exception:
            # fallback to similarity score if API fails
            normalized = r["score"]

        r["score"] = round(normalized, 4)
        reranked.append(r)

    return sorted(reranked, key=lambda x: x["score"], reverse=True)


# ----------------------------
# SEARCH ENDPOINT
# ----------------------------

@app.post("/")
def search(request: SearchRequest):
    global doc_embeddings

    start = time.time()

    if len(documents) == 0:
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 0,
                "totalDocs": 0
            }
        }

    # Compute embeddings once
    if doc_embeddings is None:
        doc_embeddings = embed_texts([doc["content"] for doc in documents])

    # Embed query
    query_embedding = embed_texts([request.query])[0]

    # Compute cosine similarities
    similarities = []
    for i, doc_embedding in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((i, score))

    # Normalize similarity scores to 0-1
    scores_only = [s[1] for s in similarities]
    min_score = min(scores_only)
    max_score = max(scores_only)

    normalized = []
    for idx, score in similarities:
        if max_score - min_score == 0:
            norm = 0.0
        else:
            norm = (score - min_score) / (max_score - min_score)
        normalized.append((idx, norm))

    # Sort descending
    normalized.sort(key=lambda x: x[1], reverse=True)

    # Top K
    top_k = normalized[:request.k]

    results = []
    for idx, score in top_k:
        results.append({
            "id": documents[idx]["id"],
            "score": round(score, 4),
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"]
        })

    # Rerank if enabled
    if request.rerank:
        results = rerank_results(request.query, results)
        results = results[:request.rerankK]

    latency = int((time.time() - start) * 1000)

    return {
        "results": results,
        "reranked": request.rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
