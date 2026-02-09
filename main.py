from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import time
import os
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

client = OpenAI()

app = FastAPI()

DOC_FOLDER = "docs"

documents = []
doc_embeddings = []


# ----------------------------
# LOAD DOCUMENTS
# ----------------------------

def load_documents():
    docs = []
    if not os.path.exists(DOC_FOLDER):
        return docs

    for filename in os.listdir(DOC_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(DOC_FOLDER, filename), "r", encoding="utf-8") as f:
                docs.append({
                    "id": len(docs),
                    "content": f.read(),
                    "metadata": {"source": filename}
                })
    return docs


def embed_texts(texts):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [np.array(e.embedding) for e in response.data]


# Load once at startup
documents = load_documents()

if documents:
    doc_embeddings = embed_texts([doc["content"] for doc in documents])


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
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ----------------------------
# RERANK FUNCTION
# ----------------------------

def rerank_results(query, results):
    for r in results:
        prompt = f"""
Query: "{query}"

Document: "{r['content']}"

Rate relevance from 0-10.
Respond only with the number.
"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        score = float(response.choices[0].message.content.strip())
        r["score"] = round(score / 10.0, 4)

    return sorted(results, key=lambda x: x["score"], reverse=True)


# ----------------------------
# SEARCH ENDPOINT
# ----------------------------

@app.post("/")
def search(request: SearchRequest):
    start = time.time()

    if not documents:
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 0,
                "totalDocs": 0
            }
        }

    # Embed query
    query_embedding = client.embeddings.create(
        model=EMBED_MODEL,
        input=request.query
    ).data[0].embedding

    query_embedding = np.array(query_embedding)

    # Compute similarities
    similarities = []
    for i, doc_embedding in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((i, score))

    # Normalize scores 0-1
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

    normalized.sort(key=lambda x: x[1], reverse=True)
    top_k = normalized[:request.k]

    results = []
    for idx, score in top_k:
        results.append({
            "id": documents[idx]["id"],
            "score": round(score, 4),
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"]
        })

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
