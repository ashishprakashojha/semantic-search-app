from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

# ----------------------------
# CORS (IMPORTANT FOR RENDER)
# ----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# LOAD DOCUMENTS
# ----------------------------

DOC_FOLDER = "docs"

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

documents = load_documents()
doc_embeddings = []

# ----------------------------
# EMBEDDING FUNCTION
# ----------------------------

def embed_texts(texts):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [np.array(e.embedding) for e in response.data]

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
    scores = []

    for r in results:
        prompt = f"""
Query: "{query}"

Document: "{r['content']}"

Rate the relevance from 0-10.
Respond ONLY with the number.
"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            score = float(response.choices[0].message.content.strip())
        except:
            score = 5.0

        scores.append(score / 10.0)

    for i in range(len(results)):
        results[i]["score"] = round(scores[i], 4)

    return sorted(results, key=lambda x: x["score"], reverse=True)

# ----------------------------
# HEALTH CHECK (IMPORTANT)
# ----------------------------

@app.get("/")
def health():
    return {"status": "running", "totalDocs": len(documents)}

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

    global doc_embeddings
    if not doc_embeddings:
        doc_embeddings = embed_texts([doc["content"] for doc in documents])

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

    # Normalize scores
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
