from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Vector store
VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store"
COLLECTION_NAME = "sherlock_holmes"

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# Load embedding model once
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Connect to existing Chroma vector store
chroma_client = chromadb.PersistentClient(
    path=str(VECTOR_STORE_PATH)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# Load all stored chunks for TF-IDF retrieval
all_data = collection.get(
    include=["documents", "metadatas"]
)

documents = all_data["documents"]
metadatas = all_data["metadatas"]


# Remove very short chunks
useful_indices = [
    i for i, text in enumerate(documents)
    if len(text.split()) >= 20
]

useful_documents = [
    documents[i]
    for i in useful_indices
]

useful_metadatas = [
    metadatas[i]
    for i in useful_indices
]


# Build TF-IDF index once
tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

tfidf_matrix = tfidf_vectorizer.fit_transform(
    useful_documents
)


def _min_max_normalize(scores):
    scores = np.array(scores)

    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        return np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


def retrieve_documents(query: str, top_k: int = 5, candidate_k: int = 20):
    """
    Retrieve relevant chunks using hybrid retrieval.

    Combines:
    - Semantic similarity from Chroma
    - Keyword similarity from TF-IDF
    """

    # -------------------------
    # Semantic retrieval
    # -------------------------

    query_embedding = embedding_model.encode([query])

    semantic_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=candidate_k
    )

    candidates = {}

    for i in range(len(semantic_results["documents"][0])):
        text = semantic_results["documents"][0][i]
        metadata = semantic_results["metadatas"][0][i]

        if len(text.split()) < 20:
            continue

        key = (
            metadata["page_number"],
            metadata["chunk_index"]
        )

        candidates[key] = {
            "text": text,
            "page_number": metadata["page_number"],
            "chunk_index": metadata["chunk_index"],
            "source": metadata["source"],
            "semantic_score": 1 - semantic_results["distances"][0][i],
            "keyword_score": 0.0
        }

    # -------------------------
    # TF-IDF retrieval
    # -------------------------

    query_vector = tfidf_vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarities.argsort()[-candidate_k:][::-1]

    for index in top_indices:
        metadata = useful_metadatas[index]
        text = useful_documents[index]

        key = (
            metadata["page_number"],
            metadata["chunk_index"]
        )

        if key not in candidates:
            candidates[key] = {
                "text": text,
                "page_number": metadata["page_number"],
                "chunk_index": metadata["chunk_index"],
                "source": metadata["source"],
                "semantic_score": 0.0,
                "keyword_score": similarities[index]
            }
        else:
            candidates[key]["keyword_score"] = similarities[index]

    # -------------------------
    # Weighted hybrid scoring
    # -------------------------

    results = list(candidates.values())

    semantic_scores = [
        result["semantic_score"]
        for result in results
    ]

    keyword_scores = [
        result["keyword_score"]
        for result in results
    ]

    semantic_normalized = _min_max_normalize(
        semantic_scores
    )

    keyword_normalized = _min_max_normalize(
        keyword_scores
    )

    for i, result in enumerate(results):
        result["hybrid_score"] = (
            0.4 * semantic_normalized[i]
            + 0.6 * keyword_normalized[i]
        )

    # Highest score first
    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return results[:top_k]