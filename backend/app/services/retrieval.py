import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings


class HybridRetriever:
    def __init__(self):
        self.embedding_model = None
        self.collection = None
        self.documents = None
        self.metadatas = None
        self.useful_documents = None
        self.useful_metadatas = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def load(self):
        """Load the embedding model, vector store, and TF-IDF index once."""

        # Load embedding model
        self.embedding_model = SentenceTransformer(
            settings.embedding_model
        )

        # Connect to existing Chroma vector store
        chroma_client = chromadb.PersistentClient(
            path=str(settings.vector_store_path)
        )

        self.collection = chroma_client.get_collection(
            name=settings.collection_name
        )

        # Load stored chunks for TF-IDF retrieval
        all_data = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = all_data["documents"]
        self.metadatas = all_data["metadatas"]

        # Remove very short chunks
        useful_indices = [
            i
            for i, text in enumerate(self.documents)
            if len(text.split()) >= 20
        ]

        self.useful_documents = [
            self.documents[i]
            for i in useful_indices
        ]

        self.useful_metadatas = [
            self.metadatas[i]
            for i in useful_indices
        ]

        # Build TF-IDF index once
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.useful_documents
        )

    @staticmethod
    def _min_max_normalize(scores):
        scores = np.array(scores)

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.zeros_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20
    ):
        """
        Retrieve relevant chunks using hybrid retrieval.

        Combines:
        - Semantic similarity from Chroma
        - Keyword similarity from TF-IDF
        """

        # -------------------------
        # Semantic retrieval
        # -------------------------

        query_embedding = self.embedding_model.encode([query])

        semantic_results = self.collection.query(
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
                "semantic_score": (
                    1 - semantic_results["distances"][0][i]
                ),
                "keyword_score": 0.0
            }

        # -------------------------
        # TF-IDF retrieval
        # -------------------------

        query_vector = self.tfidf_vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.tfidf_matrix
        ).flatten()

        top_indices = similarities.argsort()[-candidate_k:][::-1]

        for index in top_indices:

            metadata = self.useful_metadatas[index]
            text = self.useful_documents[index]

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

        semantic_normalized = self._min_max_normalize(
            semantic_scores
        )

        keyword_normalized = self._min_max_normalize(
            keyword_scores
        )

        for i, result in enumerate(results):

            result["hybrid_score"] = (
                settings.semantic_weight
                * semantic_normalized[i]
                + settings.keyword_weight
                * keyword_normalized[i]
            )

        # Highest score first
        results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return results[:top_k]


# Create the retriever object.
# The heavy resources are NOT loaded here.
retriever = HybridRetriever()


def initialize_retriever():
    """Initialize the retriever during application startup."""
    retriever.load()


def retrieve_documents(
    query: str,
    top_k: int = 5,
    candidate_k: int = 20
):
    """Public function used by the API route."""

    return retriever.retrieve(
        query=query,
        top_k=top_k,
        candidate_k=candidate_k
    )