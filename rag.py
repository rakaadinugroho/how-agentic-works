#!/usr/bin/env python3
"""
Simple RAG (Retrieval Augmented Generation) system.
Uses bge-m3 via Ollama for embeddings and numpy for cosine similarity search.
"""
import numpy as np
from ollama_client import embed


class RAG:
    def __init__(self):
        self.documents = []
        self.vectors = None

    def add(self, texts):
        """Add documents to the knowledge base. Embeds each text and stores it."""
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            if text.strip():
                self.documents.append(text)
        # Re-embed all documents
        self._reindex()

    def _reindex(self):
        """Re-embed all stored documents."""
        if not self.documents:
            self.vectors = np.array([])
            return
        embeddings = embed(self.documents)
        self.vectors = np.array(embeddings)

    def search(self, query, top_k=3):
        """
        Search the knowledge base for documents most similar to the query.
        Returns list of (document, similarity_score) tuples.
        """
        if not self.documents or self.vectors is None or len(self.vectors) == 0:
            return []

        query_vec = np.array(embed(query)[0])

        # Cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return []

        similarities = np.dot(self.vectors, query_vec) / (norms * query_norm + 1e-10)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.documents[idx], float(similarities[idx])))

        return results

    def query(self, query, top_k=3):
        """Return a formatted string with search results."""
        results = self.search(query, top_k=top_k)
        if not results:
            return "Knowledge base is empty or no relevant documents found."

        output = f"RAG results for '{query}':\n"
        for i, (doc, score) in enumerate(results):
            snippet = doc[:300] + ("..." if len(doc) > 300 else "")
            output += f"\n--- Result {i+1} (score: {score:.3f}) ---\n{snippet}\n"
        return output

    def size(self):
        return len(self.documents)
