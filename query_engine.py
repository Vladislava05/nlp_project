import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def tokenize(text):
    return [word.lower() for word in word_tokenize(text) if word.isalnum()]

class RAGEngine:
    def __init__(self, index_dir="index", model_name="all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.model = SentenceTransformer(model_name)

        self.index = faiss.read_index(os.path.join(index_dir, "faiss_index.bin"))

        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.meta = data["meta"]
            self.tokenized_chunks = data.get("tokenized_chunks")

        if self.tokenized_chunks:
            self.bm25 = BM25Okapi(self.tokenized_chunks)
        else:
            self.tokenized_chunks = [tokenize(chunk) for chunk in self.chunks]
            self.bm25 = BM25Okapi(self.tokenized_chunks)

    def dense_search(self, query, top_k=10):
        """Поиск по FAISS (dense)"""
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec.astype('float32'), top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "idx": idx,
                "text": self.chunks[idx],
                "meta": self.meta[idx],
                "score": -float(distances[0][i])  
            })
        return results

    def bm25_search(self, query, top_k=10):
        """Поиск по BM25"""
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "idx": idx,
                    "text": self.chunks[idx],
                    "meta": self.meta[idx],
                    "score": float(scores[idx])
                })
        return results

    def hybrid_search(self, query, top_k=3, bm25_weight=1.0, dense_weight=1.0, k_rrf=60):
        candidate_k = max(top_k * 3, 10)
        dense_results = self.dense_search(query, top_k=candidate_k)
        bm25_results = self.bm25_search(query, top_k=candidate_k)

        rrf_scores = {}
        for rank, res in enumerate(dense_results):
            idx = res["idx"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_rrf + rank + 1)
        for rank, res in enumerate(bm25_results):
            idx = res["idx"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k_rrf + rank + 1)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        final_results = []
        for idx in sorted_indices:
            final_results.append((
                self.chunks[idx],
                self.meta[idx],
                rrf_scores[idx]  
            ))
        return final_results


    def search(self, query, top_k=3):
        """Обычный dense поиск"""
        results = self.dense_search(query, top_k)
        return [(r["text"], r["meta"], -r["score"]) for r in results]  