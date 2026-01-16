from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np


class HybridRetriever:
    
    def __init__(self, vector_store, embedding_model, chunks: List[Dict[str, Any]]):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.chunks = chunks

        print("Initializing BM25 keyword search...")
        tokenized_docs = [chunk['content'].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_docs)
        print("✓ BM25 initialized")
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        semantic_weight: float = 0.7,
        categories: List[str] = None
    ) -> List[Dict[str, Any]]:

        keyword_weight = 1 - semantic_weight
        
        filter_metadata = None
        if categories and len(categories) > 0:
            if len(categories) == 1:
                filter_metadata = {"category": categories[0]}
            else:
                filter_metadata = {"$or": [{"category": cat} for cat in categories]}
        
        query_embedding = self.embedding_model.embed_text(query)
        semantic_results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results * 2,
            filter_metadata=filter_metadata
        )
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            semantic_results['documents'],
            semantic_results['metadatas'],
            semantic_results['distances']
        )):
            if categories and metadata.get('category') not in categories:
                continue

            chunk_idx = None
            for idx, chunk in enumerate(self.chunks):
                if chunk['content'] == doc and chunk['metadata'].get('source') == metadata.get('source'):
                    chunk_idx = idx
                    break
            
            semantic_score = 1 - (distance / 2)
            keyword_score = bm25_scores[chunk_idx] if chunk_idx is not None else 0
            keyword_score = keyword_score / (max(bm25_scores) + 1e-6)

            combined_score = (
                semantic_weight * semantic_score +
                keyword_weight * keyword_score
            )
            
            results.append({
                'content': doc,
                'metadata': metadata,
                'score': combined_score,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:n_results]