# app/engines/rag/reranker.py
import numpy as np

class Reranker:
    def __init__(self):
        pass
    
    def rank(self, query: str, documents: list) -> list:
        """
        Simple reranking based on relevance score (distance converted to similarity)
        """
        for doc in documents:
            sim = 1 - doc.get('distance', 1)
            doc['relevance'] = sim
        documents.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        return documents

class ContextMerger:
    def __init__(self, max_tokens=3000):
        self.max_tokens = max_tokens
    
    def merge(self, query: str, memories: list, knowledge: list, associative: list) -> str:
        context_parts = []
        tokens_used = 0
        for item in memories:
            text = item['text']
            tokens = len(text.split())
            if tokens_used + tokens < self.max_tokens:
                context_parts.append(f"[Memory] {text}")
                tokens_used += tokens
        for item in knowledge:
            text = item['text']
            tokens = len(text.split())
            if tokens_used + tokens < self.max_tokens:
                context_parts.append(f"[Knowledge] {text}")
                tokens_used += tokens
        for meta, score in associative:
            text = meta.get('text', '')
            tokens = len(text.split())
            if tokens_used + tokens < self.max_tokens:
                context_parts.append(f"[Associative] {text}")
                tokens_used += tokens
        return "\n\n".join(context_parts)