import numpy as np
from typing import List, Tuple
import asyncio


class HopfieldMemory:
    """
    Simplified Hopfield network for associative memory retrieval.
    Stores patterns (embeddings) and retrieves via energy minimization.
    """
    def __init__(self, size: int = 1536):
        self.size = size
        self.patterns = []       # list of stored embeddings
        self.metadata = []       # associated metadata
        self.W = None            # weight matrix (lazy init)
    
    async def store(self, embedding: List[float], meta: dict):
        self.patterns.append(np.array(embedding))
        self.metadata.append(meta)
        # Update weight matrix: Hebbian learning (sum of outer products)
        if self.W is None:
            self.W = np.zeros((self.size, self.size))
        new_pattern = self.patterns[-1].reshape(-1, 1)
        self.W += new_pattern @ new_pattern.T
        # Normalize to avoid explosion
        self.W = self.W / (len(self.patterns) + 1)
    
    async def retrieve(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[dict, float]]:
        if not self.patterns:
            return []
        query = np.array(query_embedding)
        # Compute energies: - (W * query) dot query (simplified)
        energies = []
        for i, pat in enumerate(self.patterns):
            energy = -np.dot(pat, query)
            energies.append((i, energy))
        energies.sort(key=lambda x: x[1])  # lower energy = more similar
        results = []
        for i, eng in energies[:top_k]:
            similarity = 1 / (1 + abs(eng))  # convert to similarity
            results.append((self.metadata[i], similarity))
        return results

# For production, we use vector DB as the primary storage, and Hopfield as an additional associative layer.
class HopfieldAssociativeMemory:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.hopfield = HopfieldMemory()
        self.cache_size = 100  # store recent 100 in Hopfield
    
    async def add_memory(self, user_id: str, text: str, embedding: List[float], metadata: dict):
        # Store in vector DB (persistent)
        await self.vector_store.upsert(user_id, text, embedding, metadata)
        # Also add to Hopfield for fast associative recall
        if len(self.hopfield.patterns) < self.cache_size:
            await self.hopfield.store(embedding, {**metadata, "user_id": user_id})
    
    async def retrieve_associative(self, query_embedding: List[float], user_id: str = None, top_k=3):
        # First try Hopfield for quick associative matches
        hopfield_results = await self.hopfield.retrieve(query_embedding, top_k)
        # Filter by user_id if needed
        filtered = [(m, s) for m, s in hopfield_results if user_id is None or m.get("user_id") == user_id]
        return filtered