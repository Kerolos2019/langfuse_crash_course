# semantic_cache_openai.py
"""
Semantic Cache for LLM Responses (OpenAI end to end)
Reduces costs by 30-50% for applications with repeated query patterns

Required packages:
    pip install langfuse chromadb openai python-dotenv

Required env vars (.env):
    OPENAI_API_KEY=sk-...
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com

NOTE: this uses 1536-dim OpenAI embeddings. If you ran the MiniLM version
before, delete ./chroma_cache_db or keep the new collection name below —
vectors of different dimensions cannot share a collection.
"""

import chromadb
import hashlib
import json
import os
from typing import Optional, Tuple, List
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langfuse import observe, get_client

# Drop-in replacement: traces every call with model, tokens and cost
from langfuse.openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # dim 1536

# Roughly what one cache lookup costs: ~20 tokens at $0.02 / 1M tokens
LOOKUP_COST = 0.0000004

client = OpenAI()
langfuse = get_client()


def call_openai(query: str) -> str:
    """Call the OpenAI API and return the answer text."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": query}],
        name="cache-miss-completion",  # consumed by Langfuse, not sent to OpenAI
    )
    return response.choices[0].message.content


class SemanticCache:
    def __init__(
        self,
        similarity_threshold: float = 0.92,  # How similar queries must be
        ttl_hours: int = 24,  # Cache expiration
        persist_directory: str = "./chroma_cache_db",  # Persist across runs
        embed_model: str = EMBED_MODEL,
    ):
        # Use PersistentClient for cache to survive across runs
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="llm_cache_openai", metadata={"hnsw:space": "cosine"}
        )
        self.embed_model = embed_model
        self.threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)

    def embed(self, text: str) -> List[float]:
        """Embed one string via the OpenAI embeddings API.

        Public so callers can embed once and reuse the vector for both the
        lookup and the write — a miss would otherwise pay for two calls.
        """
        response = client.embeddings.create(
            model=self.embed_model,
            input=text,
            name="cache-lookup-embed",
        )
        return response.data[0].embedding

    def _is_expired(self, timestamp: str) -> bool:
        cached_time = datetime.fromisoformat(timestamp)
        return datetime.now() - cached_time > self.ttl

    def get(
        self, query: str, embedding: Optional[List[float]] = None
    ) -> Optional[Tuple[str, float]]:
        """
        Look for a cached response.

        Returns: (cached_response, similarity_score) or None
        """
        query_embedding = embedding or self.embed(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"][0]:
            return None

        # Chroma returns cosine distance; flip it to similarity
        distance = results["distances"][0][0]
        similarity = 1 - distance

        if similarity < self.threshold:
            return None

        # Check expiration
        metadata = results["metadatas"][0][0]
        if self._is_expired(metadata["timestamp"]):
            return None

        cached_response = json.loads(metadata["response"])
        return (cached_response, similarity)

    def set(
        self,
        query: str,
        response: str,
        metadata: dict = None,
        embedding: Optional[List[float]] = None,
    ):
        """Cache a response for a query."""
        query_embedding = embedding or self.embed(query)
        doc_id = hashlib.md5(query.encode()).hexdigest()

        self.collection.upsert(
            ids=[doc_id],
            embeddings=[query_embedding],
            documents=[query],
            metadatas=[
                {
                    "response": json.dumps(response),
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {}),
                }
            ],
        )


# Usage
cache = SemanticCache(similarity_threshold=0.85)


@observe()
def cached_llm_call(query: str) -> Tuple[str, Optional[float]]:
    """LLM call with semantic caching.

    Returns (response_text, similarity) where similarity is None on a miss.
    Every query goes through here, so every query gets its own trace.
    """
    # Embed once, reuse for both the lookup and (on a miss) the write
    embedding = cache.embed(query)

    cached = cache.get(query, embedding=embedding)
    if cached:
        response, similarity = cached
        langfuse.update_current_span(
            metadata={
                "cache_hit": True,
                "similarity": similarity,
                "lookup_cost_usd": LOOKUP_COST,
            }
        )
        langfuse.update_current_trace(tags=["cache-hit"])
        return response, similarity

    # Cache miss - call the LLM
    response_text = call_openai(query)
    cache.set(query, response_text, embedding=embedding)

    langfuse.update_current_span(metadata={"cache_hit": False})
    langfuse.update_current_trace(tags=["cache-miss"])

    return response_text, None


def simulate_semantic_cache():
    """
    Simulation demonstrating semantic cache hits with similar questions.

    The key insight: Questions don't need to be IDENTICAL - they need to be
    SEMANTICALLY SIMILAR (meaning the same thing in different words).
    """
    print("=" * 70)
    print("🧠 SEMANTIC CACHE SIMULATION")
    print("=" * 70)

    total_queries = 0
    cache_hits = 0
    cache_misses = 0
    api_calls_saved = 0

    question_groups = [
        {
            "topic": "Python Programming",
            "questions": [
                "What is a Python list comprehension?",
                "Explain list comprehensions in Python",
                "How do list comprehensions work in Python?",
                "What are Python list comprehensions and how to use them?",
            ],
        },
        {
            "topic": "Machine Learning",
            "questions": [
                "What is the difference between supervised and unsupervised learning?",
                "Explain supervised vs unsupervised machine learning",
                "How does supervised learning differ from unsupervised learning?",
                "Compare supervised and unsupervised learning in ML",
            ],
        },
        {
            "topic": "API Concepts",
            "questions": [
                "What is a REST API?",
                "Explain what REST APIs are",
                "What does REST API mean?",
                "Can you describe what a RESTful API is?",
            ],
        },
    ]

    for group in question_groups:
        print(f"\n{'─' * 70}")
        print(f"📁 TOPIC: {group['topic']}")
        print(f"{'─' * 70}")

        for i, question in enumerate(group["questions"]):
            total_queries += 1
            print(f'\n🔍 Query {i+1}: "{question}"')

            # One path for every query: the decorated function handles both
            # the lookup and the API call, so the trace reflects reality
            result, similarity = cached_llm_call(question)

            if similarity is not None:
                cache_hits += 1
                api_calls_saved += 1
                print("   ✅ CACHE HIT")
                print(f"   📊 Similarity: {similarity:.2%}")
                print(f"   💰 Completion skipped (lookup cost ~${LOOKUP_COST:.7f})")
            else:
                cache_misses += 1
                print("   ❌ CACHE MISS - called the OpenAI API")
                print("   💾 Response cached for future similar queries")

            display_response = result[:80] + "..." if len(result) > 80 else result
            print(f"   📝 Response: {display_response}")

    hit_rate = (cache_hits / total_queries * 100) if total_queries else 0
    print(f"\n{'=' * 70}")
    print("📊 CACHE PERFORMANCE SUMMARY")
    print(f"{'=' * 70}")
    print(f"    Total queries:     {total_queries:3d}")
    print(f"    Cache hits:        {cache_hits:3d}  ✅")
    print(f"    Cache misses:      {cache_misses:3d}  ❌")
    print(f"    Hit rate:          {hit_rate:5.1f}%")
    print(f"    Completions saved: {api_calls_saved:3d}  💰")

    if cache_hits > 0:
        print("\n🎉 SEMANTIC CACHING IS WORKING!")
        print("   Similar questions are returning cached responses.")

    print("\n💡 TIP: Run this script again to see even MORE cache hits.")
    print("   The cache persists to disk, so previous queries are remembered.")


if __name__ == "__main__":
    simulate_semantic_cache()
    langfuse.flush()