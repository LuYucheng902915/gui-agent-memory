"""
Retrieval layer for hybrid search and reranking of memories.

This module handles:
- Parallel hybrid retrieval (vector + keyword-based)
- Unified re-ranking using reranker models
- Query processing and keyword extraction
- Result fusion and filtering
"""

import json
import logging
from typing import Any

import jieba
import requests

from .config import get_config
from .models import ActionStep, ExperienceRecord, FactRecord, RetrievalResult
from .storage import MemoryStorage


class RetrievalError(Exception):
    """Exception raised for retrieval-related errors."""


class MemoryRetriever:
    """
    Retrieval layer implementing hybrid search with unified re-ranking.

    Combines vector similarity search with keyword-based filtering,
    then applies re-ranking for optimal result quality.
    """

    def __init__(self) -> None:
        """Initialize the retrieval system."""
        self.config = get_config()
        self.storage = MemoryStorage()
        self.logger = logging.getLogger(__name__)

    def _generate_query_embedding(self, query: str) -> list[float]:
        """
        Generate embedding for the query text.

        Args:
            query: Query text to embed

        Returns:
            Query embedding vector

        Raises:
            RetrievalError: If embedding generation fails
        """
        try:
            client = self.config.get_embedding_client()
            response = client.embeddings.create(
                model=self.config.embedding_model, input=query
            )
            return response.data[0].embedding
        except Exception as e:
            raise RetrievalError(f"Failed to generate query embedding: {e}") from e

    def _extract_query_keywords(self, query: str) -> list[str]:
        """
        Extract keywords from query using jieba tokenization.

        Args:
            query: Query string to tokenize

        Returns:
            List of extracted keywords
        """
        # Tokenize the query
        tokens = jieba.lcut(query, cut_all=False)

        # Filter and normalize keywords
        stop_words = {
            "的",
            "是",
            "在",
            "和",
            "与",
            "或",
            "但是",
            "如果",
            "那么",
            "了",
            "也",
            "就",
            "都",
            "要",
            "可以",
            "这",
            "那",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "then",
            "to",
            "for",
            "with",
            "by",
            "from",
            "at",
            "on",
            "in",
        }

        keywords = []
        for token in tokens:
            if len(token) > 1 and token.lower() not in stop_words:
                keywords.append(token.lower())

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)

        return unique_keywords

    def _extract_keywords(self, query: str) -> list[str]:
        """
        Alias for _extract_query_keywords for backward compatibility.

        Args:
            query: Query string to tokenize

        Returns:
            List of extracted keywords
        """
        return self._extract_query_keywords(query)

    def _build_keyword_filter(self, keywords: list[str]) -> dict[str, Any]:
        """
        Build ChromaDB filter for keyword-based search.

        Args:
            keywords: List of keywords to search for

        Returns:
            ChromaDB where filter dictionary
        """
        if not keywords:
            return {}

        # Since keywords are stored as comma-separated strings in ChromaDB,
        # we'll use vector search primarily and skip complex keyword filtering
        # This is a temporary workaround - in production, consider using full-text search
        return {}

    def _vector_search_experiences(
        self, query_embedding: list[float], top_k: int = 20
    ) -> list[dict[str, Any]]:
        """
        Perform vector similarity search on experiences.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to retrieve

        Returns:
            List of experience search results with metadata
        """
        try:
            results = self.storage.query_experiences(
                query_embeddings=[query_embedding], n_results=top_k
            )

            # Convert ChromaDB results to standardized format
            experiences = []
            for i in range(len(results["ids"][0])):
                experience_data = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": (
                        results["distances"][0][i] if "distances" in results else 0.0
                    ),
                    "type": "experience",
                }
                experiences.append(experience_data)

            return experiences

        except Exception as e:
            raise RetrievalError(f"Vector search for experiences failed: {e}") from e

    def _vector_search_facts(
        self, query_embedding: list[float], top_k: int = 20
    ) -> list[dict[str, Any]]:
        """
        Perform vector similarity search on facts.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to retrieve

        Returns:
            List of fact search results with metadata
        """
        try:
            results = self.storage.query_facts(
                query_embeddings=[query_embedding], n_results=top_k
            )

            # Convert ChromaDB results to standardized format
            facts = []
            for i in range(len(results["ids"][0])):
                fact_data = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": (
                        results["distances"][0][i] if "distances" in results else 0.0
                    ),
                    "type": "fact",
                }
                facts.append(fact_data)

            return facts

        except Exception as e:
            raise RetrievalError(f"Vector search for facts failed: {e}") from e

    def _keyword_search_experiences(
        self, keywords: list[str], top_k: int = 20
    ) -> list[dict[str, Any]]:
        """
        Perform keyword-based search on experiences.

        Args:
            keywords: List of keywords to search for
            top_k: Number of results to retrieve

        Returns:
            List of experience search results with metadata
        """
        if not keywords:
            return []

        try:
            keyword_filter = self._build_keyword_filter(keywords)

            # If no valid keyword filter, skip keyword search
            if not keyword_filter:
                return []

            results = self.storage.query_experiences(
                query_texts=keywords[:1],  # Use first keyword as query text
                where=keyword_filter,
                n_results=top_k,
            )

            # Convert ChromaDB results to standardized format
            experiences = []
            for i in range(len(results["ids"][0])):
                experience_data = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": (
                        results["distances"][0][i] if "distances" in results else 0.0
                    ),
                    "type": "experience",
                }
                experiences.append(experience_data)

            return experiences

        except Exception as e:
            raise RetrievalError(f"Keyword search for experiences failed: {e}") from e

    def _keyword_search_facts(
        self, keywords: list[str], top_k: int = 20
    ) -> list[dict[str, Any]]:
        """
        Perform keyword-based search on facts.

        Args:
            keywords: List of keywords to search for
            top_k: Number of results to retrieve

        Returns:
            List of fact search results with metadata
        """
        if not keywords:
            return []

        try:
            keyword_filter = self._build_keyword_filter(keywords)

            # If no valid keyword filter, skip keyword search
            if not keyword_filter:
                return []

            results = self.storage.query_facts(
                query_texts=keywords[:1],  # Use first keyword as query text
                where=keyword_filter,
                n_results=top_k,
            )

            # Convert ChromaDB results to standardized format
            facts = []
            for i in range(len(results["ids"][0])):
                fact_data = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": (
                        results["distances"][0][i] if "distances" in results else 0.0
                    ),
                    "type": "fact",
                }
                facts.append(fact_data)

            return facts

        except Exception as e:
            raise RetrievalError(f"Keyword search for facts failed: {e}") from e

    def _merge_and_deduplicate_results(
        self,
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Merge vector and keyword search results, removing duplicates.

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search

        Returns:
            Merged and deduplicated results
        """
        seen_ids = set()
        merged_results = []

        # Add vector results first (typically higher quality)
        for result in vector_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                result["retrieval_method"] = "vector"
                merged_results.append(result)

        # Add keyword results that weren't already found
        for result in keyword_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                result["retrieval_method"] = "keyword"
                merged_results.append(result)

        return merged_results

    def _rerank_results(
        self, query: str, candidates: list[dict[str, Any]], top_n: int = 10
    ) -> list[dict[str, Any]]:
        """
        Re-rank candidate results using the reranker model.

        Args:
            query: Original query string
            candidates: List of candidate results to rerank
            top_n: Number of top results to return

        Returns:
            Re-ranked results
        """
        if not candidates:
            return []

        # Limit candidates to avoid API limits
        candidates = candidates[:20]

        try:
            reranker_config = self.config.get_reranker_config()

            # Prepare documents for reranking
            documents = [result["document"] for result in candidates]

            # Prepare headers for the request
            headers = {
                "X-Failover-Enabled": "true",
                "Authorization": f"Bearer {reranker_config['api_key']}",
                "Content-Type": "application/json",
            }

            # Prepare payload for reranker API
            payload = {
                "query": query,
                "documents": documents,
                "model": reranker_config["model"],
            }

            # Call reranker API
            response = requests.post(
                reranker_config["base_url"], headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()

            # Parse reranking results from API response
            api_response = response.json()

            # The reranker API returns ranked results with scores
            if "results" in api_response:
                reranked_results = []
                for i, result_data in enumerate(api_response["results"][:top_n]):
                    # Get the document index from the API response
                    doc_index = result_data.get("index", i)
                    if doc_index < len(candidates):
                        result = candidates[doc_index].copy()
                        result["rerank_score"] = result_data.get(
                            "relevance_score", 1.0 - i * 0.1
                        )
                        reranked_results.append(result)
                return reranked_results
            # Fallback: return original results if API format is unexpected
            for i, result in enumerate(candidates[:top_n]):
                result["rerank_score"] = 1.0 - i * 0.1
            return candidates[:top_n]

        except Exception as e:
            # Fallback: return original results on reranking failure
            self.logger.warning("Reranking failed, using original order: %s", e)
            return candidates[:top_n]

    def _convert_results_to_models(
        self, results: list[dict[str, Any]]
    ) -> tuple[list[ExperienceRecord], list[FactRecord]]:
        """
        Convert search results back to Pydantic models.

        Args:
            results: List of search results with metadata

        Returns:
            Tuple of (experiences, facts) as Pydantic models
        """
        experiences = []
        facts = []

        for result in results:
            try:
                if result["type"] == "experience":
                    # Reconstruct ExperienceRecord
                    metadata = result["metadata"]

                    # Parse action_flow back from JSON
                    action_flow_data = json.loads(metadata.get("action_flow", "[]"))
                    action_steps = [
                        ActionStep(**step_data) for step_data in action_flow_data
                    ]

                    experience = ExperienceRecord(
                        task_description=result["document"],
                        keywords=(
                            metadata.get("keywords", "").split(",")
                            if metadata.get("keywords")
                            else []
                        ),
                        action_flow=action_steps,
                        preconditions=metadata.get("preconditions", ""),
                        is_successful=metadata.get("is_successful", True),
                        usage_count=metadata.get("usage_count", 0),
                        last_used_at=metadata.get("last_used_at", ""),
                        source_task_id=metadata.get("source_task_id", ""),
                    )
                    experiences.append(experience)

                elif result["type"] == "fact":
                    # Reconstruct FactRecord
                    metadata = result["metadata"]

                    fact = FactRecord(
                        content=result["document"],
                        keywords=(
                            metadata.get("keywords", "").split(",")
                            if metadata.get("keywords")
                            else []
                        ),
                        source=metadata.get("source", "unknown"),
                        usage_count=metadata.get("usage_count", 0),
                        last_used_at=metadata.get("last_used_at", ""),
                    )
                    facts.append(fact)

            except Exception as e:
                self.logger.warning("Failed to convert result to model: %s", e)
                continue

        return experiences, facts

    def retrieve_memories(
        self, query: str, top_n: int = 3, top_k: int = 20
    ) -> RetrievalResult:
        """
        Retrieve relevant memories using hybrid search and reranking.

        Args:
            query: Query string describing the current task or intent
            top_n: Number of final results to return for each memory type
            top_k: Number of candidates to retrieve before reranking

        Returns:
            RetrievalResult containing relevant experiences and facts

        Raises:
            RetrievalError: If retrieval process fails
        """
        try:
            # Step 1: Generate query embedding
            query_embedding = self._generate_query_embedding(query)

            # Step 2: Extract query keywords
            query_keywords = self._extract_query_keywords(query)

            # Step 3: Parallel hybrid retrieval for experiences
            vector_experiences = self._vector_search_experiences(query_embedding, top_k)
            keyword_experiences = self._keyword_search_experiences(
                query_keywords, top_k
            )

            # Step 4: Parallel hybrid retrieval for facts
            vector_facts = self._vector_search_facts(query_embedding, top_k)
            keyword_facts = self._keyword_search_facts(query_keywords, top_k)

            # Step 5: Merge and deduplicate results
            merged_experiences = self._merge_and_deduplicate_results(
                vector_experiences, keyword_experiences
            )
            merged_facts = self._merge_and_deduplicate_results(
                vector_facts, keyword_facts
            )

            # Step 6: Rerank results
            reranked_experiences = self._rerank_results(
                query, merged_experiences, top_n
            )
            reranked_facts = self._rerank_results(query, merged_facts, top_n)

            # Step 7: Convert results back to Pydantic models
            final_experiences, _ = self._convert_results_to_models(reranked_experiences)
            _, final_facts = self._convert_results_to_models(reranked_facts)

            # Step 8: Create and return retrieval result
            return RetrievalResult(
                experiences=final_experiences,
                facts=final_facts,
                query=query,
                total_results=len(final_experiences) + len(final_facts),
            )

        except Exception as e:
            raise RetrievalError(f"Memory retrieval failed: {e}") from e

    def get_similar_experiences(
        self, task_description: str, top_n: int = 5
    ) -> list[ExperienceRecord]:
        """
        Get experiences similar to a given task description.

        Args:
            task_description: Description of the task to find similar experiences for
            top_n: Number of similar experiences to return

        Returns:
            List of similar experiences
        """
        try:
            query_embedding = self._generate_query_embedding(task_description)
            vector_results = self._vector_search_experiences(query_embedding, top_n)

            experiences, _ = self._convert_results_to_models(vector_results)
            return experiences

        except Exception as e:
            raise RetrievalError(f"Failed to get similar experiences: {e}") from e

    def get_related_facts(self, topic: str, top_n: int = 5) -> list[FactRecord]:
        """
        Get facts related to a specific topic.

        Args:
            topic: Topic to search for related facts
            top_n: Number of related facts to return

        Returns:
            List of related facts
        """
        try:
            query_embedding = self._generate_query_embedding(topic)
            vector_results = self._vector_search_facts(query_embedding, top_n)

            _, facts = self._convert_results_to_models(vector_results)
            return facts

        except Exception as e:
            raise RetrievalError(f"Failed to get related facts: {e}") from e
