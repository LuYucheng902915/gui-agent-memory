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
from pydantic import SecretStr

from .config import MemoryConfig, get_config
from .models import ActionStep, ExperienceRecord, FactRecord, RetrievalResult
from .storage import MemoryStorage

# Module-level logger
logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Exception raised for retrieval-related errors."""


class MemoryRetriever:
    """
    Retrieval layer implementing hybrid search with unified re-ranking.

    Combines vector similarity search with keyword-based filtering,
    then applies re-ranking for optimal result quality.
    """

    def __init__(
        self, storage: MemoryStorage | None = None, config: MemoryConfig | None = None
    ) -> None:
        """Initialize the retrieval system with optional dependency injection."""
        self.config = config or get_config()
        self.storage = storage or MemoryStorage(self.config)
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

    def _keyword_filter_experiences(self, keywords: list[str]) -> dict[str, Any]:
        """
        Perform keyword-based filtering on experiences collection.

        Args:
            keywords: List of keywords to filter by

        Returns:
            ChromaDB query results for experiences matching keywords
        """
        if not keywords:
            return {}

        try:
            keyword_filter = self._build_keyword_filter(keywords)
            if not keyword_filter:
                return {}

            result = self.storage.query_experiences(
                query_texts=None,
                where=keyword_filter,
                n_results=50,  # Get a reasonable number for filtering
            )
            return result

        except Exception as e:
            self.logger.warning(f"Keyword filtering for experiences failed: {e}")
            return {}

    def _keyword_filter_facts(self, keywords: list[str]) -> dict[str, Any]:
        """
        Perform keyword-based filtering on facts collection.

        Args:
            keywords: List of keywords to filter by

        Returns:
            ChromaDB query results for facts matching keywords
        """
        if not keywords:
            return {}

        try:
            keyword_filter = self._build_keyword_filter(keywords)
            if not keyword_filter:
                return {}

            result = self.storage.query_facts(
                query_texts=None,
                where=keyword_filter,
                n_results=50,  # Get a reasonable number for filtering
            )
            return result

        except Exception as e:
            self.logger.warning(f"Keyword filtering for facts failed: {e}")
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
            experiences: list[dict[str, Any]] = []

            # Validate results structure
            if not all(key in results for key in ["ids", "documents", "metadatas"]):
                raise RetrievalError("Malformed storage results: missing required keys")

            if not results["ids"] or not results["ids"][0]:
                return experiences  # Return empty list for no results

            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = (
                results.get("distances", [[]])[0] if "distances" in results else []
            )

            # Check for length consistency
            if not (len(ids) == len(documents) == len(metadatas)):
                raise RetrievalError(
                    "Malformed storage results: inconsistent array lengths"
                )

            for i in range(len(ids)):
                experience_data = {
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i] if i < len(distances) else 0.0,
                    "type": "experience",
                }
                experiences.append(experience_data)

            return experiences

        except Exception as e:
            raise RetrievalError(
                f"Failed to perform vector search on experiences: {e}"
            ) from e

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
            facts: list[dict[str, Any]] = []

            # Validate results structure
            if not all(key in results for key in ["ids", "documents", "metadatas"]):
                raise RetrievalError("Malformed storage results: missing required keys")

            if not results["ids"] or not results["ids"][0]:
                return facts  # Return empty list for no results

            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = (
                results.get("distances", [[]])[0] if "distances" in results else []
            )

            # Check for length consistency
            if not (len(ids) == len(documents) == len(metadatas)):
                raise RetrievalError(
                    "Malformed storage results: inconsistent array lengths"
                )

            for i in range(len(ids)):
                fact_data = {
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i] if i < len(distances) else 0.0,
                    "type": "fact",
                }
                facts.append(fact_data)

            return facts

        except Exception as e:
            raise RetrievalError(
                f"Failed to perform vector search on facts: {e}"
            ) from e

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
        self, query: str | Any, candidates: list[dict[str, Any]] | Any, top_n: int = 10
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
        # Some tests may accidentally swap argument order; normalize
        if isinstance(query, list) and not isinstance(candidates, list):
            query, candidates = candidates, query

        if not candidates:
            return []

        # Limit candidates to avoid API limits
        candidates = candidates[: self.config.rerank_candidate_limit]

        # For single candidate, return as-is without reranking
        if len(candidates) == 1:
            # Return a copy to avoid modifying the original
            result = (
                candidates[0].copy()
                if isinstance(candidates[0], dict)
                else candidates[0]
            )
            return [result]

        # Normalize candidate shape to dict with 'document' key
        norm_candidates: list[dict[str, Any]] = []
        for c in candidates:
            if isinstance(c, dict):
                norm_candidates.append(c.copy())
            elif isinstance(c, tuple) and len(c) == 2:
                ctype, obj = c
                doc = getattr(obj, "task_description", None) or getattr(
                    obj, "content", str(obj)
                )
                norm_candidates.append({"type": ctype, "document": doc, "metadata": {}})
            else:
                norm_candidates.append(
                    {"type": "unknown", "document": str(c), "metadata": {}}
                )

        try:
            # Prepare documents for reranking
            documents = [result["document"] for result in norm_candidates]

            # Use requests to call the reranker API directly (as per official docs)

            # Support both SecretStr and plain str for tests/mocks
            api_key_obj = getattr(self.config, "reranker_llm_api_key", "")
            api_key = (
                api_key_obj.get_secret_value()
                if isinstance(api_key_obj, SecretStr)
                else str(api_key_obj)
            )

            headers = {
                "X-Failover-Enabled": "true",
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "query": query,
                "documents": documents,
                "model": self.config.reranker_model,
                "top_n": top_n,
            }

            # Call reranker API
            response = requests.post(
                str(self.config.reranker_llm_base_url),
                headers=headers,
                json=payload,
                timeout=self.config.http_timeout_seconds,
            )
            response.raise_for_status()

            # Parse reranking results from API response
            api_response = response.json()

            # Parse reranking results from API response
            if "results" in api_response and isinstance(api_response["results"], list):
                reranked_results = []
                for i, result_data in enumerate(api_response["results"][:top_n]):
                    # Get the document index from the API response
                    doc_index = result_data.get("index", i)
                    if doc_index < len(norm_candidates):
                        result = norm_candidates[doc_index].copy()
                        result["rerank_score"] = result_data.get(
                            "relevance_score", 1.0 - i * 0.1
                        )
                        reranked_results.append(result)
                return reranked_results

            # Fallback: return original results if API format is unexpected
            return candidates[:top_n]

        except Exception as e:
            # Fallback: return original results on reranking failure
            self.logger.warning("Reranking failed, using original order: %s", e)
            return candidates[:top_n]

    def _convert_to_memory_objects(
        self, results: list[dict[str, Any]]
    ) -> tuple[list[ExperienceRecord], list[FactRecord]]:
        """
        Convert search results back to Pydantic models.

        Args:
            results: List of search results with metadata

        Returns:
            Tuple of (experiences, facts) as Pydantic models

        Raises:
            RetrievalError: If conversion fails
        """
        return self._convert_results_to_models(results)

    def _parse_keywords_from_metadata(self, metadata: dict[str, Any]) -> list[str]:
        """
        Parse keywords from metadata dictionary.

        Args:
            metadata: Metadata dictionary containing keywords

        Returns:
            List of keywords
        """
        keywords_str = metadata.get("keywords", "")
        if isinstance(keywords_str, str):
            return [k.strip() for k in keywords_str.split(",") if k.strip()]
        elif isinstance(keywords_str, list):
            return keywords_str
        else:
            return []

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
        conversion_errors = []

        for result in results:
            try:
                if result["type"] == "experience":
                    # Reconstruct ExperienceRecord
                    metadata = result["metadata"]

                    # Parse action_flow back from JSON
                    try:
                        action_flow_data = json.loads(metadata.get("action_flow", "[]"))
                        action_steps = [
                            ActionStep(**step_data) for step_data in action_flow_data
                        ]
                    except json.JSONDecodeError as e:
                        raise RetrievalError(
                            f"Failed to parse action_flow JSON: {e}"
                        ) from e

                    experience = ExperienceRecord(
                        task_description=result["document"],
                        keywords=self._parse_keywords_from_metadata(metadata),
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
                        keywords=self._parse_keywords_from_metadata(metadata),
                        source=metadata.get("source", "unknown"),
                        usage_count=metadata.get("usage_count", 0),
                        last_used_at=metadata.get("last_used_at", ""),
                    )
                    facts.append(fact)

            except Exception as e:
                conversion_errors.append(str(e))
                self.logger.warning("Failed to convert result to model: %s", e)
                continue

        # If we have results but no successful conversions, raise an error
        if results and not experiences and not facts and conversion_errors:
            raise RetrievalError(
                f"All conversion attempts failed: {conversion_errors[0]}"
            )

        return experiences, facts

    def _update_experience_usage(
        self, reranked_experiences: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Update usage statistics for experience results."""
        if not reranked_experiences:
            return reranked_experiences

        from datetime import datetime

        current_time = datetime.now()

        experience_ids = []
        for result in reranked_experiences:
            if result.get("type") == "experience":
                experience_ids.append(result["metadata"]["source_task_id"])

        if experience_ids:
            self.storage.update_usage_stats(
                experience_ids, self.config.experiential_collection_name
            )

            # Update the metadata in results to reflect new usage stats
            for result in reranked_experiences:
                if result.get("type") == "experience":
                    current_count = result["metadata"].get("usage_count", 0)
                    result["metadata"]["usage_count"] = current_count + 1
                    result["metadata"]["last_used_at"] = current_time.isoformat()

        return reranked_experiences

    def _update_fact_usage(
        self, reranked_facts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Update usage statistics for fact results."""
        if not reranked_facts:
            return reranked_facts

        from datetime import datetime

        current_time = datetime.now()

        fact_ids = [
            result["id"] for result in reranked_facts if result.get("type") == "fact"
        ]

        if fact_ids:
            self.storage.update_usage_stats(
                fact_ids, self.config.declarative_collection_name
            )

            # Update the metadata in results to reflect new usage stats
            for result in reranked_facts:
                if result.get("type") == "fact":
                    current_count = result["metadata"].get("usage_count", 0)
                    result["metadata"]["usage_count"] = current_count + 1
                    result["metadata"]["last_used_at"] = current_time.isoformat()

        return reranked_facts

    def _update_usage_statistics(
        self,
        reranked_experiences: list[dict[str, Any]],
        reranked_facts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Update usage statistics for retrieved memories.

        Args:
            reranked_experiences: List of experience results from reranking
            reranked_facts: List of fact results from reranking

        Returns:
            Tuple of (updated_experiences, updated_facts) with new usage stats
        """
        try:
            updated_experiences = self._update_experience_usage(reranked_experiences)
            updated_facts = self._update_fact_usage(reranked_facts)
            return updated_experiences, updated_facts
        except Exception as e:
            # Log the error but don't fail the retrieval
            self.logger.warning(f"Failed to update usage statistics: {e}")
            logger.warning(f"Failed to update usage statistics: {e}")
            return reranked_experiences, reranked_facts

    def _update_usage_stats(self, memory_ids: list[str]) -> None:
        """
        Update usage statistics for a list of memory IDs.

        Args:
            memory_ids: List of memory IDs to update

        Note:
            This method handles storage errors gracefully and logs them.
        """
        try:
            # For simplicity, assume all are experiences - in practice this would
            # need to identify the collection for each ID
            if memory_ids:
                self.storage.update_usage_stats(
                    memory_ids, self.config.experiential_collection_name
                )
        except Exception as e:
            # Handle storage errors gracefully and log them
            logger.error(f"Failed to update usage stats for memories {memory_ids}: {e}")
            self.logger.warning(f"Failed to update usage stats: {e}")

    def _merge_search_results(
        self,
        experience_results: list[dict[str, Any]],
        fact_results: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Merge and sort search results from experiences and facts.

        Args:
            experience_results: List of experience search results
            fact_results: List of fact search results
            limit: Maximum number of results to return

        Returns:
            Merged and sorted results limited by the specified limit
        """
        # Combine all results
        all_results = experience_results + fact_results

        # Sort by score in descending order (highest scores first)
        sorted_results = sorted(
            all_results, key=lambda x: x.get("score", 0.0), reverse=True
        )

        # Apply limit
        return sorted_results[:limit]

    def _perform_hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        query_keywords: list[str],
        top_n: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Perform hybrid search for both experiences and facts."""
        # Use larger candidate pool for hybrid search
        top_k = top_n * self.config.hybrid_topk_multiplier  # configurable multiplier

        # Parallel hybrid retrieval for experiences
        vector_experiences = self._vector_search_experiences(query_embedding, top_k)
        keyword_experiences = self._keyword_search_experiences(query_keywords, top_k)

        # Parallel hybrid retrieval for facts
        vector_facts = self._vector_search_facts(query_embedding, top_k)
        keyword_facts = self._keyword_search_facts(query_keywords, top_k)

        # Merge and deduplicate results
        merged_experiences = self._merge_and_deduplicate_results(
            vector_experiences, keyword_experiences
        )
        merged_facts = self._merge_and_deduplicate_results(vector_facts, keyword_facts)

        # Rerank results
        reranked_experiences = self._rerank_results(query, merged_experiences, top_n)
        reranked_facts = self._rerank_results(query, merged_facts, top_n)

        return reranked_experiences, reranked_facts

    def retrieve_memories(
        self, query: str, top_n: int | None = None
    ) -> RetrievalResult:
        """
        Retrieve relevant memories using hybrid search and reranking.

        Args:
            query: Query string describing the current task or intent
            top_n: Number of final results to return for each memory type

        Returns:
            RetrievalResult containing relevant experiences and facts

        Raises:
            RetrievalError: If retrieval process fails
        """
        try:
            if top_n is None:
                top_n = self.config.default_top_n
            if top_n <= 0:
                return RetrievalResult(
                    experiences=[], facts=[], query=query, total_results=0
                )

            # Prepare query components
            query_embedding = self._generate_query_embedding(query)
            query_keywords = self._extract_query_keywords(query)

            # Perform hybrid search
            reranked_experiences, reranked_facts = self._perform_hybrid_search(
                query, query_embedding, query_keywords, top_n
            )

            # Finalize results
            result = self._finalize_retrieval_results(
                reranked_experiences, reranked_facts
            )
            result.query = query
            return result

        except Exception as e:
            raise RetrievalError(f"Memory retrieval failed: {e}") from e

    def _finalize_retrieval_results(
        self,
        reranked_experiences: list[dict[str, Any]],
        reranked_facts: list[dict[str, Any]],
    ) -> RetrievalResult:
        """Convert search results to models and create final retrieval result."""
        # Update usage statistics and get updated results
        updated_experiences, updated_facts = self._update_usage_statistics(
            reranked_experiences, reranked_facts
        )

        # Convert to domain models with updated metadata
        final_experiences, _ = self._convert_results_to_models(updated_experiences)
        _, final_facts = self._convert_results_to_models(updated_facts)

        return RetrievalResult(
            experiences=final_experiences,
            facts=final_facts,
            query="",  # Will be set in main method
            total_results=len(final_experiences) + len(final_facts),
        )

    def get_similar_experiences(
        self, task_description: str, top_n: int | None = None
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
            if top_n is None:
                top_n = self.config.default_top_n
            query_embedding = self._generate_query_embedding(task_description)
            vector_results = self._vector_search_experiences(query_embedding, top_n)

            try:
                experiences, _ = self._convert_to_memory_objects(vector_results)
                return experiences
            except Exception as e:
                raise RetrievalError(f"Failed to convert experiences: {e}") from e

        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"Failed to get similar experiences: {e}") from e

    def get_related_facts(
        self, topic: str, top_n: int | None = None
    ) -> list[FactRecord]:
        """
        Get facts related to a specific topic.

        Args:
            topic: Topic to search for related facts
            top_n: Number of related facts to return

        Returns:
            List of related facts
        """
        try:
            if top_n is None:
                top_n = self.config.default_top_n
            query_embedding = self._generate_query_embedding(topic)
            vector_results = self._vector_search_facts(query_embedding, top_n)

            try:
                _, facts = self._convert_to_memory_objects(vector_results)
                return facts
            except Exception as e:
                raise RetrievalError(f"Failed to convert facts: {e}") from e

        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"Failed to get related facts: {e}") from e
