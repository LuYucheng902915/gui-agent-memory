"""
Ingestion layer for learning experiences and importing knowledge.

This module handles:
- Experience learning from raw task execution history
- Knowledge ingestion from various sources
- LLM-based experience distillation
- Embedding generation and storage
- Idempotency guarantees and error handling
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import jieba

from .config import get_config
from .models import ActionStep, ExperienceRecord, FactRecord, LearningRequest
from .storage import MemoryStorage

# Module-level logger
logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Exception raised for ingestion-related errors."""


class MemoryIngestion:
    """
    Ingestion layer for learning experiences and importing knowledge.

    Handles the conversion of raw execution history into structured memory records
    and their storage with appropriate embeddings.
    """

    def __init__(self) -> None:
        """Initialize the ingestion system."""
        self.config = get_config()
        self.storage = MemoryStorage()

        # Setup logging
        self._setup_logging()

        # Load prompt templates
        self._load_prompts()

    def _setup_logging(self) -> None:
        """Setup logging for failed learning tasks."""
        log_path = Path(self.config.failed_learning_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create file handler for failed learning tasks
        handler = logging.FileHandler(self.config.failed_learning_log_path)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _load_prompts(self) -> None:
        """Load prompt templates from files."""
        try:
            # Load experience distillation prompt
            experience_prompt_path = (
                Path(__file__).parent / "prompts" / "experience_distillation.txt"
            )
            self.experience_distillation_prompt = experience_prompt_path.read_text(
                encoding="utf-8"
            )

            # Load keyword extraction prompt
            keyword_prompt_path = (
                Path(__file__).parent / "prompts" / "keyword_extraction.txt"
            )
            self.keyword_extraction_prompt = keyword_prompt_path.read_text(
                encoding="utf-8"
            )

        except Exception as e:
            raise IngestionError(f"Failed to load prompt templates: {e}") from e

    def _generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for the given text using an OpenAI-compatible embedding service.

        Args:
            text: Text to embed

        Returns:
            List of embedding values

        Raises:
            IngestionError: If embedding generation fails
        """
        try:
            client = self.config.get_embedding_client()
            response = client.embeddings.create(
                model=self.config.embedding_model, input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise IngestionError(f"Failed to generate embedding: {e}") from e

    def _extract_keywords_with_jieba(self, text: str) -> list[str]:
        """
        Extract keywords from text using jieba tokenizer.

        Args:
            text: Text to tokenize

        Returns:
            List of keywords
        """
        # Tokenize the text
        tokens = jieba.lcut(text, cut_all=False)

        # Filter out short tokens and common stop words
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
        keywords = [
            token.lower()
            for token in tokens
            if len(token) > 1 and token.lower() not in stop_words
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)

        return unique_keywords[:10]  # Limit to 10 keywords

    def _extract_keywords_with_llm(self, query: str) -> list[str]:
        """
        Extract keywords using LLM for better quality.

        Args:
            query: Query text to extract keywords from

        Returns:
            List of extracted keywords
        """
        try:
            client = self.config.get_experience_llm_client()

            prompt = self.keyword_extraction_prompt.format(query=query)

            response = client.chat.completions.create(
                model=self.config.experience_llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a keyword extraction expert.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            result = response.choices[0].message.content
            if result is None:
                raise IngestionError("LLM returned empty response")

            result = result.strip()
            keywords = json.loads(result)

            return keywords if isinstance(keywords, list) else []

        except Exception as e:
            # Fallback to jieba if LLM fails
            self.logger.warning(f"LLM keyword extraction failed, using jieba: {e}")
            return self._extract_keywords_with_jieba(query)

    def _distill_experience_with_llm(
        self, learning_request: LearningRequest
    ) -> dict[str, Any]:
        """
        Use LLM to distill raw history into structured experience.

        Args:
            learning_request: Request containing raw history and metadata

        Returns:
            Dictionary containing distilled experience data

        Raises:
            IngestionError: If distillation fails
        """
        try:
            client = self.config.get_experience_llm_client()

            # Ensure all parameters are strings to avoid formatting issues
            task_desc = learning_request.task_description or "Not specified"
            app_name = learning_request.app_name or "Unknown application"

            prompt = self.experience_distillation_prompt.format(
                task_description=task_desc,
                app_name=app_name,
                is_successful=learning_request.is_successful,
                raw_history=json.dumps(learning_request.raw_history, indent=2),
            )

            response = client.chat.completions.create(
                model=self.config.experience_llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing GUI automation tasks.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            result = response.choices[0].message.content
            if result is None:
                raise IngestionError("LLM returned empty response")

            result = result.strip()

            # Try to extract JSON from the response
            if "```json" in result:
                json_start = result.find("```json") + 7
                json_end = result.find("```", json_start)
                json_str = result[json_start:json_end].strip()
            else:
                json_str = result

            return json.loads(json_str)

        except Exception as e:
            raise IngestionError(f"Failed to distill experience with LLM: {e}") from e

    def learn_from_task(
        self,
        raw_history: list[dict[str, Any]],
        is_successful: bool,
        source_task_id: str,
        app_name: str = "",
        task_description: str = "",
    ) -> str:
        """
        Learn from task execution history (V1.0 temporary interface).

        Args:
            raw_history: Raw operational history from task execution
            is_successful: Whether the task was completed successfully
            source_task_id: Unique identifier for the source task
            app_name: Name of the application being operated on
            task_description: Optional description of the task

        Returns:
            Success message with details

        Raises:
            IngestionError: If learning process fails
        """
        try:
            # Check for idempotency - skip if already exists
            if self.storage.experience_exists(source_task_id):
                return f"Experience with source_task_id '{source_task_id}' already exists, skipping."

            # Create learning request
            learning_request = LearningRequest(
                raw_history=raw_history,
                is_successful=is_successful,
                source_task_id=source_task_id,
                app_name=app_name,
                task_description=task_description,
            )

            # Distill experience using LLM
            distilled_data = self._distill_experience_with_llm(learning_request)

            # Create action steps
            action_steps = []
            for step_data in distilled_data.get("action_flow", []):
                action_step = ActionStep(
                    thought=step_data.get("thought", ""),
                    action=step_data.get("action", ""),
                    target_element_description=step_data.get(
                        "target_element_description", ""
                    ),
                )
                action_steps.append(action_step)

            # Add app_name to keywords if provided
            keywords = distilled_data.get("keywords", [])
            if app_name and app_name.lower() not in [k.lower() for k in keywords]:
                keywords.insert(0, app_name.lower())

            # Create experience record
            experience = ExperienceRecord(
                task_description=distilled_data.get(
                    "task_description", task_description
                ),
                keywords=keywords,
                action_flow=action_steps,
                preconditions=distilled_data.get("preconditions", ""),
                is_successful=is_successful,
                source_task_id=source_task_id,
            )

            # Generate embedding for the task description
            embedding = self._generate_embedding(experience.task_description)

            # Store the experience
            record_ids = self.storage.add_experiences([experience], [embedding])

            return f"Successfully learned experience from task '{source_task_id}'. Record ID: {record_ids[0]}"

        except Exception as e:
            # Log the failure
            failure_data = {
                "timestamp": datetime.now().isoformat(),
                "source_task_id": source_task_id,
                "error": str(e),
                "raw_history": raw_history,
                "is_successful": is_successful,
                "app_name": app_name,
                "task_description": task_description,
            }

            self.logger.error(f"Failed to learn from task: {json.dumps(failure_data)}")
            logger.error(f"Failed to learn from task: {json.dumps(failure_data)}")

            raise IngestionError(
                f"Failed to learn from task '{source_task_id}': {e}"
            ) from e

    def add_experience(self, experience: ExperienceRecord) -> str:
        """
        Add a pre-structured experience record (future-facing interface).

        Args:
            experience: Complete experience record to add

        Returns:
            Success message with record ID

        Raises:
            IngestionError: If adding experience fails
        """
        try:
            # Check for idempotency
            if self.storage.experience_exists(experience.source_task_id):
                return f"Experience with source_task_id '{experience.source_task_id}' already exists, skipping."

            # Generate embedding
            embedding = self._generate_embedding(experience.task_description)

            # Store the experience
            record_ids = self.storage.add_experiences([experience], [embedding])

            return f"Successfully added experience '{experience.source_task_id}'. Record ID: {record_ids[0]}"

        except Exception as e:
            raise IngestionError(f"Failed to add experience: {e}") from e

    def add_fact(
        self, content: str, keywords: list[str], source: str = "manual"
    ) -> str:
        """
        Add a semantic fact to the knowledge base.

        Args:
            content: The factual content to add
            keywords: List of keywords for retrieval
            source: Source of this knowledge

        Returns:
            Success message with record ID

        Raises:
            IngestionError: If adding fact fails
        """
        try:
            # Create fact record
            fact = FactRecord(content=content, keywords=keywords, source=source)

            # Generate embedding
            embedding = self._generate_embedding(content)

            # Store the fact
            record_ids = self.storage.add_facts([fact], [embedding])

            return f"Successfully added fact. Record ID: {record_ids[0]}"

        except Exception as e:
            raise IngestionError(f"Failed to add fact: {e}") from e

    def batch_add_facts(self, facts_data: list[dict[str, Any]]) -> list[str]:
        """
        Add multiple facts in batch.

        Args:
            facts_data: List of dictionaries containing fact data

        Returns:
            List of success messages

        Raises:
            IngestionError: If batch operation fails
        """
        try:
            # Validate all facts first
            for i, fact_data in enumerate(facts_data):
                if not fact_data.get("content", "").strip():
                    raise IngestionError(
                        f"Content cannot be empty for fact at index {i}"
                    )

            facts = []
            embeddings = []

            for fact_data in facts_data:
                fact = FactRecord(
                    content=fact_data["content"],
                    keywords=fact_data.get("keywords", []),
                    source=fact_data.get("source", "batch_import"),
                )
                facts.append(fact)

                # Generate embedding
                embedding = self._generate_embedding(fact.content)
                embeddings.append(embedding)

            # Store all facts
            record_ids = self.storage.add_facts(facts, embeddings)

            return [f"Successfully added fact. Record ID: {rid}" for rid in record_ids]

        except Exception as e:
            # Handle storage-specific errors with more specific messages
            if "storage" in str(e).lower() or "StorageError" in str(type(e).__name__):
                raise IngestionError(f"Failed to add facts to storage: {e}") from e
            raise IngestionError(f"Failed to batch add facts: {e}") from e
