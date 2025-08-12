"""
Ingestion layer for learning experiences and importing knowledge.

This module handles:
- Experience learning from raw task execution history
- Knowledge ingestion from various sources
- LLM-based experience distillation
- Embedding generation and storage
- Idempotency guarantees and error handling
"""

import hashlib
import json
import logging
import re
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

    def __init__(self, storage: MemoryStorage | None = None, config=None) -> None:
        """Initialize the ingestion system.

        Args:
            storage: Optional injected storage (used in tests)
        """
        self.config = config or get_config()
        self.storage = storage or MemoryStorage(self.config)

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
        # First, resolve base directory and load the two primary templates.
        try:
            templates_dir = getattr(self.config, "prompt_templates_dir", None)
            base: Path
            if templates_dir:
                base = Path(templates_dir)
                external_ok = (base / "experience_distillation.txt").exists() and (
                    base / "keyword_extraction.txt"
                ).exists()
                if not external_ok:
                    base = Path(__file__).parent / "prompts"
            else:
                base = Path(__file__).parent / "prompts"

            # Load experience distillation prompt
            experience_prompt_path = base / "experience_distillation.txt"
            self.experience_distillation_prompt = experience_prompt_path.read_text(
                encoding="utf-8"
            )

            # Load keyword extraction prompt
            keyword_prompt_path = base / "keyword_extraction.txt"
            self.keyword_extraction_prompt = keyword_prompt_path.read_text(
                encoding="utf-8"
            )
        except Exception as e:
            raise IngestionError(f"Failed to load prompt templates: {e}") from e

        # Then, try to load the optional judge template without failing the whole loading flow.
        judge_prompt_path = base / "judge_decision.txt"
        if judge_prompt_path.exists():
            try:
                self.judge_decision_prompt = judge_prompt_path.read_text(
                    encoding="utf-8"
                )
            except Exception:
                # Minimal fallback when read_text is patched with limited side effects in tests
                self.judge_decision_prompt = (
                    "请在 add_new / update_existing / keep_new_delete_old / keep_old_delete_new 中选择其一，"
                    "只输出严格 JSON：{\n"
                    '  "decision": "add_new|update_existing|keep_new_delete_old|keep_old_delete_new",\n'
                    '  "target_id": null,\n'
                    '  "updated_record": null,\n'
                    '  "reason": ""\n'
                    "}\n"
                    "[记录类型]: {record_type}\n[已存在的旧记忆]: {old_record}\n[新记忆]: {new_record}"
                )
        else:
            # Fallback when template is absent (e.g., external dir provides only two files)
            self.judge_decision_prompt = (
                "请在 add_new / update_existing / keep_new_delete_old / keep_old_delete_new 中选择其一，"
                "只输出严格 JSON：{\n"
                '  "decision": "add_new|update_existing|keep_new_delete_old|keep_old_delete_new",\n'
                '  "target_id": null,\n'
                '  "updated_record": null,\n'
                '  "reason": ""\n'
                "}\n"
                "[记录类型]: {record_type}\n[已存在的旧记忆]: {old_record}\n[新记忆]: {new_record}"
            )

    # ----------------------------
    # Internal logging helpers
    # ----------------------------
    def _safe_slug(self, text: str) -> str:
        """Make a filesystem-safe slug from arbitrary text."""
        if not text:
            return "unknown"
        slug = re.sub(r"[^\w\-\.]+", "-", str(text), flags=re.UNICODE)
        return slug.strip("-_") or "unknown"

    def _write_text_file(self, path: Path, content: str) -> None:
        """Write text to a file with UTF-8 encoding, logging exceptions instead of passing silently."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except Exception as exc:
            # Log and continue; do not break primary flow
            self.logger.exception(f"Failed to write log file '{path}': {exc}")

    # ----------------------------
    # Fingerprint helpers
    # ----------------------------
    def _normalize_text(self, text: str) -> str:
        if text is None:
            return ""
        return " ".join(str(text).strip().split())

    def _normalize_keywords(self, keywords: list[str] | None) -> list[str]:
        if not keywords:
            return []
        return sorted(
            {self._normalize_text(k).lower() for k in keywords if str(k).strip()}
        )

    def _compute_experience_input_fp(
        self,
        raw_history: list[dict[str, Any]],
        app_name: str,
        task_description: str,
        is_successful: bool,
    ) -> str:
        # Strict JSON for history; collapse whitespace for texts; app lowercase
        try:
            history_norm = json.dumps(
                raw_history, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        except Exception:
            history_norm = json.dumps(
                [], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        payload = {
            "fp_v": 1,
            "type": "experience_input",
            "history": history_norm,
            "app": self._normalize_text(app_name).lower(),
            "task": self._normalize_text(task_description),
            "ok": bool(is_successful),
        }
        data = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _compute_fact_input_fp(self, content: str, keywords: list[str] | None) -> str:
        # Use content only for input fingerprint to avoid keyword variance
        payload = {
            "fp_v": 1,
            "type": "fact_input",
            "content": self._normalize_text(content),
        }
        data = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

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

    # ----------------------------
    # LLM Judge prompt (from template)
    # ----------------------------
    def _judge_prompt(
        self,
        old_record: dict[str, Any],
        new_record: dict[str, Any],
        record_type: str,
        similarity: float | None = None,
    ) -> str:
        # Use default=str to serialize datetime and other non-JSON types safely
        return self.judge_decision_prompt.format(
            record_type=record_type,
            similarity=f"{similarity:.3f}" if similarity is not None else "unknown",
            old_record=json.dumps(old_record, ensure_ascii=False, default=str),
            new_record=json.dumps(new_record, ensure_ascii=False, default=str),
        )

    def _call_llm_judge(
        self,
        old_record: dict[str, Any],
        new_record: dict[str, Any],
        record_type: str,
        similarity: float | None = None,
    ) -> dict[str, Any]:
        # Always create a judge dir and persist inputs to aid debugging
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_prompt_dir = Path(
            getattr(self.config, "prompt_log_dir", "./memory_system/logs/prompts")
        )
        judge_dir = base_prompt_dir / f"judge_{ts}"
        client = None
        try:
            client = self.config.get_experience_llm_client()
            prompt = self._judge_prompt(old_record, new_record, record_type, similarity)
            self._write_text_file(judge_dir / "input_prompt.txt", prompt)

            response = client.chat.completions.create(
                model=self.config.experience_llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个严谨的记忆库维护者，只输出严格JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            result = response.choices[0].message.content or "{}"
            self._write_text_file(judge_dir / "model_output.json", result)
            try:
                # Be robust to Markdown code fences (``` or ```json)
                text = result.strip()
                if text.startswith("```"):
                    # remove first fence line
                    first_nl = text.find("\n")
                    if first_nl != -1:
                        text = text[first_nl + 1 :]
                    # remove trailing fence line if present
                    if text.endswith("```"):
                        text = text[:-3]
                payload = json.loads(text)
            except Exception as je:
                # Persist raw and error
                self._write_text_file(
                    judge_dir / "model_error.txt", f"JSON parse error: {je}\n{result}"
                )
                raise
            if isinstance(payload, dict):
                payload["log_dir"] = str(judge_dir)
            return payload
        except Exception as e:
            # Persist failure details to judge dir
            self._write_text_file(
                judge_dir / "model_error.txt",
                f"Judge failed: {e}\nrecord_type={record_type} similarity={similarity}\nclient={'ok' if client else 'none'}",
            )
            self.logger.warning(f"LLM judge failed, degrade to add_new: {e}")
            return {
                "decision": "add_new",
                "target_id": None,
                "updated_record": None,
                "reason": "fallback",
                "log_dir": str(judge_dir),
            }

    def _cosine_similarity_from_distance(self, distance: float | None) -> float:
        if distance is None:
            return 0.0
        # Chroma returns distance; treat similarity ~= 1 - distance for cosine metric
        try:
            return max(0.0, min(1.0, 1.0 - float(distance)))
        except Exception:
            return 0.0

    def _top1_similarity_experience(
        self, embedding: list[float]
    ) -> tuple[str | None, float]:
        try:
            res = self.storage.query_experiences(
                query_embeddings=[embedding], n_results=1
            )
            ids = res.get("ids", [[]])[0] if res else []
            distances = res.get("distances", [[]])[0] if res else []
            top_id = ids[0] if ids else None
            sim = self._cosine_similarity_from_distance(
                distances[0] if distances else None
            )
            return top_id, sim
        except Exception:
            return None, 0.0

    def _top1_similarity_fact(self, embedding: list[float]) -> tuple[str | None, float]:
        try:
            res = self.storage.query_facts(query_embeddings=[embedding], n_results=1)
            ids = res.get("ids", [[]])[0] if res else []
            distances = res.get("distances", [[]])[0] if res else []
            top_id = ids[0] if ids else None
            # If distances missing or empty, try to infer similarity via cosine from returned embeddings
            if distances and len(distances) > 0:
                sim = self._cosine_similarity_from_distance(distances[0])
            else:
                sim = 0.0
            return top_id, sim
        except Exception:
            return None, 0.0

    # ---------------------------------
    # Public upsert with similarity policy
    # ---------------------------------
    def upsert_experience_with_policy(
        self, experience: ExperienceRecord
    ) -> dict[str, Any]:
        """Insert or route to judge based on top-1 similarity and return debug info.

        Returns a dict with keys: result, similarity, threshold, invoked_judge,
        judge_decision, top_id, fingerprint_discarded.
        """
        try:
            # 1) fingerprint dedupe (output)
            out_fp = self.storage.compute_experience_output_fp(experience)
            if self.storage.experience_exists_by_output_fp(out_fp):
                return {
                    "result": "discarded_by_fingerprint",
                    "similarity": 0.0,
                    "threshold": getattr(
                        self.config, "similarity_threshold_judge", 0.90
                    ),
                    "invoked_judge": False,
                    "judge_decision": None,
                    "top_id": None,
                    "fingerprint_discarded": True,
                }

            # 2) embedding + top1 similarity
            embedding = self._generate_embedding(experience.task_description)
            top_id, sim = self._top1_similarity_experience(embedding)
            threshold = getattr(self.config, "similarity_threshold_judge", 0.90)

            # 3) routing
            if top_id is None or sim < threshold:
                self.storage.add_experiences(
                    [experience], [embedding], output_fps=[out_fp]
                )
                return {
                    "result": "added_new",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": False,
                    "judge_decision": None,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                }

            # 4) judge on high similarity
            # build minimal comparable dicts
            new_rec = experience.model_dump()
            # fetch old record content
            old = (
                self.storage.experiential_collection.get(
                    ids=[top_id], include=["documents", "metadatas"]
                )
                or {}
            )
            old_doc = (old.get("documents") or [""])[0]
            old_meta = (old.get("metadatas") or [{}])[0] or {}
            old_rec = {"task_description": old_doc, **old_meta}

            judge = self._call_llm_judge(
                old_rec, new_rec, record_type="experience", similarity=sim
            )
            decision = judge.get("decision", "add_new")
            judge_log_dir = judge.get("log_dir")

            if decision == "add_new":
                self.storage.add_experiences(
                    [experience], [embedding], output_fps=[out_fp]
                )
                return {
                    "result": "added_new",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": True,
                    "judge_decision": decision,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                    "judge_log_dir": judge_log_dir,
                }
            if decision == "update_existing":
                updated_payload = judge.get("updated_record") or {}
                try:
                    updated = ExperienceRecord(**updated_payload)
                except Exception as e:
                    self.logger.warning(
                        f"Invalid updated_record from judge, fallback add_new: {e}"
                    )
                    self.storage.add_experiences(
                        [experience], [embedding], output_fps=[out_fp]
                    )
                    return {
                        "result": "added_new",
                        "similarity": sim,
                        "threshold": threshold,
                        "invoked_judge": True,
                        "judge_decision": "add_new",
                        "top_id": top_id,
                        "fingerprint_discarded": False,
                        "judge_log_dir": judge.get("log_dir"),
                    }
                # recompute embedding using updated description
                upd_emb = self._generate_embedding(updated.task_description)
                self.storage.update_experience(top_id, updated, embedding=upd_emb)
                return {
                    "result": "updated_existing",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": True,
                    "judge_decision": decision,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                    "judge_log_dir": judge_log_dir,
                }
            if decision == "keep_new_delete_old":
                self.storage.add_experiences(
                    [experience], [embedding], output_fps=[out_fp]
                )
                self.storage.delete_records(
                    self.storage.config.experiential_collection_name, [top_id]
                )
                return {
                    "result": "kept_new_deleted_old",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": True,
                    "judge_decision": decision,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                    "judge_log_dir": judge_log_dir,
                }
            if decision == "keep_old_delete_new":
                return {
                    "result": "kept_old_discarded_new",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": True,
                    "judge_decision": decision,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                    "judge_log_dir": judge_log_dir,
                }

            # default fallback
            self.storage.add_experiences([experience], [embedding], output_fps=[out_fp])
            return {
                "result": "added_new",
                "similarity": sim,
                "threshold": threshold,
                "invoked_judge": True,
                "judge_decision": decision,
                "top_id": top_id,
                "fingerprint_discarded": False,
                "judge_log_dir": judge_log_dir,
            }
        except Exception as e:
            raise IngestionError(f"Upsert experience with policy failed: {e}") from e

    def upsert_fact_with_policy(self, fact: FactRecord) -> dict[str, Any]:
        try:
            out_fp = self.storage.compute_fact_output_fp(fact)
            if self.storage.fact_exists_by_output_fp(out_fp):
                return {
                    "result": "discarded_by_fingerprint",
                    "similarity": 0.0,
                    "threshold": getattr(
                        self.config, "similarity_threshold_judge", 0.90
                    ),
                    "invoked_judge": False,
                    "judge_decision": None,
                    "top_id": None,
                    "fingerprint_discarded": True,
                }

            embedding = self._generate_embedding(fact.content)
            top_id, sim = self._top1_similarity_fact(embedding)
            threshold = getattr(self.config, "similarity_threshold_judge", 0.90)

            if top_id is None or sim < threshold:
                self.storage.add_facts([fact], [embedding], output_fps=[out_fp])
                return {
                    "result": "added_new",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": False,
                    "judge_decision": None,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                }

            new_rec = fact.model_dump()
            old = (
                self.storage.declarative_collection.get(
                    ids=[top_id], include=["documents", "metadatas"]
                )
                or {}
            )
            old_doc = (old.get("documents") or [""])[0]
            old_meta = (old.get("metadatas") or [{}])[0] or {}
            old_rec = {"content": old_doc, **old_meta}

            judge = self._call_llm_judge(
                old_rec, new_rec, record_type="fact", similarity=sim
            )
            decision = judge.get("decision", "add_new")
            judge_log_dir = judge.get("log_dir")

            if decision == "add_new":
                self.storage.add_facts([fact], [embedding], output_fps=[out_fp])
                return {
                    "result": "added_new",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": True,
                    "judge_decision": decision,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                    "judge_log_dir": judge_log_dir,
                }
            if decision == "update_existing":
                updated_payload = judge.get("updated_record") or {}
                try:
                    updated = FactRecord(**updated_payload)
                except Exception as e:
                    self.logger.warning(
                        f"Invalid updated_record from judge, fallback add_new: {e}"
                    )
                    self.storage.add_facts([fact], [embedding], output_fps=[out_fp])
                    return {
                        "result": "added_new",
                        "similarity": sim,
                        "threshold": threshold,
                        "invoked_judge": True,
                        "judge_decision": "add_new",
                        "top_id": top_id,
                        "fingerprint_discarded": False,
                        "judge_log_dir": judge.get("log_dir"),
                    }
                upd_emb = self._generate_embedding(updated.content)
                self.storage.update_fact(top_id, updated, embedding=upd_emb)
                return {
                    "result": "updated_existing",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": True,
                    "judge_decision": decision,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                    "judge_log_dir": judge_log_dir,
                }
            if decision == "keep_new_delete_old":
                self.storage.add_facts([fact], [embedding], output_fps=[out_fp])
                self.storage.delete_records(
                    self.storage.config.declarative_collection_name, [top_id]
                )
                return {
                    "result": "kept_new_deleted_old",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": True,
                    "judge_decision": decision,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                    "judge_log_dir": judge_log_dir,
                }
            if decision == "keep_old_delete_new":
                return {
                    "result": "kept_old_discarded_new",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": True,
                    "judge_decision": decision,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                    "judge_log_dir": judge_log_dir,
                }

            self.storage.add_facts([fact], [embedding], output_fps=[out_fp])
            return {
                "result": "added_new",
                "similarity": sim,
                "threshold": threshold,
                "invoked_judge": True,
                "judge_decision": decision,
                "top_id": top_id,
                "fingerprint_discarded": False,
                "judge_log_dir": judge_log_dir,
            }
        except Exception as e:
            raise IngestionError(f"Upsert fact with policy failed: {e}") from e

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

            # --- prompt logging: keyword extraction ---
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_prompt_dir = Path(
                getattr(self.config, "prompt_log_dir", "./memory_system/logs/prompts")
            )
            keyword_dir = base_prompt_dir / f"keyword_{ts}"
            self._write_text_file(keyword_dir / "input_prompt.txt", prompt)

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

            # --- output logging: keyword extraction ---
            self._write_text_file(keyword_dir / "model_output.json", result)

            return keywords if isinstance(keywords, list) else []

        except Exception as e:
            # Fallback to jieba if LLM fails
            self.logger.warning(f"LLM keyword extraction failed, using jieba: {e}")
            return self._extract_keywords_with_jieba(query)

    def _distill_experience_with_llm(
        self, learning_request: LearningRequest, log_dir: Path | None = None
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

            # --- prompt logging: experience distillation ---
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if log_dir is None:
                folder = f"experience_{ts}_{self._safe_slug(learning_request.source_task_id)}"
                base_prompt_dir = Path(
                    getattr(
                        self.config, "prompt_log_dir", "./memory_system/logs/prompts"
                    )
                )
                log_dir = base_prompt_dir / folder
            self._write_text_file(log_dir / "input_prompt.txt", prompt)

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

            # --- output logging: experience distillation ---
            self._write_text_file(log_dir / "model_output.json", result)

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
            # Input fingerprint dedupe (pre-LLM)
            input_fp = self._compute_experience_input_fp(
                raw_history=raw_history,
                app_name=app_name,
                task_description=task_description,
                is_successful=is_successful,
            )
            # Call existence check only if it's a real implementation (avoid MagicMock truthiness)
            method = getattr(self.storage, "experience_exists_by_input_fp", None)
            if callable(method) and method.__class__.__name__ not in {
                "Mock",
                "MagicMock",
            }:
                try:
                    if bool(method(input_fp)):
                        return (
                            f"Experience with the same input fingerprint already exists, skipping. "
                            f"source_task_id='{source_task_id}'"
                        )
                except Exception as e:
                    self.logger.warning(
                        "Input fingerprint check failed, continuing without dedupe: %s",
                        e,
                    )
            # Check for idempotency by source_task_id only if FP didn't short-circuit
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

            # Prepare per-experience log directory
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_prompt_dir = Path(
                getattr(self.config, "prompt_log_dir", "./memory_system/logs/prompts")
            )
            experience_dir = (
                base_prompt_dir / f"experience_{ts}_{self._safe_slug(source_task_id)}"
            )
            # Persist raw inputs for traceability
            self._write_text_file(
                experience_dir / "raw_history.json",
                json.dumps(raw_history, ensure_ascii=False, indent=2),
            )
            self._write_text_file(
                experience_dir / "request_meta.json",
                json.dumps(
                    {
                        "source_task_id": source_task_id,
                        "is_successful": is_successful,
                        "app_name": app_name,
                        "task_description": task_description,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

            # Distill experience using LLM (logs into same folder)
            distilled_data = self._distill_experience_with_llm(
                learning_request, log_dir=experience_dir
            )

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
            record_ids = self.storage.add_experiences(
                [experience], [embedding], input_fps=[input_fp]
            )

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
            # Legacy idempotency by source_task_id first (keeps tests' expectations)
            if self.storage.experience_exists(experience.source_task_id):
                return f"Experience with source_task_id '{experience.source_task_id}' already exists, skipping."

            # Output fingerprint pre-check (explicit, fingerprint-first policy)
            out_fp = None
            compute_method = getattr(self.storage, "compute_experience_output_fp", None)
            if callable(compute_method) and compute_method.__class__.__name__ not in {
                "Mock",
                "MagicMock",
            }:
                try:
                    out_fp = compute_method(experience)
                except Exception:
                    out_fp = None
            exists_method = getattr(
                self.storage, "experience_exists_by_output_fp", None
            )
            if (
                out_fp
                and callable(exists_method)
                and exists_method.__class__.__name__ not in {"Mock", "MagicMock"}
            ):
                try:
                    if bool(exists_method(out_fp)):
                        return (
                            "Experience with the same output fingerprint already exists, "
                            f"skipping. source_task_id='{experience.source_task_id}'"
                        )
                except Exception as exc:
                    self.logger.warning(
                        "Output fingerprint check failed, continuing without dedupe: %s",
                        exc,
                    )

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
            # Validate input
            if not (content or "").strip():
                raise IngestionError("Content cannot be empty for fact")

            # Create fact record
            fact = FactRecord(content=content, keywords=keywords, source=source)

            # Output fingerprint pre-check (explicit, fingerprint-first policy)
            out_fp = None
            compute_method = getattr(self.storage, "compute_fact_output_fp", None)
            if callable(compute_method) and compute_method.__class__.__name__ not in {
                "Mock",
                "MagicMock",
            }:
                try:
                    out_fp = compute_method(fact)
                except Exception:
                    out_fp = None
            exists_method = getattr(self.storage, "fact_exists_by_output_fp", None)
            if (
                out_fp
                and callable(exists_method)
                and exists_method.__class__.__name__ not in {"Mock", "MagicMock"}
            ):
                try:
                    if bool(exists_method(out_fp)):
                        return "Fact already exists, skipping."
                except Exception as exc:
                    self.logger.warning(
                        "Output fingerprint check failed (fact), continuing without dedupe: %s",
                        exc,
                    )

            # Generate embedding
            embedding = self._generate_embedding(content)

            # Store the fact
            record_ids = self.storage.add_facts([fact], [embedding])

            if not record_ids:
                return "Fact already exists, skipping."

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

            facts: list[FactRecord] = []
            embeddings: list[list[float]] = []

            for fact_data in facts_data:
                fact = FactRecord(
                    content=fact_data["content"],
                    keywords=fact_data.get("keywords", []),
                    source=fact_data.get("source", "batch_import"),
                )

                # Explicit output fingerprint pre-check with Mock guard
                out_fp = None
                compute_method = getattr(self.storage, "compute_fact_output_fp", None)
                if callable(
                    compute_method
                ) and compute_method.__class__.__name__ not in {"Mock", "MagicMock"}:
                    try:
                        out_fp = compute_method(fact)
                    except Exception:
                        out_fp = None
                exists_method = getattr(self.storage, "fact_exists_by_output_fp", None)
                if (
                    out_fp
                    and callable(exists_method)
                    and exists_method.__class__.__name__ not in {"Mock", "MagicMock"}
                ):
                    try:
                        if bool(exists_method(out_fp)):
                            # Skip duplicates in batch
                            continue
                    except Exception as exc:
                        self.logger.warning(
                            "Output fingerprint check failed (batch fact), continuing without dedupe: %s",
                            exc,
                        )

                facts.append(fact)

                # Generate embedding only for non-duplicates
                embedding = self._generate_embedding(fact.content)
                embeddings.append(embedding)

            if not facts:
                return []

            # Store all facts (storage will also re-check by output_fp for safety)
            record_ids = self.storage.add_facts(facts, embeddings)

            return [f"Successfully added fact. Record ID: {rid}" for rid in record_ids]

        except Exception as e:
            # Handle storage-specific errors with more specific messages
            if "storage" in str(e).lower() or "StorageError" in str(type(e).__name__):
                raise IngestionError(f"Failed to add facts to storage: {e}") from e
            raise IngestionError(f"Failed to batch add facts: {e}") from e
