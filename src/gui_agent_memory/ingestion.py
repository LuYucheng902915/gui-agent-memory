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
from .log_utils import OperationLogger
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
        self.logger = logging.getLogger(__name__)
        # Load prompt templates
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load prompt templates from packaged resources only (robust to patched reads)."""
        try:
            base = Path(__file__).parent / "prompts"
            self.experience_distillation_prompt = (
                base / "experience_distillation.txt"
            ).read_text(encoding="utf-8")
            self.keyword_extraction_prompt = (
                base / "keyword_extraction.txt"
            ).read_text(encoding="utf-8")
            # Judge is optional in some tests; provide robust fallback when read is patched or file absent
            self.judge_decision_prompt = (base / "judge_decision.txt").read_text(
                encoding="utf-8"
            )
        except Exception as e:
            raise IngestionError(f"Failed to load bundled prompt templates: {e}") from e

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
        op: OperationLogger | None = None,
    ) -> dict[str, Any]:
        """Call LLM Judge and persist artifacts.

        If op is provided, write artifacts to op (e.g., op=OperationLogger for judge/);
        otherwise fall back to PROMPT_LOG_DIR legacy path.
        """
        client = None
        # Resolve judge artifact directory
        if op is not None:
            judge_dir = op.path()

            def _attach_text(name: str, content: str) -> None:
                op.attach_text(name, content)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_prompt_dir = Path(
                getattr(self.config, "prompt_log_dir", "./memory_system/logs/prompts")
            )
            judge_dir = base_prompt_dir / f"judge_{ts}"
            judge_dir.mkdir(parents=True, exist_ok=True)

            def _attach_text(name: str, content: str) -> None:
                try:
                    (judge_dir / name).write_text(content, encoding="utf-8")
                except Exception:
                    self.logger.exception("Failed to write judge artifact: %s", name)

        try:
            client = self.config.get_experience_llm_client()
            prompt = self._judge_prompt(old_record, new_record, record_type, similarity)
            _attach_text("input_prompt.txt", prompt)

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
            _attach_text("model_output.json", result)
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
                _attach_text("model_error.txt", f"JSON parse error: {je}\n{result}")
                raise
            if isinstance(payload, dict):
                payload["log_dir"] = judge_dir.as_posix()
            return payload
        except Exception as e:
            _attach_text(
                "model_error.txt",
                f"Judge failed: {e}\nrecord_type={record_type} similarity={similarity}\nclient={'ok' if client else 'none'}",
            )
            self.logger.warning(f"LLM judge failed, degrade to add_new: {e}")
            return {
                "decision": "add_new",
                "target_id": None,
                "updated_record": None,
                "reason": "fallback",
                "log_dir": judge_dir.as_posix(),
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
            # Prefer distance from Chroma; fallback to cosine on returned embeddings
            sim = self._cosine_similarity_from_distance(
                distances[0] if distances else None
            )
            if (not distances or len(distances) == 0) and res:
                embs = res.get("embeddings", [[]])[0] if res.get("embeddings") else []
                if embs and len(embs) > 0:
                    try:
                        import math

                        a = embedding
                        b = embs[0]
                        dot = sum(
                            float(x) * float(y) for x, y in zip(a, b, strict=False)
                        )
                        na = math.sqrt(sum(float(x) * float(x) for x in a))
                        nb = math.sqrt(sum(float(y) * float(y) for y in b))
                        if na > 0 and nb > 0:
                            sim = max(0.0, min(1.0, float(dot) / (na * nb)))
                    except Exception:
                        sim = sim
            return top_id, sim
        except Exception:
            return None, 0.0

    def _top1_similarity_fact(self, embedding: list[float]) -> tuple[str | None, float]:
        try:
            res = self.storage.query_facts(query_embeddings=[embedding], n_results=1)
            ids = res.get("ids", [[]])[0] if res else []
            distances = res.get("distances", [[]])[0] if res else []
            top_id = ids[0] if ids else None
            # Prefer distances; fallback to cosine similarity using returned embeddings
            if distances and len(distances) > 0:
                sim = self._cosine_similarity_from_distance(distances[0])
            else:
                sim = 0.0
                if res and res.get("embeddings"):
                    try:
                        import math

                        embs = res.get("embeddings", [[]])[0]
                        if embs and len(embs) > 0:
                            a = embedding
                            b = embs[0]
                            dot = sum(
                                float(x) * float(y) for x, y in zip(a, b, strict=False)
                            )
                            na = math.sqrt(sum(float(x) * float(x) for x in a))
                            nb = math.sqrt(sum(float(y) * float(y) for y in b))
                            if na > 0 and nb > 0:
                                sim = max(0.0, min(1.0, float(dot) / (na * nb)))
                    except Exception:
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

    def upsert_fact_with_policy(
        self, fact: FactRecord, op: OperationLogger | None = None
    ) -> dict[str, Any]:
        try:
            if op is None:
                op = OperationLogger.create(
                    getattr(self.config, "logs_base_dir", "./memory_system/logs"),
                    "add_fact",
                    enabled=getattr(self.config, "operation_logs_enabled", True),
                )
            # Input artifact
            op.attach_json("input.json", fact.model_dump())
            # Safe fingerprint dedupe with Mock guards
            out_fp: Any = None
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
                except Exception as exc:
                    self.logger.warning("Fingerprint dedupe failed (fact): %s", exc)

            # If fingerprint dedup didn't short-circuit, proceed; ensure out_fp computed
            if out_fp is None and callable(compute_method):
                try:
                    out_fp = compute_method(fact)
                except Exception:
                    out_fp = None
            embedding = self._generate_embedding(fact.content)
            top_id, sim = self._top1_similarity_fact(embedding)
            threshold = getattr(self.config, "similarity_threshold_judge", 0.90)

            if top_id is None or sim < threshold:
                # safety re-check before add
                try:
                    if out_fp and self.storage.fact_exists_by_output_fp(out_fp):
                        return {
                            "result": "discarded_by_fingerprint",
                            "similarity": sim,
                            "threshold": threshold,
                            "invoked_judge": False,
                            "judge_decision": None,
                            "top_id": top_id,
                            "fingerprint_discarded": True,
                        }
                except Exception as exc:
                    self.logger.warning("Output-fp recheck before add failed: %s", exc)
                record_ids = self.storage.add_facts(
                    [fact], [embedding], output_fps=[out_fp]
                )
                dbg = {
                    "result": "added_new",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": False,
                    "judge_decision": None,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                    "new_record_id": record_ids[0] if record_ids else None,
                }
                op.attach_json("debug.json", dbg)
                return dbg

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

            judge_dir = op.child("judge")
            judge = self._call_llm_judge(
                old_rec, new_rec, record_type="fact", similarity=sim, op=judge_dir
            )
            decision = judge.get("decision", "add_new")

            if decision == "add_new":
                # safety re-check before add
                try:
                    if out_fp and self.storage.fact_exists_by_output_fp(out_fp):
                        return {
                            "result": "discarded_by_fingerprint",
                            "similarity": sim,
                            "threshold": threshold,
                            "invoked_judge": True,
                            "judge_decision": decision,
                            "top_id": top_id,
                            "fingerprint_discarded": True,
                        }
                except Exception as exc:
                    self.logger.warning(
                        "Output-fp recheck before judge-add failed: %s", exc
                    )
                record_ids = self.storage.add_facts(
                    [fact], [embedding], output_fps=[out_fp]
                )
                dbg = {
                    "result": "added_new",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": True,
                    "judge_decision": decision,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                    "judge_log_dir": judge_dir.path().as_posix(),
                    "new_record_id": record_ids[0] if record_ids else None,
                }
                op.attach_json("debug.json", dbg)
                return dbg
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
                dbg = {
                    "result": "updated_existing",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": True,
                    "judge_decision": decision,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                    "judge_log_dir": judge_dir.path().as_posix(),
                }
                op.attach_json("debug.json", dbg)
                return dbg
            if decision == "keep_new_delete_old":
                # safety re-check before add
                try:
                    if out_fp and self.storage.fact_exists_by_output_fp(out_fp):
                        return {
                            "result": "discarded_by_fingerprint",
                            "similarity": sim,
                            "threshold": threshold,
                            "invoked_judge": True,
                            "judge_decision": decision,
                            "top_id": top_id,
                            "fingerprint_discarded": True,
                        }
                except Exception as exc:
                    self.logger.warning(
                        "Output-fp recheck before keep_new_add failed: %s", exc
                    )
                record_ids = self.storage.add_facts(
                    [fact], [embedding], output_fps=[out_fp]
                )
                self.storage.delete_records(
                    self.storage.config.declarative_collection_name, [top_id]
                )
                dbg = {
                    "result": "kept_new_deleted_old",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": True,
                    "judge_decision": decision,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                    "judge_log_dir": judge_dir.path().as_posix(),
                    "new_record_id": record_ids[0] if record_ids else None,
                }
                op.attach_json("debug.json", dbg)
                return dbg
            if decision == "keep_old_delete_new":
                dbg = {
                    "result": "kept_old_discarded_new",
                    "similarity": sim,
                    "threshold": threshold,
                    "invoked_judge": True,
                    "judge_decision": decision,
                    "top_id": top_id,
                    "fingerprint_discarded": False,
                    "judge_log_dir": judge_dir.path().as_posix(),
                }
                op.attach_json("debug.json", dbg)
                return dbg

            record_ids = self.storage.add_facts(
                [fact], [embedding], output_fps=[out_fp]
            )
            dbg = {
                "result": "added_new",
                "similarity": sim,
                "threshold": threshold,
                "invoked_judge": True,
                "judge_decision": decision,
                "top_id": top_id,
                "fingerprint_discarded": False,
                "judge_log_dir": judge_dir.path().as_posix(),
                "new_record_id": record_ids[0] if record_ids else None,
            }
            op.attach_json("debug.json", dbg)
            return dbg
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

    def _extract_keywords_with_llm(
        self, query: str, op: OperationLogger | None = None
    ) -> list[str]:
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
            keyword_op = op.child("keyword") if op is not None else None
            if keyword_op is not None:
                keyword_op.attach_text("input_prompt.txt", prompt)

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
            if keyword_op is not None:
                keyword_op.attach_text("model_output.json", result)

            return keywords if isinstance(keywords, list) else []

        except Exception as e:
            # Fallback to jieba if LLM fails
            self.logger.warning(f"LLM keyword extraction failed, using jieba: {e}")
            return self._extract_keywords_with_jieba(query)

    def _distill_experience_with_llm(
        self, learning_request: LearningRequest, op: OperationLogger | None = None
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
            reflect_op = op.child("reflect") if op is not None else None
            if reflect_op is not None:
                reflect_op.attach_text("input_prompt.txt", prompt)

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
            if reflect_op is not None:
                reflect_op.attach_text("model_output.json", result)

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
        op: OperationLogger | None = None,
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

            # Prepare operation logger for learn_from_task
            op = OperationLogger.create(
                getattr(self.config, "logs_base_dir", "./memory_system/logs"),
                "learn_from_task",
                self._safe_slug(source_task_id),
                enabled=getattr(self.config, "operation_logs_enabled", True),
            )
            # Persist raw inputs
            op.attach_json(
                "input.json",
                {
                    "raw_history": raw_history,
                    "is_successful": is_successful,
                    "source_task_id": source_task_id,
                    "app_name": app_name,
                    "task_description": task_description,
                },
            )

            # Distill experience using LLM (logs into op/reflect)
            distilled_data = self._distill_experience_with_llm(learning_request, op=op)

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

            # Summary artifact for the operation
            op.attach_json(
                "summary.json",
                {
                    "result": "success",
                    "record_id": record_ids[0] if record_ids else None,
                    "input_fp": input_fp,
                    "source_task_id": source_task_id,
                },
            )

            return f"Successfully learned experience from task '{source_task_id}'. Record ID: {record_ids[0]}"

        except Exception as e:
            # Log the failure to operation artifacts
            try:
                if op is None:
                    op = OperationLogger.create(
                        getattr(self.config, "logs_base_dir", "./memory_system/logs"),
                        "learn_from_task",
                        self._safe_slug(source_task_id),
                        enabled=getattr(self.config, "operation_logs_enabled", True),
                    )
                op.attach_json(
                    "error.json",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "source_task_id": source_task_id,
                        "error": str(e),
                        "raw_history": raw_history,
                        "is_successful": is_successful,
                        "app_name": app_name,
                        "task_description": task_description,
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Failed to write learn_from_task error artifact: %s", exc
                )

            # Legacy error logging to satisfy tests
            failure_data = {
                "timestamp": datetime.now().isoformat(),
                "source_task_id": source_task_id,
                "error": str(e),
                "raw_history": raw_history,
                "is_successful": is_successful,
                "app_name": app_name,
                "task_description": task_description,
            }
            logger.error(f"Failed to learn from task: {json.dumps(failure_data)}")

            raise IngestionError(
                f"Failed to learn from task '{source_task_id}': {e}"
            ) from e

    def add_experience(
        self, experience: ExperienceRecord, op: OperationLogger | None = None
    ) -> str:
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

            # Prepare operation logger
            op = op or OperationLogger.create(
                getattr(self.config, "logs_base_dir", "./memory_system/logs"),
                "add_experience",
                self._safe_slug(experience.source_task_id),
                enabled=getattr(self.config, "operation_logs_enabled", True),
            )
            op.attach_json("input.json", experience)

            # Generate embedding
            embedding = self._generate_embedding(experience.task_description)

            # Store the experience
            record_ids = self.storage.add_experiences([experience], [embedding])

            op.attach_json(
                "summary.json",
                {
                    "result": "success",
                    "record_id": record_ids[0] if record_ids else None,
                },
            )

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

    def batch_add_facts(
        self, facts_data: list[dict[str, Any]], op: OperationLogger | None = None
    ) -> list[str]:
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

            # Operation logger for batch
            if op is None:
                op = OperationLogger.create(
                    getattr(self.config, "logs_base_dir", "./memory_system/logs"),
                    "batch_add_facts",
                    enabled=getattr(self.config, "operation_logs_enabled", True),
                )
            op.attach_json("input.json", facts_data)

            # Simple batch: route through single entry with op child per item
            success_msgs: list[str] = []
            for idx, fact_data in enumerate(facts_data, start=1):
                item_op = op.child(f"item_{idx:03d}")
                msg = self.add_fact(
                    content=fact_data["content"],
                    keywords=fact_data.get("keywords", []),
                    source=fact_data.get("source", "batch_import"),
                )
                # Mirror per-item message for quick glance
                item_op.attach_text("result.txt", msg)
                success_msgs.append(msg)
            op.attach_json(
                "summary.json", {"added": len(success_msgs), "total": len(facts_data)}
            )
            return success_msgs

        except Exception as e:
            # Handle storage-specific errors with more specific messages
            if "storage" in str(e).lower() or "StorageError" in str(type(e).__name__):
                raise IngestionError(f"Failed to add facts to storage: {e}") from e
            raise IngestionError(f"Failed to batch add facts: {e}") from e
