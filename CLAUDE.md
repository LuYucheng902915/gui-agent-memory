# Enhanced Memory Module (RAG-based) Technical Design (v0.1.0)

## 1. Overview

This document outlines the technical design for the Enhanced Memory Module, a component for the GUI Agent project. This module functions as a self-contained system based on Retrieval-Augmented Generation (RAG) for the long-term storage, management, and retrieval of two core types of agent memories:

1.  **Episodic Memory**: Reusable operational flows, including both successful and failed experiences, learned by the agent during task execution.
2.  **Semantic Memory**: Objective facts and rules about operating systems, specific applications, or general domains.

By combining these two types of memory, the goal is to comprehensively improve the agent's decision-making efficiency and accuracy in future tasks.

## 2. System Goals

### 2.1. Functional Goals
- **Modularity**: The memory system exists as an independent Python package, decoupled from the main Agent logic, to facilitate independent development and testing.
- **Scalability**: The architecture supports easy replacement or upgrading of core components (e.g., vector database, embedding models).
- **High Performance**: The system is designed to efficiently handle the storage and retrieval of large-scale memory data.
- **High-Quality Retrieval**: A hybrid retrieval and unified re-ranking mechanism is implemented for both memory types to ensure high relevance of retrieved results.

### 2.2. Non-Functional Goals
- **Security**: All sensitive information, such as API keys, is managed securely via environment variables and is never hardcoded.
- **Cross-Platform Compatibility**: The module is guaranteed to function correctly on both Windows and Linux platforms.

## 3. System Architecture

### 3.1. High-Level Design

The memory system consists of three core components: **Storage Layer**, **Ingestion Layer**, and **Retrieval Layer**.

- **Storage Layer**: The persistent foundation of the system, efficiently storing two types of structured memories and their vector representations in different collections within a unified database.
- **Ingestion Layer (Learning/Ingestion)**:
    - **Experience Learning**: Responsible for automatically converting the agent's raw operational history after task completion into structured "operational experiences" and storing them.
    - **Knowledge Ingestion**: Provides an interface for manually or batch-importing "semantic knowledge."
- **Retrieval Layer (Recall)**: Responsible for retrieving both types of memories in parallel based on the current task instruction and precisely extracting the most relevant results through a unified re-ranking strategy.

### 3.2. Directory Structure

All code for the memory system is located in the `gui_agent_memory` directory.

```
gui-agent-memory/
├── gui_agent_memory/
│   ├── __init__.py             # Package initializer, exports main classes
│   ├── main.py                 # Main module entry point, provides high-level APIs
│   ├── config.py               # Module configuration (models, DB paths, etc.)
│   ├── models.py               # Pydantic data models for all memory types
│   ├── storage.py              # Storage Layer: Encapsulates database interactions
│   ├── ingestion.py            # Ingestion Layer: Implements learning and ingestion flows
│   ├── retriever.py            # Retrieval Layer: Implements retrieval and re-ranking logic
│   └── prompts/                # Directory for prompt templates
│       ├── experience_distillation.txt
│       └── keyword_extraction.txt
└── tests/                      # Independent test suite for the memory module
```

### 3.3. Core Component Details

#### 3.3.1. Storage Layer (`storage.py`)

- **Technology**: `ChromaDB` is used as the local vector database.
- **Logical Partitioning**: Two independent `Collection`s are created within the database:
    - `experiential_memories`: For storing operational experiences.
    - `declarative_memories`: For storing semantic knowledge.
- **Core Data Models (`models.py`)**:
  ```python
  # Defines the structure of a single action step
  class ActionStep(BaseModel):
      thought: str
      action: str
      target_element_description: str

  # Model for episodic memory (operational experiences)
  class ExperienceRecord(BaseModel):
      task_description: str
      keywords: List[str]
      action_flow: List[ActionStep]
      preconditions: str
      is_successful: bool
      usage_count: int
      last_used_at: datetime
      source_task_id: str

  # Model for semantic memory (facts)
  class FactRecord(BaseModel):
      content: str
      keywords: List[str]
      source: str
      usage_count: int
      last_used_at: datetime

  # Model for the structured output of a retrieval operation
  class RetrievalResult(BaseModel):
      experiences: List[ExperienceRecord]
      facts: List[FactRecord]
      query: str
      total_results: int
  ```
- **Note on Reserved Fields**: The `usage_count` and `last_used_at` fields are reserved for future memory management strategies (e.g., LRU cache eviction). In `v0.1.0`, these fields are initialized with default values but are not actively used for advanced memory management. However, a basic usage tracking mechanism is implemented: when memories are retrieved via `retrieve_memories()`, only the final top-N results (after reranking) have their `usage_count` incremented and `last_used_at` updated to reflect actual usage patterns.

#### 3.3.2. Ingestion Layer (`ingestion.py`)

- **Vectorization and Metadata**:
    - **Vectorized Content**: For `ExperienceRecord`, the `task_description` is vectorized. For `FactRecord`, the `content` is vectorized.
    - **Metadata Content**: All other fields in the data objects are stored as metadata in ChromaDB.
- **Reflection Mechanism**: The `learn_from_task` interface triggers an LLM-based reflection function. It calls an independently configured "experience distillation" LLM service to refine raw history into a structured `ExperienceRecord`.
- **Idempotency**: Before writing a new `ExperienceRecord`, the system queries the database using the `source_task_id`. If the record already exists, the write operation is skipped to prevent duplicates.

#### 3.3.3. Retrieval Layer (`retriever.py`)

- **Core Strategy**: Parallel Hybrid Retrieval + Unified Re-ranking.
- **Hybrid Retrieval Mechanism**:
  1.  **Keyword Generation (Ingestion)**: The experience distillation LLM service populates the `keywords` field.
  2.  **Keyword Extraction (Retrieval)**: The `jieba` library tokenizes the input `query`.
  3.  **Parallel Retrieval**: The system performs both vector search and keyword-based filtering in parallel for both memory types.
  4.  **Result Merging and Re-ranking**: Candidate sets are merged, truncated to a predefined limit (e.g., 20), and then sorted by a Reranker model.
  5.  **Output**: The top-N most relevant memories are returned as a `RetrievalResult` object.
  6.  **Usage Tracking**: The final top-N results (after reranking) have their `usage_count` incremented and `last_used_at` timestamp updated to track actual usage patterns for future memory management.

#### 3.3.4. Dual-Interface Design for Learning

To accommodate both immediate testing needs and a future-proof architecture, the ingestion process provides two distinct APIs:

-   **`learn_from_task(...)`**: This is a high-level, temporary interface for `v0.1.0`. It accepts raw task history and internally uses the "experience distillation" LLM to perform a reflection step, converting the raw data into a structured `ExperienceRecord`. This allows for end-to-end testing of the learning flow without requiring an external reflection module.

-   **`add_experience(...)`**: This is the primary, future-facing interface. It is designed to accept a pre-structured `ExperienceRecord` object. The design anticipates that a more sophisticated, dedicated Reflection Module will eventually be developed. This external module will handle the complex process of reflection and then use this API to persist the finalized memory, decoupling the memory storage system from the reflection logic.

## 4. Module API and Integration

### 4.1. Main API (`main.py` via `MemorySystem` class)

- `retrieve_memories(query: str, top_n: int = 3) -> RetrievalResult`
- `learn_from_task(raw_history: List[dict], is_successful: bool, source_task_id: str, app_name: str = "", task_description: str = "") -> str`
- `add_experience(experience: ExperienceRecord) -> str`
- `add_fact(content: str, keywords: List[str], source: str = "manual") -> str`
- `batch_add_facts(facts_data: List[dict]) -> List[str]`
- `get_similar_experiences(task_description: str, top_n: int = 5) -> List[ExperienceRecord]`
- `get_related_facts(topic: str, top_n: int = 5) -> List[FactRecord]`

### 4.2. Integration Flow with Main Agent

#### 4.2.1. Retrieval Flow
1.  **Trigger**: In the main Agent's decision loop.
2.  **Construct Query**: The Agent constructs a descriptive query.
3.  **Call Memory Module**: `memory_system.retrieve_memories(query=...)`.
4.  **Format and Inject**: The Agent formats the `ExperienceRecord` and `FactRecord` objects from the `RetrievalResult` into a concise text format for injection into its main prompt.

#### 4.2.2. Learning Flow
1.  **Trigger**: After the main Agent completes a task.
2.  **Data Collection**: The Agent collects `raw_history` and a unique `source_task_id`.
3.  **Invocation**: Call the `learn_from_task(...)` interface.
4.  **Robustness**: All learning-related APIs include a top-level `try...except` block. On failure, the system appends the input data and error to a local log file (`./memory_system/logs/failed_learning_tasks.jsonl`).

## 5. Technology Stack and Configuration

### 5.1. Technology Stack

- **Package Management**: `uv`
- **Vector Database**: `ChromaDB`
- **Data Modeling & Validation**: `Pydantic`
- **Tokenizer**: `jieba`
- **AI Services**:
    - **Vector Embedding**: Gitee AI (`Qwen3-Embedding-8B`)
    - **Re-ranker**: Gitee AI (`Qwen3-Reranker-8B`)
    - **Experience Distillation**: An independent conversational LLM service.

### 5.2. Configuration and Key Management

All API keys and sensitive configurations are loaded from a `.env` file in the project root. The `config.py` module handles loading these variables and initializing all AI service clients.

**`.env` Example:**
```env
# Gitee AI service for Embeddings
GITEE_AI_EMBEDDING_BASE_URL="https://ai.gitee.com/v1"
GITEE_AI_EMBEDDING_API_KEY="your_gitee_embedding_api_key"

# Gitee AI service for Reranking
GITEE_AI_RERANKER_BASE_URL="https://ai.gitee.com/v1/rerank"
GITEE_AI_RERANKER_API_KEY="your_gitee_reranker_api_key"

# Conversational LLM service for experience distillation
EXPERIENCE_LLM_BASE_URL="https://poloai.top/v1"
EXPERIENCE_LLM_API_KEY="your_experience_llm_api_key"
```
The `config.py` module implements a "fail-fast" strategy, raising a `ConfigurationError` if any required environment variable is missing.
