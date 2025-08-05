# Enhanced Memory Module (RAG-based) Technical Design Document (V1.0)

## 1. Overview

This section defines the design goals and core architecture of the **Version 1.0 (V1.0)** memory module. Future feature iterations will be described in new version sections.

This document aims to design an independently deployable, testable, and extensible enhanced memory module for the GUI Agent project. This module will function as a system based on Retrieval-Augmented Generation (RAG) for the long-term storage, management, and retrieval of two core types of agent memories:

1.  **Episodic Memory**: Reusable operational flows, including both successful and failed experiences, learned by the agent during task execution.
2.  **Semantic Memory**: Objective facts and rules about operating systems, specific applications, or general domains.

By combining these two types of memory, the goal is to comprehensively improve the agent's decision-making efficiency and accuracy in future tasks.

## 2. System Goals

### 2.1. Functional Goals
- **Modularity**: The memory system shall exist as an independent Python package, decoupled from the main Agent logic, to facilitate independent development and testing.
- **Scalability**: The architecture must support easy replacement or upgrading of core components (e.g., vector database, embedding models) in the future.
- **High Performance**: Capable of efficiently handling the storage and retrieval of large-scale memory data.
- **High-Quality Retrieval**: Implement a hybrid retrieval and unified re-ranking mechanism for both memory types to ensure high relevance of retrieved results to the current task.

### 2.2. Non-Functional Goals
- **Security**: All sensitive information, such as API keys, must be managed securely via environment variables and never be hardcoded. All data from external sources must be rigorously validated upon ingestion.
- **Maintainability**: The codebase should adhere to modern Python best practices, including clear coding standards, comprehensive testing, and logical separation of concerns to ensure long-term maintainability.
- **Cross-Platform Compatibility**: Given that the selected technologies are compatible, the module must be guaranteed to function correctly on both Windows and Linux platforms.

## 3. System Architecture

### 3.1. High-Level Design

The memory system consists of three core components: **Storage Layer**, **Ingestion Layer**, and **Retrieval Layer**.

- **Storage Layer**: The persistent foundation of the system, efficiently storing two types of structured memories and their vector representations in different collections within a unified database.
- **Ingestion Layer (Learning/Ingestion)**:
    - **Experience Learning**: Responsible for automatically converting the agent's raw operational history after task completion into structured "operational experiences" and storing them in the storage layer.
    - **Knowledge Ingestion**: Provides an interface for manually or batch-importing "semantic knowledge."
- **Retrieval Layer (Recall)**: Responsible for retrieving both types of memories in parallel based on the current task instruction and precisely extracting the most relevant historical experiences and background knowledge through a unified re-ranking strategy.

### 3.2. Directory Structure

All code related to the memory system will be located in the `gui_agent_memory` directory.

```
gui-agent/
└── gui_agent_memory/
    ├── __init__.py
    ├── main.py             # Main module entry point, provides high-level APIs
    ├── config.py           # Module configuration (model names, DB paths, etc.)
    ├── models.py           # Defines data models for both memory types (Pydantic-based)
    ├── storage.py          # Storage Layer: Encapsulates database interactions
    ├── ingestion.py        # Ingestion Layer: Implements experience learning and knowledge ingestion flows
    ├── retriever.py        # Retrieval Layer: Implements parallel retrieval and unified re-ranking logic
    └── tests/              # Independent test suite for the memory module
```

### 3.3. Core Component Details

#### 3.3.1. Storage Layer (`storage.py`)

- **Technology Stack**: `ChromaDB` (local vector database).
- **Logical Partitioning**: Create two independent `Collection`s within the database:
    - `experiential_memories`: For storing operational experiences.
    - `declarative_memories`: For storing semantic knowledge.
- **Core Data Models (`models.py`)**:
  ```python
  # gui_agent_memory/models.py
  from pydantic import BaseModel, Field
  from typing import List, Dict, Any
  from datetime import datetime

  # Defines the structure of a single action step, serving as a basic unit of an "experience"
  class ActionStep(BaseModel):
      thought: str  # The thought process when executing this step
      action: str   # The type of action (e.g., click, type)
      target_element_description: str # Description of the action's target

  class ExperienceRecord(BaseModel):
      task_description: str
      keywords: List[str]
      action_flow: List[ActionStep]  # A complete, ordered list of action steps to complete the task
      preconditions: str
      is_successful: bool
      # Metadata
      usage_count: int
      last_used_at: datetime
      source_task_id: str

  # 2. Model for semantic knowledge
  class FactRecord(BaseModel):
      content: str = Field(description="The core content of the fact or knowledge, used for vector retrieval.")
      keywords: List[str]
      source: str = Field(default="manual")
      # Metadata
      usage_count: int
      last_used_at: datetime
  ```
- **V1.0 Scope Note on Reserved Fields**:
  > The `usage_count` and `last_used_at` fields in `ExperienceRecord` and `FactRecord` are **reserved** for future advanced memory management strategies. In the V1.0 implementation, these fields will be created with default values but **will not** be read or updated by any logic.

- **Data Model Notes (Schema Design & Evolution)**:
    - **Blueprint for LLM Prompts**: The Pydantic models serve as the core "data contract" of the system.
    - **Flexibility in Development**: These models can be flexibly adjusted during the development and testing phases.
    - **Evolution in Production**: Once the system is live, future model evolution will require a standard **Data Migration** strategy.

- **Core Functions**:
  - `get_collection(name: str)`
  - `add_memories(collection_name: str, records: List[BaseModel], vectors: List[List[float]])`
  - `query(collection_name: str, query_vector: List[float], filter_dict: dict, top_k: int)`

#### 3.3.2. Ingestion Layer (`ingestion.py`)

- **Implementation Details: Vectorization and Metadata**
    - **Vectorized Content**: For `ExperienceRecord`, the `task_description` field is vectorized. For `FactRecord`, the `content` field is vectorized.
    - **Metadata Content**: **All other fields** in the data objects will be stored as metadata.

- **V1.0 Temporary Reflection Mechanism**
  > To support end-to-end testing in V1.0 and ensure reflection quality, this module provides a temporary, LLM-based reflection function triggered by the `learn_from_task` interface. It internally calls the **independently configured experience distillation LLM service** to refine `raw_history` into a high-quality `ExperienceRecord` object.

- **Idempotency Guarantee**
  > Before writing a new `ExperienceRecord`, the system must first query the database using the incoming `source_task_id`. If the record already exists, the write operation will be **skipped** to prevent data duplication.

#### 3.3.3. Retrieval Layer (`retriever.py`)

- **Core Strategy**: Parallel Hybrid Retrieval + Unified Re-ranking.

- **Implementation Details: Hybrid Retrieval Mechanism (V1.0 Baseline Strategy)**
  1.  **Keyword Generation (at Ingestion time)**: Use the **experience distillation LLM service** with a constrained prompt to populate the `keywords` field, prioritizing application names and core feature nouns. Programmatically add explicit context like `app_name` if available.
  2.  **Keyword Extraction (at Retrieval time)**: Use the **`jieba`** library to tokenize the input `query` string.
  3.  **Parallel Retrieval**: Perform both vector search and keyword-based filtering.
  4.  **Result Merging and Re-ranking**: Perform a **union** on the candidate sets, **truncate** the merged list to a predefined limit (e.g., 20), and then use the Reranker model for fine-grained sorting.
  5.  **Output**: Return the top-N most relevant memories.

## 4. Module Interaction and Integration Flow

> **Important Development Note (V1.0 Scope)**
>
> This module (`gui_agent_memory`) will be developed and tested as an independent Python package. The integration flow described below should be considered a **design reference and future integration guide**.

### 4.1. Main API (`main.py`)

- `retrieve_memories(query: str, top_n: int = 3) -> Dict[str, List[BaseModel]]`
- `learn_from_task(raw_history: List[dict], is_successful: bool, source_task_id: str) -> str`: **[V1.0 Temporary Interface]**
- `add_experience(experience: ExperienceRecord) -> str`: **[Future-facing Interface]**
- `add_fact(content: str, keywords: List[str]) -> str`

### 4.2. Integration Flow with the Main Agent

#### 4.2.1. Retrieval Flow

##### Input Query Specification
> The `query` parameter should be a grammatically complete and meaningful sentence describing the Agent's current intent. Vague inputs may lead to poor retrieval quality.

**Flow**:
1.  **Trigger**: In the main Agent's decision loop.
2.  **Construct Query**: The main Agent constructs a query adhering to the specification.
3.  **Call Memory Module**: `gui_agent_memory.main.retrieve_memories(query=...)`.
4.  **Format Memories**: The main Agent formats the results into concise text.
5.  **Inject into Prompt**: The formatted text is passed as `custom_prompt` to the main Agent's LLM call.

#### 4.2.2. Learning Flow

1.  **Trigger**: After the main Agent completes a task.
2.  **Data Collection**: The main Agent collects `raw_history` and a unique `source_task_id`.
3.  **Invocation (V1.0)**: Call the `learn_from_task(...)` interface.
4.  **Invocation (Future)**: Switch to calling `add_experience(...)` once an independent reflection module is available.

##### Robustness Design: Handling Learning Failures
> All learning-related APIs will include a top-level `try...except` block. Upon catching any exception, the system will append the failed input data and error message to a local log file (`./memory_system/logs/failed_learning_tasks.jsonl`).

## 5. Tech Stack, Configuration, and Key Management

### 5.1. Tech Stack

- **Package Management**: `uv`
- **Vector Database**: `ChromaDB`
- **Data Modeling / Validation**: `Pydantic`
- **Tokenizer**: `jieba`
- **Vector Embedding**: `Gitee AI` (Model: `Qwen3-Embedding-8B`)
- **Re-ranker**: `Gitee AI` (Model: `Qwen3-Reranker-8B`)
- **Experience Distillation LLM Service**: An independent conversational LLM.

### 5.2. Key Management: Using `.env` Files

All API keys and sensitive configurations will be loaded from a `.env` file in the project root.

**`gui-agent/.env` Example:**
```env
# Gitee AI service for Embedding and Re-ranking
GITEE_AI_BASE_URL="https://ai.gitee.com/v1"
GITEE_AI_API_KEY="your_gitee_ai_api_key_here"

# Conversational LLM service for experience distillation within this module
EXPERIENCE_LLM_BASE_URL="https://poloai.top/v1"
EXPERIENCE_LLM_API_KEY="your_polo_ai_api_key_here"
```

### 5.3. Configuration Loading and Client Initialization (`config.py`)

`gui_agent_memory/config.py` will be responsible for loading all configuration and initializing all AI service client instances for this module.

### 5.4. Dependency Management
Package management will be handled exclusively by `uv`.
- **Source of Truth**: The `pyproject.toml` file is the single source of truth for all project dependencies.
- **Adding Dependencies**: `uv add <package>`
- **Adding Dev Dependencies**: `uv add --dev <package>`
- **Forbidden**: The use of `uv pip` or `@latest` syntax is strictly forbidden to ensure deterministic builds.

## 6. Testing Strategy

- **Test Runner**: Tests will be executed via `uv run pytest`.
- **Test Structure**: Tests should be organized to mirror the source code structure. The **Arrange, Act, Assert (AAA)** pattern is the required convention.
- **Mocking**: All external service calls (e.g., to Gitee AI, Experience LLM) **must be mocked**.
- **Coverage Requirements**:
    - Tests must cover all significant logic, including edge cases and error conditions.
    - All new features must be accompanied by a comprehensive test suite.
    - All bug fixes must include a regression test that fails before the fix and passes after.
- **Test Types**:
    - **Unit Tests**: Target atomic functions.
    - **Integration Tests**: Verify module interactions, using mocked external services.
    - **End-to-End Tests**: Validate the complete closed-loop flow via public APIs.

## 7. Code Quality and Conventions

To ensure a high-quality and maintainable codebase, all contributions must adhere to the following standards:

- **Tooling**: All code quality tools will be executed via `uv run <tool_name>` (e.g., `uv run black .`).
- **Code Formatting**: Code must be formatted using `black` (max line length 88 characters).
- **Import Sorting**: Imports must be sorted using `isort`.
- **Type Safety**:
    - **Type hints are required for all code**, including function arguments, return values, and variables.
    - Static type checking will be enforced using `mypy`.
- **Documentation**:
    - **Public APIs must have docstrings** explaining their purpose, arguments, and return values.
- **Function Design**:
    - Functions must be focused, single-purpose, and small.
    - Adherence to existing design patterns within the codebase is required.
- **Style Guide**: Adherence to the PEP 8 style guide is required.

## 8. Development and Implementation Mandates

To ensure robustness and maintainability, the following practices are mandatory for V1.0 development.

### 8.1. Structured Mock Data for Testing
- All mock API responses for external services (Gitee AI, Experience LLM) must be stored as static data files (e.g., JSON).
- A dedicated `tests/mocks/` or `tests/fixtures/` directory shall be created to house these files. This practice decouples test logic from mock data, improving clarity and reusability.

### 8.2. Externalized and Version-Controlled Prompts
- Prompts for the "Experience Distillation LLM Service" must not be hardcoded within the Python source.
- They shall be maintained in external template files (e.g., `.txt`, `.prompt`) and managed under version control.
- The development schedule must allocate specific time for multiple iterations of prompt engineering and evaluation to optimize for performance and cost.

### 8.3. Robust Configuration Loading
- The `config.py` module must implement a "fail-fast" strategy for environment variable loading.
- Upon initialization, it must perform an existence check for all critical keys (e.g., `GITEE_AI_API_KEY`, `EXPERIENCE_LLM_API_KEY`).
- If a required key is missing, the system must raise a definitive `ConfigurationError` immediately to prevent runtime failures due to improper environment setup.
