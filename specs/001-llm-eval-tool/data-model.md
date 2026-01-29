# Data Model: LLM Evaluation Tool

**Feature Branch**: `001-llm-eval-tool`  
**Created**: 2026-01-28  
**Purpose**: Define Pydantic models for all entities

## Entity Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        LLMEvalConfig                            │
│  (persisted: .conf/llm_eval_config.json)                       │
├─────────────────────────────────────────────────────────────────┤
│  generator_model: LLMModel                                      │
│  evaluation_models: list[LLMModel]  (max 5)                    │
│  settings: EvalSettings                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      QuestionDataset                            │
│  (persisted: outputs/llm_eval/datasets/{dataset_id}.json)      │
├─────────────────────────────────────────────────────────────────┤
│  dataset_id: str                                                │
│  source_document: GroundingDocument                             │
│  conversations: list[Conversation]                              │
│  created_at: datetime                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EvaluationRun                             │
│  (persisted: outputs/llm_eval/runs/{run_id}.json)              │
├─────────────────────────────────────────────────────────────────┤
│  run_id: str                                                    │
│  dataset_id: str                                                │
│  models: list[str]  (model names)                              │
│  results: list[EvaluationResult]                               │
│  status: RunStatus                                              │
│  started_at: datetime                                           │
│  completed_at: datetime | None                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Pydantic Models

### Configuration Models

```python
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, SecretStr


class ModelType(str, Enum):
    """Supported LLM model types."""
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    CUSTOM = "custom"


class LLMModel(BaseModel):
    """Configuration for a single LLM model."""
    name: str = Field(..., min_length=1, max_length=50, description="Display name for the model")
    endpoint: str = Field(..., description="API endpoint URL")
    api_key: SecretStr = Field(..., description="API key (stored securely)")
    model_type: ModelType = Field(default=ModelType.AZURE_OPENAI)
    deployment_name: str | None = Field(default=None, description="Azure OpenAI deployment name")
    api_version: str = Field(default="2024-02-15-preview", description="API version")
    
    class Config:
        json_encoders = {SecretStr: lambda v: v.get_secret_value() if v else None}


class EvalSettings(BaseModel):
    """Configurable evaluation settings."""
    max_concurrent_calls: int = Field(default=10, ge=1, le=50, description="Max parallel API calls")
    timeout_seconds: float = Field(default=60.0, ge=10, le=300, description="Per-model timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Retry count for transient errors")
    max_conversation_history: int = Field(default=10, ge=1, le=50, description="Max user/assistant pairs to retain")


class LLMEvalConfig(BaseModel):
    """Top-level configuration for LLM Evaluation tool."""
    generator_model: LLMModel | None = Field(default=None, description="Model for question generation")
    evaluation_models: list[LLMModel] = Field(default_factory=list, max_length=5)
    settings: EvalSettings = Field(default_factory=EvalSettings)
    
    def add_evaluation_model(self, model: LLMModel) -> bool:
        """Add evaluation model if under limit. Returns success status."""
        if len(self.evaluation_models) >= 5:
            return False
        self.evaluation_models.append(model)
        return True
```

### Document & Dataset Models

```python
class GroundingDocument(BaseModel):
    """Represents an uploaded PDF document."""
    filename: str = Field(..., description="Original filename")
    content: str = Field(..., description="Extracted text content")
    page_count: int = Field(default=1, ge=1)
    file_size_bytes: int = Field(default=0, ge=0)
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class ConversationTurn(BaseModel):
    """A single turn in a multi-turn conversation."""
    turn_number: int = Field(..., ge=1, description="1-indexed turn number")
    question: str = Field(..., min_length=1, description="User question")
    ground_truth: str = Field(..., min_length=1, description="Expected answer from document")


class Conversation(BaseModel):
    """A multi-turn conversation chain."""
    conversation_id: str = Field(..., description="Unique conversation identifier")
    turns: list[ConversationTurn] = Field(..., min_length=1, max_length=10)
    selected: bool = Field(default=True, description="Whether to include in evaluation")


class QuestionDataset(BaseModel):
    """Collection of conversations for evaluation."""
    dataset_id: str = Field(..., description="Unique dataset identifier")
    source_document: GroundingDocument
    conversations: list[Conversation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def total_questions(self) -> int:
        """Total number of questions across all conversations."""
        return sum(len(c.turns) for c in self.conversations)
    
    @property
    def selected_questions(self) -> int:
        """Number of selected questions."""
        return sum(len(c.turns) for c in self.conversations if c.selected)
```

### Evaluation Models

```python
class RunStatus(str, Enum):
    """Status of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some models succeeded, some failed


class RAGMetrics(BaseModel):
    """RAG evaluation metrics for a single response."""
    latency_ms: float = Field(..., ge=0, description="Response time in milliseconds")
    groundedness: float = Field(..., ge=1, le=5, description="Alignment with source document (1-5)")
    relevance: float = Field(..., ge=1, le=5, description="Answer quality for question (1-5)")
    coherence: float = Field(..., ge=1, le=5, description="Logical consistency (1-5)")
    fluency: float = Field(..., ge=1, le=5, description="Grammar and readability (1-5)")
    
    @property
    def average_score(self) -> float:
        """Average of semantic metrics (excludes latency)."""
        return (self.groundedness + self.relevance + self.coherence + self.fluency) / 4


class EvaluationResult(BaseModel):
    """Result for a single model-question evaluation."""
    model_name: str = Field(..., description="Name of the evaluated model")
    conversation_id: str = Field(..., description="Conversation being evaluated")
    turn_number: int = Field(..., ge=1, description="Turn within conversation")
    question: str = Field(..., description="Question asked")
    ground_truth: str = Field(..., description="Expected answer")
    model_response: str | None = Field(default=None, description="Model's response")
    metrics: RAGMetrics | None = Field(default=None, description="Calculated metrics")
    error: str | None = Field(default=None, description="Error message if failed")
    
    @property
    def success(self) -> bool:
        """Whether evaluation succeeded."""
        return self.error is None and self.model_response is not None


class EvaluationRun(BaseModel):
    """A complete evaluation run across models and questions."""
    run_id: str = Field(..., description="Unique run identifier")
    dataset_id: str = Field(..., description="Source dataset ID")
    model_names: list[str] = Field(..., description="Models evaluated")
    results: list[EvaluationResult] = Field(default_factory=list)
    status: RunStatus = Field(default=RunStatus.PENDING)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = Field(default=None)
    
    @property
    def success_rate(self) -> float:
        """Percentage of successful evaluations."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)
    
    def get_model_summary(self, model_name: str) -> dict[str, Any]:
        """Get aggregated metrics for a specific model."""
        model_results = [r for r in self.results if r.model_name == model_name and r.metrics]
        if not model_results:
            return {"model": model_name, "count": 0}
        
        metrics = [r.metrics for r in model_results]
        return {
            "model": model_name,
            "count": len(metrics),
            "avg_latency_ms": sum(m.latency_ms for m in metrics) / len(metrics),
            "avg_groundedness": sum(m.groundedness for m in metrics) / len(metrics),
            "avg_relevance": sum(m.relevance for m in metrics) / len(metrics),
            "avg_coherence": sum(m.coherence for m in metrics) / len(metrics),
            "avg_fluency": sum(m.fluency for m in metrics) / len(metrics),
        }
```

## Storage Patterns

### File Paths

```python
from pathlib import Path

# Configuration (git-ignored)
CONFIG_PATH = Path(".conf") / "llm_eval_config.json"

# Datasets (separate file per dataset)
DATASETS_DIR = Path("outputs") / "llm_eval" / "datasets"
def dataset_path(dataset_id: str) -> Path:
    return DATASETS_DIR / f"{dataset_id}.json"

# Evaluation runs (separate file per run)
RUNS_DIR = Path("outputs") / "llm_eval" / "runs"
def run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"
```

### Persistence Functions

```python
def save_config(config: LLMEvalConfig) -> None:
    """Save configuration to .conf/ directory."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(config.model_dump_json(indent=2))

def load_config() -> LLMEvalConfig:
    """Load configuration from .conf/ directory."""
    if CONFIG_PATH.exists():
        return LLMEvalConfig.model_validate_json(CONFIG_PATH.read_text())
    return LLMEvalConfig()

def save_dataset(dataset: QuestionDataset) -> None:
    """Save dataset to outputs/llm_eval/datasets/."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path(dataset.dataset_id).write_text(dataset.model_dump_json(indent=2))

def save_run(run: EvaluationRun) -> None:
    """Save evaluation run to outputs/llm_eval/runs/."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path(run.run_id).write_text(run.model_dump_json(indent=2))

def list_runs() -> list[str]:
    """List all evaluation run IDs."""
    if not RUNS_DIR.exists():
        return []
    return [f.stem for f in RUNS_DIR.glob("*.json")]
```

## Validation Rules

| Entity | Rule | Enforcement |
|--------|------|-------------|
| LLMEvalConfig.evaluation_models | Max 5 models | `max_length=5` on field, `add_evaluation_model()` method |
| LLMModel.name | 1-50 characters | `min_length=1, max_length=50` |
| EvalSettings.max_concurrent_calls | 1-50 | `ge=1, le=50` |
| EvalSettings.timeout_seconds | 10-300 seconds | `ge=10, le=300` |
| RAGMetrics scores | 1-5 scale | `ge=1, le=5` |
| Conversation.turns | 1-10 turns | `min_length=1, max_length=10` |
| ConversationTurn.turn_number | 1+ | `ge=1` |

## State Diagram: EvaluationRun

```
┌─────────┐     start()     ┌─────────┐
│ PENDING │────────────────▶│ RUNNING │
└─────────┘                 └────┬────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
        ┌───────────┐     ┌───────────┐     ┌─────────┐
        │ COMPLETED │     │  PARTIAL  │     │  FAILED │
        │ (all ok)  │     │(some fail)│     │(all fail)│
        └───────────┘     └───────────┘     └─────────┘
```
