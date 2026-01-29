"""
LLM Evaluation Tool - Model configuration and persistence.

This module provides Pydantic models for LLM evaluation configuration,
question datasets, and evaluation runs. All data is persisted as JSON files.
"""

import logging
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel, Field, SecretStr, field_validator

# Module-level logger
logger = logging.getLogger(__name__)

# ============================================================================
# File Path Constants
# ============================================================================

# Configuration (git-ignored)
CONFIG_PATH = Path(".conf") / "llm_eval_config.json"

# Datasets (separate file per dataset)
DATASETS_DIR = Path("outputs") / "llm_eval" / "datasets"

# Evaluation runs (separate file per run)
RUNS_DIR = Path("outputs") / "llm_eval" / "runs"


def dataset_path(dataset_id: str) -> Path:
    """Get the file path for a dataset."""
    return DATASETS_DIR / f"{dataset_id}.json"


def run_path(run_id: str) -> Path:
    """Get the file path for an evaluation run."""
    return RUNS_DIR / f"{run_id}.json"


# ============================================================================
# Enums
# ============================================================================


class ModelType(str, Enum):
    """Supported LLM model types."""

    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    CUSTOM = "custom"


class RunStatus(str, Enum):
    """Status of an evaluation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some models succeeded, some failed


# ============================================================================
# Configuration Models
# ============================================================================


class LLMModel(BaseModel):
    """Configuration for a single LLM model."""

    name: str = Field(..., min_length=1, max_length=50, description="Display name for the model")
    endpoint: str = Field(..., description="API endpoint URL")
    api_key: SecretStr = Field(..., description="API key (stored securely)")
    model_type: ModelType = Field(default=ModelType.AZURE_OPENAI)
    deployment_name: str | None = Field(default=None, description="Azure OpenAI deployment name")
    api_version: str = Field(default="2024-02-15-preview", description="API version")
    pricing_key: str | None = Field(
        default=None,
        description="Key matching azure_openai_pricing.json for cost calculation (e.g., 'GPT-4o')",
    )

    model_config = {"json_encoders": {SecretStr: lambda v: v.get_secret_value() if v else None}}

    def model_dump_with_secrets(self) -> dict[str, Any]:
        """Dump model including secret values for persistence."""
        data = self.model_dump()
        data["api_key"] = self.api_key.get_secret_value()
        return data


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MAX_CONCURRENT = 10


class EvalSettings(BaseModel):
    """Configurable evaluation settings."""

    max_concurrent_calls: int = Field(default=10, ge=1, le=50, description="Max parallel API calls")
    timeout_seconds: float = Field(default=60.0, ge=10, le=300, description="Per-model timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Retry count for transient errors")
    max_conversation_history: int = Field(
        default=10, ge=1, le=50, description="Max user/assistant pairs to retain"
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt for evaluation models",
    )


class LLMEvalConfig(BaseModel):
    """Top-level configuration for LLM Evaluation tool."""

    generator_model: LLMModel | None = Field(
        default=None, description="Model for question generation"
    )
    evaluation_models: list[LLMModel] = Field(default_factory=list)
    settings: EvalSettings = Field(default_factory=EvalSettings)

    def add_evaluation_model(self, model: LLMModel) -> bool:
        """Add evaluation model if under limit. Returns success status."""
        if len(self.evaluation_models) >= 5:
            logger.warning("Cannot add model: limit of 5 evaluation models reached")
            return False
        self.evaluation_models.append(model)
        logger.info(f"Added evaluation model: {model.name}")
        return True

    def remove_evaluation_model(self, model_name: str) -> bool:
        """Remove evaluation model by name. Returns success status."""
        original_count = len(self.evaluation_models)
        self.evaluation_models = [m for m in self.evaluation_models if m.name != model_name]
        if len(self.evaluation_models) < original_count:
            logger.info(f"Removed evaluation model: {model_name}")
            return True
        logger.warning(f"Model not found for removal: {model_name}")
        return False

    def get_evaluation_model(self, model_name: str) -> LLMModel | None:
        """Get evaluation model by name."""
        for model in self.evaluation_models:
            if model.name == model_name:
                return model
        return None


# ============================================================================
# Model Connection Validation
# ============================================================================


async def validate_model_connection(model: LLMModel) -> dict[str, Any]:
    """
    Validate model endpoint connectivity by making a simple API call.

    Args:
        model: LLM model configuration to validate

    Returns:
        {"ok": True, "model_name": str, "latency_ms": float} on success
        {"ok": False, "error": str, "model_name": str} on failure
    """
    import time

    start_time = time.time()

    try:
        # Create appropriate client based on model type
        if model.model_type == ModelType.AZURE_OPENAI:
            # Use Entra ID authentication for Azure OpenAI
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            client = AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version=model.api_version,
                azure_endpoint=model.endpoint,
            )
            deployment = model.deployment_name or "gpt-4"
        elif model.model_type == ModelType.OPENAI:
            client = AsyncOpenAI(
                api_key=model.api_key.get_secret_value(),
                base_url=model.endpoint if model.endpoint else None,
            )
            deployment = model.deployment_name or "gpt-4"
        else:
            # Custom model type - use OpenAI-compatible client
            client = AsyncOpenAI(
                api_key=model.api_key.get_secret_value(),
                base_url=model.endpoint,
            )
            deployment = model.deployment_name or "gpt-4"

        # Make a simple test call
        await client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Hi"}],
            max_completion_tokens=5,
        )

        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"Model connection validated: {model.name} ({latency_ms:.1f}ms)")

        return {
            "ok": True,
            "model_name": model.name,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        error_msg = str(e)
        logger.error(f"Model connection failed: {model.name} - {error_msg}")

        return {
            "ok": False,
            "error": error_msg,
            "model_name": model.name,
            "latency_ms": latency_ms,
        }


# ============================================================================
# Document & Dataset Models
# ============================================================================


class GroundingDocument(BaseModel):
    """Represents an uploaded PDF document with per-page content."""

    filename: str = Field(..., description="Original filename")
    pages: list[str] = Field(default_factory=list, description="Text content per page (0-indexed)")
    page_count: int = Field(default=1, ge=1)
    file_size_bytes: int = Field(default=0, ge=0)
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def content(self) -> str:
        """Backward-compatible: join all pages into single string."""
        return "\n\n".join(p for p in self.pages if p)


class ConversationTurn(BaseModel):
    """A single turn in a multi-turn conversation."""

    turn_number: int = Field(..., ge=1, description="1-indexed turn number")
    question: str = Field(..., min_length=1, description="User question")


class Conversation(BaseModel):
    """A multi-turn conversation chain referencing a specific page."""

    conversation_id: str = Field(..., description="Unique conversation identifier")
    page_reference: int = Field(..., ge=0, description="0-indexed page number for RAG grounding")
    turns: list[ConversationTurn] = Field(..., min_length=1)
    selected: bool = Field(default=True, description="Whether to include in evaluation")


class QuestionDataset(BaseModel):
    """Collection of conversations for evaluation."""

    dataset_id: str = Field(..., description="Unique dataset identifier")
    source_document: GroundingDocument
    conversations: list[Conversation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_questions(self) -> int:
        """Total number of questions across all conversations."""
        return sum(len(c.turns) for c in self.conversations)

    @property
    def selected_questions(self) -> int:
        """Number of selected questions."""
        return sum(len(c.turns) for c in self.conversations if c.selected)


# ============================================================================
# Evaluation Models
# ============================================================================


class RAGMetrics(BaseModel):
    """RAG evaluation metrics for a single response."""

    latency_ms: float | None = Field(
        default=None, ge=0, description="Response time in milliseconds"
    )
    groundedness: float | None = Field(
        default=None, ge=1, le=5, description="Alignment with source document (1-5)"
    )
    relevance: float | None = Field(
        default=None, ge=1, le=5, description="Answer quality for question (1-5)"
    )
    coherence: float | None = Field(
        default=None, ge=1, le=5, description="Logical consistency (1-5)"
    )
    fluency: float | None = Field(
        default=None, ge=1, le=5, description="Grammar and readability (1-5)"
    )

    @property
    def average_score(self) -> float | None:
        """Average of semantic metrics (excludes latency)."""
        scores = [
            s
            for s in [self.groundedness, self.relevance, self.coherence, self.fluency]
            if s is not None
        ]
        return sum(scores) / len(scores) if scores else None


class ChatHistoryItem(BaseModel):
    """A single item in conversation history for judge context."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class EvaluationResult(BaseModel):
    """Result for a single model-question evaluation."""

    model_name: str = Field(..., description="Name of the evaluated model")
    conversation_id: str = Field(..., description="Conversation being evaluated")
    turn_number: int = Field(..., ge=1, description="Turn within conversation")
    question: str = Field(default="", description="Question asked")
    page_reference: int | None = Field(
        default=None, ge=0, description="0-indexed page for RAG grounding"
    )
    response: str | None = Field(default=None, description="Model's response")
    latency_ms: float = Field(default=0.0, ge=0, description="Total end-to-end latency in ms")
    api_latency_ms: float = Field(default=0.0, ge=0, description="Pure API call latency in ms")
    success: bool = Field(default=True, description="Whether evaluation succeeded")
    metrics: RAGMetrics | None = Field(default=None, description="Calculated metrics")
    error: str | None = Field(default=None, description="Error message if failed")
    chat_history: list[ChatHistoryItem] | None = Field(
        default=None, description="Prior conversation turns for multi-turn context"
    )
    # Token tracking fields
    prompt_tokens: int = Field(default=0, ge=0, description="Input tokens used")
    completion_tokens: int = Field(default=0, ge=0, description="Output tokens generated")
    cached_tokens: int = Field(default=0, ge=0, description="Cached input tokens (lower cost)")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens (prompt + completion)")
    messages: list[dict[str, str]] | None = Field(
        default=None, description="Complete API messages payload (request + response)"
    )

    @field_validator("chat_history")
    @classmethod
    def validate_chat_history(cls, v: list[ChatHistoryItem] | None) -> list[ChatHistoryItem] | None:
        """Validate chat_history strictly alternates user → assistant."""
        if v is None or len(v) == 0:
            return v

        # Must start with 'user'
        if v[0].role != "user":
            raise ValueError(f"chat_history must start with 'user', got '{v[0].role}'")

        # Validate alternation: user, assistant, user, assistant, ...
        expected_role = "user"
        for i, item in enumerate(v):
            if item.role not in ("user", "assistant"):
                raise ValueError(
                    f"chat_history[{i}] has invalid role '{item.role}', expected 'user' or 'assistant'"
                )
            if item.role != expected_role:
                raise ValueError(
                    f"chat_history[{i}] has role '{item.role}', expected '{expected_role}' "
                    f"(must alternate user → assistant)"
                )
            expected_role = "assistant" if expected_role == "user" else "user"

        return v

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[dict[str, str]] | None) -> list[dict[str, str]] | None:
        """Validate messages follows OpenAI API format: system, then alternating user/assistant."""
        if v is None or len(v) == 0:
            return v

        # Must start with 'system'
        if v[0].get("role") != "system":
            raise ValueError(f"messages must start with 'system', got '{v[0].get('role')}'")

        # Must end with 'assistant' (includes model response)
        if v[-1].get("role") != "assistant":
            raise ValueError(
                f"messages must end with 'assistant', got '{v[-1].get('role')}' "
                f"(should include model response)"
            )

        # Validate alternation after system: user, assistant, user, assistant, ...
        if len(v) < 2:
            return v

        expected_role = "user"
        for i, msg in enumerate(v[1:], start=1):
            role = msg.get("role")
            if role not in ("user", "assistant"):
                raise ValueError(
                    f"messages[{i}] has invalid role '{role}', expected 'user' or 'assistant'"
                )
            if role != expected_role:
                raise ValueError(
                    f"messages[{i}] has role '{role}', expected '{expected_role}' "
                    f"(must alternate user → assistant after system)"
                )
            expected_role = "assistant" if expected_role == "user" else "user"

        return v


class EvaluationRun(BaseModel):
    """A complete evaluation run across models and questions."""

    run_id: str = Field(..., description="Unique run identifier")
    dataset_id: str = Field(..., description="Source dataset ID")
    model_names: list[str] = Field(..., description="Models evaluated")
    results: list[EvaluationResult] = Field(default_factory=list)
    status: RunStatus = Field(default=RunStatus.PENDING)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = Field(default=None)

    @property
    def success_rate(self) -> float:
        """Percentage of successful evaluations."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    def get_model_summary(self, model_name: str) -> dict[str, Any]:
        """Get aggregated metrics for a specific model."""
        model_results = [r for r in self.results if r.model_name == model_name and r.success]
        if not model_results:
            return {"model": model_name, "count": 0}

        # Collect results with metrics
        results_with_metrics = [r for r in model_results if r.metrics is not None]

        # Calculate latency from results (prefer api_latency_ms if available)
        api_latencies = [r.api_latency_ms for r in model_results if r.api_latency_ms > 0]
        avg_latency = sum(r.latency_ms for r in model_results) / len(model_results)
        avg_api_latency = sum(api_latencies) / len(api_latencies) if api_latencies else avg_latency

        # Aggregate token counts
        total_prompt_tokens = sum(r.prompt_tokens for r in model_results)
        total_completion_tokens = sum(r.completion_tokens for r in model_results)
        total_cached_tokens = sum(r.cached_tokens for r in model_results)
        total_tokens = sum(r.total_tokens for r in model_results)

        if not results_with_metrics:
            return {
                "model": model_name,
                "count": len(model_results),
                "avg_latency_ms": avg_api_latency,  # Use pure API latency
                "avg_total_latency_ms": avg_latency,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_cached_tokens": total_cached_tokens,
                "total_tokens": total_tokens,
            }

        # Calculate metric averages from metrics object
        metrics = [r.metrics for r in results_with_metrics]
        return {
            "model": model_name,
            "count": len(model_results),
            "avg_latency_ms": avg_api_latency,  # Use pure API latency
            "avg_total_latency_ms": avg_latency,
            "avg_groundedness": sum(m.groundedness or 0 for m in metrics) / len(metrics),
            "avg_relevance": sum(m.relevance or 0 for m in metrics) / len(metrics),
            "avg_coherence": sum(m.coherence or 0 for m in metrics) / len(metrics),
            "avg_fluency": sum(m.fluency or 0 for m in metrics) / len(metrics),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_cached_tokens": total_cached_tokens,
            "total_tokens": total_tokens,
        }


# ============================================================================
# Persistence Functions
# ============================================================================


def save_config(config: LLMEvalConfig) -> dict[str, Any]:
    """
    Save configuration to .conf/ directory.

    Returns:
        {"ok": True, "path": str} on success
        {"ok": False, "error": str} on failure
    """
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Serialize with secrets exposed for storage
        data = config.model_dump(mode="json")
        # Convert SecretStr values to plain strings for storage
        if (
            data.get("generator_model")
            and "api_key" in data["generator_model"]
            and config.generator_model
        ):
            data["generator_model"]["api_key"] = config.generator_model.api_key.get_secret_value()
        for i, model in enumerate(config.evaluation_models):
            if "api_key" in data["evaluation_models"][i]:
                data["evaluation_models"][i]["api_key"] = model.api_key.get_secret_value()

        import json

        CONFIG_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(f"Saved LLM eval config to {CONFIG_PATH}")
        return {"ok": True, "path": str(CONFIG_PATH)}
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return {"ok": False, "error": str(e)}


def load_config() -> LLMEvalConfig:
    """
    Load configuration from .conf/ directory.

    Returns:
        LLMEvalConfig instance (default if file doesn't exist)
    """
    try:
        if CONFIG_PATH.exists():
            config = LLMEvalConfig.model_validate_json(CONFIG_PATH.read_text(encoding="utf-8"))
            logger.info(f"Loaded LLM eval config from {CONFIG_PATH}")
            return config
        logger.debug("No config file found, returning default config")
        return LLMEvalConfig()
    except Exception as e:
        logger.error(f"Failed to load config: {e}, returning default")
        return LLMEvalConfig()


def save_dataset(dataset: QuestionDataset) -> dict[str, Any]:
    """
    Save dataset to outputs/llm_eval/datasets/.

    Returns:
        {"ok": True, "path": str} on success
        {"ok": False, "error": str} on failure
    """
    try:
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        path = dataset_path(dataset.dataset_id)
        path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Saved dataset {dataset.dataset_id} to {path}")
        return {"ok": True, "path": str(path)}
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        return {"ok": False, "error": str(e)}


def load_dataset(dataset_id: str) -> QuestionDataset | None:
    """
    Load dataset from outputs/llm_eval/datasets/.

    Returns:
        QuestionDataset instance or None if not found
    """
    try:
        path = dataset_path(dataset_id)
        if path.exists():
            dataset = QuestionDataset.model_validate_json(path.read_text(encoding="utf-8"))
            logger.info(f"Loaded dataset {dataset_id} from {path}")
            return dataset
        logger.debug(f"Dataset not found: {dataset_id}")
        return None
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_id}: {e}")
        return None


def list_datasets() -> list[str]:
    """List all dataset IDs."""
    if not DATASETS_DIR.exists():
        return []
    return sorted([f.stem for f in DATASETS_DIR.glob("*.json")])


def save_run(run: EvaluationRun) -> dict[str, Any]:
    """
    Save evaluation run to outputs/llm_eval/runs/.

    Returns:
        {"ok": True, "path": str} on success
        {"ok": False, "error": str} on failure
    """
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path = run_path(run.run_id)
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Saved run {run.run_id} to {path}")
        return {"ok": True, "path": str(path)}
    except Exception as e:
        logger.error(f"Failed to save run: {e}")
        return {"ok": False, "error": str(e)}


def load_run(run_id: str) -> EvaluationRun | None:
    """
    Load evaluation run from outputs/llm_eval/runs/.

    Returns:
        EvaluationRun instance or None if not found
    """
    try:
        path = run_path(run_id)
        if path.exists():
            run = EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))
            logger.info(f"Loaded run {run_id} from {path}")
            return run
        logger.debug(f"Run not found: {run_id}")
        return None
    except Exception as e:
        logger.error(f"Failed to load run {run_id}: {e}")
        return None


def list_runs() -> list[str]:
    """List all evaluation run IDs."""
    if not RUNS_DIR.exists():
        return []
    return sorted([f.stem for f in RUNS_DIR.glob("*.json")], reverse=True)
