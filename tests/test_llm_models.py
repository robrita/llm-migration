"""
Unit tests for helpers/llm_models.py

Tests cover:
- Pydantic model validation
- Config save/load round-trip
- Model add/edit/delete operations
- 5-model limit enforcement
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from helpers.llm_models import (
    Conversation,
    ConversationTurn,
    EvalSettings,
    EvaluationResult,
    EvaluationRun,
    GroundingDocument,
    LLMEvalConfig,
    LLMModel,
    ModelType,
    QuestionDataset,
    RAGMetrics,
    RunStatus,
    list_datasets,
    list_runs,
    load_config,
    load_dataset,
    load_run,
    save_config,
    save_dataset,
    save_run,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_llm_model() -> LLMModel:
    """Create a sample LLM model for testing."""
    return LLMModel(
        name="Test-GPT-4",
        endpoint="https://test.openai.azure.com/",
        api_key="test-api-key-12345",
        model_type=ModelType.AZURE_OPENAI,
        deployment_name="gpt-4",
        api_version="2024-02-15-preview",
        pricing_key="GPT-4o",
    )


@pytest.fixture
def sample_llm_model_no_pricing() -> LLMModel:
    """Create a sample LLM model without pricing_key for testing."""
    return LLMModel(
        name="Test-GPT-4-NoPricing",
        endpoint="https://test.openai.azure.com/",
        api_key="test-api-key-12345",
        model_type=ModelType.AZURE_OPENAI,
        deployment_name="gpt-4",
        api_version="2024-02-15-preview",
    )


@pytest.fixture
def sample_config(sample_llm_model: LLMModel) -> LLMEvalConfig:
    """Create a sample config for testing."""
    return LLMEvalConfig(
        generator_model=sample_llm_model,
        evaluation_models=[sample_llm_model],
        settings=EvalSettings(max_concurrent_calls=5, timeout_seconds=30.0),
    )


@pytest.fixture
def temp_config_path(tmp_path: Path):
    """Patch CONFIG_PATH to use a temp directory."""
    test_config_path = tmp_path / ".conf" / "llm_eval_config.json"
    with patch("helpers.llm_models.CONFIG_PATH", test_config_path):
        yield test_config_path


@pytest.fixture
def temp_datasets_dir(tmp_path: Path):
    """Patch DATASETS_DIR to use a temp directory."""
    test_datasets_dir = tmp_path / "outputs" / "llm_eval" / "datasets"
    with (
        patch("helpers.llm_models.DATASETS_DIR", test_datasets_dir),
        patch(
            "helpers.llm_models.dataset_path",
            lambda dataset_id: test_datasets_dir / f"{dataset_id}.json",
        ),
    ):
        yield test_datasets_dir


@pytest.fixture
def temp_runs_dir(tmp_path: Path):
    """Patch RUNS_DIR to use a temp directory."""
    test_runs_dir = tmp_path / "outputs" / "llm_eval" / "runs"
    with (
        patch("helpers.llm_models.RUNS_DIR", test_runs_dir),
        patch(
            "helpers.llm_models.run_path",
            lambda run_id: test_runs_dir / f"{run_id}.json",
        ),
    ):
        yield test_runs_dir


# ============================================================================
# Model Validation Tests
# ============================================================================


@pytest.mark.unit
class TestLLMModel:
    """Tests for LLMModel Pydantic model."""

    def test_create_valid_model(self, sample_llm_model: LLMModel):
        """Test creating a valid LLM model."""
        assert sample_llm_model.name == "Test-GPT-4"
        assert sample_llm_model.model_type == ModelType.AZURE_OPENAI
        assert sample_llm_model.api_key.get_secret_value() == "test-api-key-12345"
        assert sample_llm_model.pricing_key == "GPT-4o"

    def test_model_without_pricing_key(self, sample_llm_model_no_pricing: LLMModel):
        """Test creating a model without pricing_key."""
        assert sample_llm_model_no_pricing.pricing_key is None

    def test_model_name_validation(self):
        """Test model name length validation."""
        # Empty name should fail
        with pytest.raises(ValueError):
            LLMModel(name="", endpoint="https://test.com", api_key="key")

        # Name too long should fail
        with pytest.raises(ValueError):
            LLMModel(name="a" * 51, endpoint="https://test.com", api_key="key")

    def test_model_dump_hides_secrets_by_default(self, sample_llm_model: LLMModel):
        """Test that model_dump hides SecretStr by default."""
        data = sample_llm_model.model_dump()
        # SecretStr is serialized as **********
        assert "test-api-key-12345" not in str(data)

    def test_model_dump_with_secrets(self, sample_llm_model: LLMModel):
        """Test model_dump_with_secrets exposes API key."""
        data = sample_llm_model.model_dump_with_secrets()
        assert data["api_key"] == "test-api-key-12345"


@pytest.mark.unit
class TestEvalSettings:
    """Tests for EvalSettings Pydantic model."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = EvalSettings()
        assert settings.max_concurrent_calls == 10
        assert settings.timeout_seconds == 60.0
        assert settings.max_retries == 3
        assert settings.max_conversation_history == 10

    def test_settings_validation(self):
        """Test settings validation bounds."""
        # max_concurrent_calls out of range
        with pytest.raises(ValueError):
            EvalSettings(max_concurrent_calls=0)
        with pytest.raises(ValueError):
            EvalSettings(max_concurrent_calls=51)

        # timeout_seconds out of range
        with pytest.raises(ValueError):
            EvalSettings(timeout_seconds=5)
        with pytest.raises(ValueError):
            EvalSettings(timeout_seconds=301)


@pytest.mark.unit
class TestRAGMetrics:
    """Tests for RAGMetrics Pydantic model."""

    def test_valid_metrics(self):
        """Test creating valid metrics."""
        metrics = RAGMetrics(
            latency_ms=150.5,
            groundedness=4.5,
            relevance=4.0,
            coherence=3.5,
            fluency=5.0,
        )
        assert metrics.average_score == (4.5 + 4.0 + 3.5 + 5.0) / 4

    def test_metrics_score_validation(self):
        """Test metric score bounds (1-5)."""
        with pytest.raises(ValueError):
            RAGMetrics(latency_ms=100, groundedness=0, relevance=3, coherence=3, fluency=3)
        with pytest.raises(ValueError):
            RAGMetrics(latency_ms=100, groundedness=6, relevance=3, coherence=3, fluency=3)


# ============================================================================
# Config Operations Tests
# ============================================================================


@pytest.mark.unit
class TestLLMEvalConfig:
    """Tests for LLMEvalConfig operations."""

    def test_add_evaluation_model_success(self, sample_llm_model: LLMModel):
        """Test adding evaluation models up to limit."""
        config = LLMEvalConfig()
        for i in range(5):
            model = LLMModel(
                name=f"Model-{i}",
                endpoint="https://test.com",
                api_key="key",
            )
            assert config.add_evaluation_model(model) is True
        assert len(config.evaluation_models) == 5

    def test_add_evaluation_model_limit(self, sample_llm_model: LLMModel):
        """Test 5-model limit enforcement."""
        config = LLMEvalConfig()
        # Add 5 models
        for i in range(5):
            model = LLMModel(
                name=f"Model-{i}",
                endpoint="https://test.com",
                api_key="key",
            )
            config.add_evaluation_model(model)

        # 6th model should fail
        sixth_model = LLMModel(
            name="Model-6",
            endpoint="https://test.com",
            api_key="key",
        )
        assert config.add_evaluation_model(sixth_model) is False
        assert len(config.evaluation_models) == 5

    def test_remove_evaluation_model(self, sample_config: LLMEvalConfig):
        """Test removing an evaluation model."""
        assert len(sample_config.evaluation_models) == 1
        result = sample_config.remove_evaluation_model("Test-GPT-4")
        assert result is True
        assert len(sample_config.evaluation_models) == 0

    def test_remove_nonexistent_model(self, sample_config: LLMEvalConfig):
        """Test removing a model that doesn't exist."""
        result = sample_config.remove_evaluation_model("NonExistent")
        assert result is False

    def test_get_evaluation_model(self, sample_config: LLMEvalConfig):
        """Test getting evaluation model by name."""
        model = sample_config.get_evaluation_model("Test-GPT-4")
        assert model is not None
        assert model.name == "Test-GPT-4"

        # Non-existent model
        model = sample_config.get_evaluation_model("NonExistent")
        assert model is None


# ============================================================================
# Persistence Tests
# ============================================================================


@pytest.mark.unit
class TestConfigPersistence:
    """Tests for config save/load operations."""

    def test_save_and_load_config_roundtrip(
        self, sample_config: LLMEvalConfig, temp_config_path: Path
    ):
        """Test config save/load round-trip preserves data."""
        # Save config
        result = save_config(sample_config)
        assert result["ok"] is True
        assert temp_config_path.exists()

        # Load config
        loaded = load_config()
        assert loaded.generator_model is not None
        assert loaded.generator_model.name == "Test-GPT-4"
        assert loaded.generator_model.api_key.get_secret_value() == "test-api-key-12345"
        assert len(loaded.evaluation_models) == 1
        assert loaded.settings.max_concurrent_calls == 5

    def test_load_config_missing_file(self, temp_config_path: Path):
        """Test loading config when file doesn't exist returns default."""
        config = load_config()
        assert config.generator_model is None
        assert len(config.evaluation_models) == 0


@pytest.mark.unit
class TestDatasetPersistence:
    """Tests for dataset save/load operations."""

    def test_save_and_load_dataset(self, temp_datasets_dir: Path):
        """Test dataset save/load round-trip."""
        document = GroundingDocument(
            filename="test.pdf",
            pages=["Sample document content for testing."],
            page_count=1,
            file_size_bytes=1024,
        )
        conversation = Conversation(
            conversation_id="conv-001",
            page_reference=0,
            turns=[
                ConversationTurn(turn_number=1, question="What is X?"),
                ConversationTurn(turn_number=2, question="Why Y?"),
            ],
        )
        dataset = QuestionDataset(
            dataset_id="test-dataset-001",
            source_document=document,
            conversations=[conversation],
        )

        # Save
        result = save_dataset(dataset)
        assert result["ok"] is True

        # Load
        loaded = load_dataset("test-dataset-001")
        assert loaded is not None
        assert loaded.dataset_id == "test-dataset-001"
        assert loaded.total_questions == 2
        assert loaded.source_document.filename == "test.pdf"

    def test_list_datasets(self, temp_datasets_dir: Path):
        """Test listing datasets."""
        # Create temp directory and files
        temp_datasets_dir.mkdir(parents=True, exist_ok=True)
        (temp_datasets_dir / "dataset-a.json").write_text("{}")
        (temp_datasets_dir / "dataset-b.json").write_text("{}")

        datasets = list_datasets()
        assert "dataset-a" in datasets
        assert "dataset-b" in datasets


@pytest.mark.unit
class TestRunPersistence:
    """Tests for evaluation run save/load operations."""

    def test_save_and_load_run(self, temp_runs_dir: Path):
        """Test run save/load round-trip."""
        run = EvaluationRun(
            run_id="run-001",
            dataset_id="dataset-001",
            model_names=["GPT-4", "Claude"],
            status=RunStatus.COMPLETED,
        )

        # Save
        result = save_run(run)
        assert result["ok"] is True

        # Load
        loaded = load_run("run-001")
        assert loaded is not None
        assert loaded.run_id == "run-001"
        assert loaded.status == RunStatus.COMPLETED

    def test_list_runs(self, temp_runs_dir: Path):
        """Test listing runs."""
        temp_runs_dir.mkdir(parents=True, exist_ok=True)
        (temp_runs_dir / "run-2024-001.json").write_text("{}")
        (temp_runs_dir / "run-2024-002.json").write_text("{}")

        runs = list_runs()
        assert "run-2024-001" in runs
        assert "run-2024-002" in runs


# ============================================================================
# EvaluationRun Tests
# ============================================================================


@pytest.mark.unit
class TestEvaluationRun:
    """Tests for EvaluationRun model and methods."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        run = EvaluationRun(
            run_id="run-001",
            dataset_id="dataset-001",
            model_names=["GPT-4"],
            results=[
                EvaluationResult(
                    model_name="GPT-4",
                    conversation_id="conv-1",
                    turn_number=1,
                    question="Q1?",
                    page_reference=0,
                    response="Response 1",
                    success=True,
                ),
                EvaluationResult(
                    model_name="GPT-4",
                    conversation_id="conv-1",
                    turn_number=2,
                    question="Q2?",
                    page_reference=0,
                    success=False,
                    error="Timeout",
                ),
            ],
        )
        assert run.success_rate == 0.5

    def test_get_model_summary(self):
        """Test model summary aggregation."""
        metrics = RAGMetrics(
            latency_ms=100,
            groundedness=4.0,
            relevance=4.5,
            coherence=4.0,
            fluency=5.0,
        )
        run = EvaluationRun(
            run_id="run-001",
            dataset_id="dataset-001",
            model_names=["GPT-4"],
            results=[
                EvaluationResult(
                    model_name="GPT-4",
                    conversation_id="conv-1",
                    turn_number=1,
                    question="Q1?",
                    page_reference=0,
                    response="Response 1",
                    latency_ms=100,
                    success=True,
                    metrics=metrics,
                ),
            ],
        )

        summary = run.get_model_summary("GPT-4")
        assert summary["count"] == 1
        assert summary["avg_latency_ms"] == 100
        assert summary["avg_groundedness"] == 4.0

    def test_get_model_summary_with_tokens(self):
        """Test model summary aggregation includes token totals."""
        metrics = RAGMetrics(
            latency_ms=100,
            groundedness=4.0,
            relevance=4.5,
            coherence=4.0,
            fluency=5.0,
        )
        run = EvaluationRun(
            run_id="run-001",
            dataset_id="dataset-001",
            model_names=["GPT-4"],
            results=[
                EvaluationResult(
                    model_name="GPT-4",
                    conversation_id="conv-1",
                    turn_number=1,
                    question="Q1?",
                    page_reference=0,
                    response="Response 1",
                    latency_ms=100,
                    success=True,
                    metrics=metrics,
                    prompt_tokens=500,
                    completion_tokens=200,
                    cached_tokens=100,
                    total_tokens=700,
                ),
                EvaluationResult(
                    model_name="GPT-4",
                    conversation_id="conv-1",
                    turn_number=2,
                    question="Q2?",
                    page_reference=0,
                    response="Response 2",
                    latency_ms=150,
                    success=True,
                    metrics=metrics,
                    prompt_tokens=600,
                    completion_tokens=250,
                    cached_tokens=150,
                    total_tokens=850,
                ),
            ],
        )

        summary = run.get_model_summary("GPT-4")
        assert summary["count"] == 2
        assert summary["total_prompt_tokens"] == 1100
        assert summary["total_completion_tokens"] == 450
        assert summary["total_cached_tokens"] == 250
        assert summary["total_tokens"] == 1550


@pytest.mark.unit
class TestEvaluationResultTokens:
    """Tests for EvaluationResult token fields."""

    def test_evaluation_result_with_tokens(self):
        """Test creating evaluation result with token fields."""
        result = EvaluationResult(
            model_name="GPT-4",
            conversation_id="conv-1",
            turn_number=1,
            question="What is X?",
            page_reference=0,
            response="X is Y.",
            success=True,
            prompt_tokens=500,
            completion_tokens=100,
            cached_tokens=50,
            total_tokens=600,
        )

        assert result.prompt_tokens == 500
        assert result.completion_tokens == 100
        assert result.cached_tokens == 50
        assert result.total_tokens == 600

    def test_evaluation_result_default_tokens(self):
        """Test EvaluationResult defaults tokens to 0."""
        result = EvaluationResult(
            model_name="GPT-4",
            conversation_id="conv-1",
            turn_number=1,
            question="What is X?",
            page_reference=0,
            success=True,
        )

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.cached_tokens == 0
        assert result.total_tokens == 0


# ============================================================================
# Chat History & Messages Validator Tests
# ============================================================================


@pytest.mark.unit
class TestChatHistoryValidator:
    """Tests for EvaluationResult.chat_history strict validation."""

    def test_valid_chat_history(self):
        """Test valid chat_history with proper alternation."""
        from helpers.llm_models import ChatHistoryItem

        result = EvaluationResult(
            model_name="GPT-4",
            conversation_id="conv-1",
            turn_number=3,
            question="Follow-up Q?",
            page_reference=0,
            success=True,
            chat_history=[
                ChatHistoryItem(role="user", content="First question"),
                ChatHistoryItem(role="assistant", content="First answer"),
                ChatHistoryItem(role="user", content="Second question"),
                ChatHistoryItem(role="assistant", content="Second answer"),
            ],
        )
        assert len(result.chat_history) == 4

    def test_chat_history_none_allowed(self):
        """Test None chat_history is valid (turn 1)."""
        result = EvaluationResult(
            model_name="GPT-4",
            conversation_id="conv-1",
            turn_number=1,
            question="First Q?",
            page_reference=0,
            success=True,
            chat_history=None,
        )
        assert result.chat_history is None

    def test_chat_history_empty_allowed(self):
        """Test empty chat_history is valid."""
        result = EvaluationResult(
            model_name="GPT-4",
            conversation_id="conv-1",
            turn_number=1,
            question="First Q?",
            page_reference=0,
            success=True,
            chat_history=[],
        )
        assert result.chat_history == []

    def test_chat_history_must_start_with_user(self):
        """Test chat_history must start with 'user' role."""
        from helpers.llm_models import ChatHistoryItem

        with pytest.raises(ValueError, match="must start with 'user'"):
            EvaluationResult(
                model_name="GPT-4",
                conversation_id="conv-1",
                turn_number=2,
                question="Q?",
                page_reference=0,
                success=True,
                chat_history=[
                    ChatHistoryItem(role="assistant", content="Wrong start"),
                ],
            )

    def test_chat_history_must_alternate(self):
        """Test chat_history must alternate user -> assistant."""
        from helpers.llm_models import ChatHistoryItem

        with pytest.raises(ValueError, match="must alternate"):
            EvaluationResult(
                model_name="GPT-4",
                conversation_id="conv-1",
                turn_number=3,
                question="Q?",
                page_reference=0,
                success=True,
                chat_history=[
                    ChatHistoryItem(role="user", content="Q1"),
                    ChatHistoryItem(role="user", content="Q2"),  # Should be assistant
                ],
            )

    def test_chat_history_invalid_role(self):
        """Test chat_history rejects invalid roles."""
        from helpers.llm_models import ChatHistoryItem

        with pytest.raises(ValueError, match="invalid role"):
            EvaluationResult(
                model_name="GPT-4",
                conversation_id="conv-1",
                turn_number=2,
                question="Q?",
                page_reference=0,
                success=True,
                chat_history=[
                    ChatHistoryItem(role="user", content="Q1"),
                    ChatHistoryItem(role="system", content="Invalid"),  # Invalid role
                ],
            )


@pytest.mark.unit
class TestMessagesValidator:
    """Tests for EvaluationResult.messages strict validation."""

    def test_valid_messages(self):
        """Test valid messages with proper format."""
        result = EvaluationResult(
            model_name="GPT-4",
            conversation_id="conv-1",
            turn_number=1,
            question="Q?",
            page_reference=0,
            success=True,
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
        )
        assert len(result.messages) == 3

    def test_messages_multi_turn(self):
        """Test valid multi-turn messages."""
        result = EvaluationResult(
            model_name="GPT-4",
            conversation_id="conv-1",
            turn_number=2,
            question="Follow-up?",
            page_reference=0,
            success=True,
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "First Q"},
                {"role": "assistant", "content": "First A"},
                {"role": "user", "content": "Second Q"},
                {"role": "assistant", "content": "Second A"},
            ],
        )
        assert len(result.messages) == 5

    def test_messages_none_allowed(self):
        """Test None messages is valid (failed evaluation)."""
        result = EvaluationResult(
            model_name="GPT-4",
            conversation_id="conv-1",
            turn_number=1,
            question="Q?",
            page_reference=0,
            success=False,
            error="Timeout",
            messages=None,
        )
        assert result.messages is None

    def test_messages_must_start_with_system(self):
        """Test messages must start with 'system' role."""
        with pytest.raises(ValueError, match="must start with 'system'"):
            EvaluationResult(
                model_name="GPT-4",
                conversation_id="conv-1",
                turn_number=1,
                question="Q?",
                page_reference=0,
                success=True,
                messages=[
                    {"role": "user", "content": "Wrong start"},
                    {"role": "assistant", "content": "Answer"},
                ],
            )

    def test_messages_must_end_with_assistant(self):
        """Test messages must end with 'assistant' (includes response)."""
        with pytest.raises(ValueError, match="must end with 'assistant'"):
            EvaluationResult(
                model_name="GPT-4",
                conversation_id="conv-1",
                turn_number=1,
                question="Q?",
                page_reference=0,
                success=True,
                messages=[
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Question"},
                ],
            )

    def test_messages_must_alternate_after_system(self):
        """Test messages must alternate user -> assistant after system."""
        with pytest.raises(ValueError, match="must alternate"):
            EvaluationResult(
                model_name="GPT-4",
                conversation_id="conv-1",
                turn_number=1,
                question="Q?",
                page_reference=0,
                success=True,
                messages=[
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Q1"},
                    {"role": "user", "content": "Q2"},  # Should be assistant
                    {"role": "assistant", "content": "Answer"},
                ],
            )

    def test_messages_invalid_role(self):
        """Test messages rejects invalid roles after system."""
        with pytest.raises(ValueError, match="invalid role"):
            EvaluationResult(
                model_name="GPT-4",
                conversation_id="conv-1",
                turn_number=1,
                question="Q?",
                page_reference=0,
                success=True,
                messages=[
                    {"role": "system", "content": "You are helpful"},
                    {"role": "function", "content": "Invalid"},  # Invalid role
                    {"role": "assistant", "content": "Answer"},
                ],
            )


# ============================================================================
# Model Connection Validation Tests (US1)
# ============================================================================


@pytest.mark.unit
class TestValidateModelConnection:
    """Tests for validate_model_connection function (US1)."""

    @pytest.mark.asyncio
    async def test_validate_connection_success(self, sample_llm_model: LLMModel):
        """Test successful model connection validation."""
        from helpers.llm_models import validate_model_connection

        # Mock the OpenAI client to return a successful response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]

        with (
            patch("helpers.llm_models.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.llm_models.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await validate_model_connection(sample_llm_model)

            assert result["ok"] is True
            assert result["model_name"] == "Test-GPT-4"
            assert "latency_ms" in result
            assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_validate_connection_invalid_endpoint(self, sample_llm_model: LLMModel):
        """Test connection validation with invalid endpoint."""
        from helpers.llm_models import validate_model_connection

        with (
            patch("helpers.llm_models.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.llm_models.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Connection refused")
            )
            mock_client_class.return_value = mock_client

            result = await validate_model_connection(sample_llm_model)

            assert result["ok"] is False
            assert "error" in result
            assert result["model_name"] == "Test-GPT-4"

    @pytest.mark.asyncio
    async def test_validate_connection_invalid_api_key(self, sample_llm_model: LLMModel):
        """Test connection validation with invalid API key."""
        from openai import AuthenticationError

        from helpers.llm_models import validate_model_connection

        with (
            patch("helpers.llm_models.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.llm_models.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=AuthenticationError(
                    message="Invalid API key",
                    response=MagicMock(status_code=401),
                    body=None,
                )
            )
            mock_client_class.return_value = mock_client

            result = await validate_model_connection(sample_llm_model)

            assert result["ok"] is False
            assert "Invalid API key" in result["error"]


class AsyncMock(MagicMock):
    """Mock class for async functions."""

    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
