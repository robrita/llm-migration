"""
Unit tests for helpers/rag_metrics.py

Tests cover:
- RAG metrics calculation with mocked LLM judge
- Aggregate run metrics calculation
- Score validation (1-5 range)
"""

from unittest.mock import MagicMock, patch

import pytest


class AsyncMock(MagicMock):
    """Mock class for async functions."""

    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_evaluation_result():
    """Sample evaluation result for testing."""
    from helpers.llm_models import EvaluationResult, RAGMetrics

    return EvaluationResult(
        model_name="Test-Model",
        conversation_id="conv-001",
        turn_number=1,
        question="What is machine learning?",
        response="Machine learning is a subset of AI.",
        page_reference=0,
        latency_ms=150.0,
        success=True,
        metrics=RAGMetrics(),
    )


@pytest.fixture
def sample_reference_content():
    """Sample page content for testing."""
    return "Machine learning is AI that learns from data."


@pytest.fixture
def mock_judge_response_valid():
    """Valid JSON response from judge LLM."""
    return """{
        "groundedness": 4,
        "relevance": 5,
        "coherence": 4,
        "fluency": 5
    }"""


# ============================================================================
# Test Cases
# ============================================================================


@pytest.mark.unit
class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    @pytest.mark.asyncio
    async def test_calculate_metrics_success(
        self, sample_evaluation_result, sample_reference_content, mock_judge_response_valid
    ):
        """Test successful metrics calculation."""
        from helpers.llm_models import LLMModel
        from helpers.rag_metrics import calculate_metrics

        judge_model = LLMModel(
            name="Judge-Model",
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=mock_judge_response_valid))]

        with (
            patch("helpers.rag_metrics.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.rag_metrics.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await calculate_metrics(
                evaluation_result=sample_evaluation_result,
                judge_model=judge_model,
                reference_content=sample_reference_content,
            )

            assert result["ok"] is True
            assert "metrics" in result
            metrics = result["metrics"]
            assert metrics.groundedness == 4
            assert metrics.relevance == 5
            assert metrics.coherence == 4
            assert metrics.fluency == 5

    @pytest.mark.asyncio
    async def test_calculate_metrics_api_error(
        self, sample_evaluation_result, sample_reference_content
    ):
        """Test handling of API errors during metrics calculation."""
        from helpers.llm_models import LLMModel
        from helpers.rag_metrics import calculate_metrics

        judge_model = LLMModel(
            name="Judge-Model",
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

        with (
            patch("helpers.rag_metrics.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.rag_metrics.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
            mock_client_class.return_value = mock_client

            result = await calculate_metrics(
                evaluation_result=sample_evaluation_result,
                judge_model=judge_model,
                reference_content=sample_reference_content,
            )

            assert result["ok"] is False
            assert "error" in result


@pytest.mark.unit
class TestAggregateRunMetrics:
    """Tests for aggregate_run_metrics function."""

    def test_aggregate_run_metrics(self):
        """Test aggregating metrics across multiple results."""
        from helpers.llm_models import EvaluationResult, RAGMetrics
        from helpers.rag_metrics import aggregate_run_metrics

        results = [
            EvaluationResult(
                model_name="Model-A",
                conversation_id="conv-001",
                turn_number=1,
                question="Q1",
                response="R1",
                page_reference=0,
                latency_ms=100.0,
                success=True,
                metrics=RAGMetrics(groundedness=4, relevance=5, coherence=4, fluency=5),
            ),
            EvaluationResult(
                model_name="Model-A",
                conversation_id="conv-002",
                turn_number=1,
                question="Q2",
                response="R2",
                page_reference=0,
                latency_ms=200.0,
                success=True,
                metrics=RAGMetrics(groundedness=3, relevance=4, coherence=5, fluency=4),
            ),
            EvaluationResult(
                model_name="Model-B",
                conversation_id="conv-001",
                turn_number=1,
                question="Q1",
                response="R1",
                page_reference=0,
                latency_ms=150.0,
                success=True,
                metrics=RAGMetrics(groundedness=5, relevance=5, coherence=5, fluency=5),
            ),
        ]

        summary = aggregate_run_metrics(results)

        assert "Model-A" in summary
        assert "Model-B" in summary
        assert summary["Model-A"]["count"] == 2
        assert summary["Model-B"]["count"] == 1
        assert summary["Model-A"]["avg_latency_ms"] == 150.0  # (100 + 200) / 2
        assert summary["Model-A"]["avg_groundedness"] == 3.5  # (4 + 3) / 2

    def test_aggregate_with_failed_results(self):
        """Test that failed results are excluded from averages."""
        from helpers.llm_models import EvaluationResult, RAGMetrics
        from helpers.rag_metrics import aggregate_run_metrics

        results = [
            EvaluationResult(
                model_name="Model-A",
                conversation_id="conv-001",
                turn_number=1,
                question="Q1",
                response="R1",
                page_reference=0,
                latency_ms=100.0,
                success=True,
                metrics=RAGMetrics(groundedness=4, relevance=4, coherence=4, fluency=4),
            ),
            EvaluationResult(
                model_name="Model-A",
                conversation_id="conv-002",
                turn_number=1,
                question="Q2",
                response="",
                page_reference=0,
                latency_ms=0,
                success=False,
                error="API Error",
                metrics=RAGMetrics(),
            ),
        ]

        summary = aggregate_run_metrics(results)

        assert summary["Model-A"]["count"] == 1
        assert summary["Model-A"]["failed"] == 1


@pytest.mark.unit
class TestScoreValidation:
    """Tests for score validation."""

    def test_score_range_validation(self):
        """Test that scores are validated to 1-5 range."""
        from pydantic import ValidationError

        from helpers.llm_models import RAGMetrics

        # Valid range
        metrics = RAGMetrics(groundedness=3, relevance=4, coherence=5, fluency=1)
        assert metrics.groundedness == 3

        # Invalid - too high
        with pytest.raises(ValidationError):
            RAGMetrics(groundedness=6)

        # Invalid - too low
        with pytest.raises(ValidationError):
            RAGMetrics(groundedness=0)

    def test_default_scores(self):
        """Test default score values."""
        from helpers.llm_models import RAGMetrics

        metrics = RAGMetrics()
        assert metrics.groundedness is None
        assert metrics.relevance is None


@pytest.mark.unit
class TestFormatConversationHistory:
    """Tests for format_conversation_history function."""

    def test_format_empty_history(self):
        """Test formatting with no history returns empty string."""
        from helpers.rag_metrics import format_conversation_history

        assert format_conversation_history(None) == ""
        assert format_conversation_history([]) == ""

    def test_format_single_turn_history(self):
        """Test formatting single turn of history."""
        from helpers.llm_models import ChatHistoryItem
        from helpers.rag_metrics import format_conversation_history

        history = [
            ChatHistoryItem(role="user", content="What is Azure?"),
            ChatHistoryItem(role="assistant", content="Azure is Microsoft's cloud platform."),
        ]

        formatted = format_conversation_history(history)

        assert "**Prior Conversation**:" in formatted
        assert "User: What is Azure?" in formatted
        assert "Assistant: Azure is Microsoft's cloud platform." in formatted

    def test_format_multi_turn_history(self):
        """Test formatting multiple turns of history."""
        from helpers.llm_models import ChatHistoryItem
        from helpers.rag_metrics import format_conversation_history

        history = [
            ChatHistoryItem(role="user", content="Question 1"),
            ChatHistoryItem(role="assistant", content="Answer 1"),
            ChatHistoryItem(role="user", content="Question 2"),
            ChatHistoryItem(role="assistant", content="Answer 2"),
        ]

        formatted = format_conversation_history(history)

        assert formatted.count("User:") == 2
        assert formatted.count("Assistant:") == 2

    def test_format_truncates_long_messages(self):
        """Test that long messages are truncated."""
        from helpers.llm_models import ChatHistoryItem
        from helpers.rag_metrics import format_conversation_history

        long_content = "A" * 600  # Exceeds 500 char limit
        history = [ChatHistoryItem(role="user", content=long_content)]

        formatted = format_conversation_history(history)

        assert "..." in formatted
        assert len(formatted) < 700  # Truncated + prefix


@pytest.mark.unit
class TestMultiTurnCalculateMetrics:
    """Tests for calculate_metrics with conversation history."""

    @pytest.mark.asyncio
    async def test_calculate_metrics_with_chat_history(self, mock_judge_response_valid):
        """Test metrics calculation includes chat history in prompt."""
        from helpers.llm_models import ChatHistoryItem, EvaluationResult, LLMModel, RAGMetrics
        from helpers.rag_metrics import calculate_metrics

        # Create evaluation result with chat history (turn 2)
        eval_result = EvaluationResult(
            model_name="Test-Model",
            conversation_id="conv-001",
            turn_number=2,
            question="Which models are available?",
            response="GPT-4 and GPT-4o are available.",
            page_reference=0,
            latency_ms=150.0,
            success=True,
            metrics=RAGMetrics(),
            chat_history=[
                ChatHistoryItem(role="user", content="What AI services does Azure provide?"),
                ChatHistoryItem(role="assistant", content="Azure provides Azure OpenAI Service."),
            ],
        )

        judge_model = LLMModel(
            name="Judge-Model",
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=mock_judge_response_valid))]

        captured_prompt = None

        async def capture_create(*args, **kwargs):
            nonlocal captured_prompt
            messages = kwargs.get("messages", [])
            if messages:
                captured_prompt = messages[-1].get("content", "")
            return mock_response

        with (
            patch("helpers.rag_metrics.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.rag_metrics.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = capture_create
            mock_client_class.return_value = mock_client

            result = await calculate_metrics(
                evaluation_result=eval_result,
                judge_model=judge_model,
                reference_content="Azure offers GPT-4, GPT-4o, and other models.",
            )

            assert result["ok"] is True
            # Verify chat history was included in the prompt
            assert captured_prompt is not None
            assert "Prior Conversation" in captured_prompt
            assert "What AI services does Azure provide?" in captured_prompt
            assert "Azure provides Azure OpenAI Service." in captured_prompt

    @pytest.mark.asyncio
    async def test_calculate_metrics_without_chat_history(
        self, sample_evaluation_result, sample_reference_content, mock_judge_response_valid
    ):
        """Test metrics calculation works without chat history (turn 1)."""
        from helpers.llm_models import LLMModel
        from helpers.rag_metrics import calculate_metrics

        judge_model = LLMModel(
            name="Judge-Model",
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=mock_judge_response_valid))]

        captured_prompt = None

        async def capture_create(*args, **kwargs):
            nonlocal captured_prompt
            messages = kwargs.get("messages", [])
            if messages:
                captured_prompt = messages[-1].get("content", "")
            return mock_response

        with (
            patch("helpers.rag_metrics.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.rag_metrics.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = capture_create
            mock_client_class.return_value = mock_client

            result = await calculate_metrics(
                evaluation_result=sample_evaluation_result,
                judge_model=judge_model,
                reference_content=sample_reference_content,
            )

            assert result["ok"] is True
            # Verify no chat history section for turn 1
            assert captured_prompt is not None
            assert "Prior Conversation" not in captured_prompt


@pytest.mark.unit
class TestCalculateAllMetricsParallel:
    """Tests for parallel execution of calculate_all_metrics."""

    @pytest.mark.asyncio
    async def test_calculate_all_metrics_parallel_execution(self, mock_judge_response_valid):
        """Test that calculate_all_metrics runs scoring in parallel."""
        import asyncio

        from helpers.llm_models import EvaluationResult, LLMModel, RAGMetrics
        from helpers.rag_metrics import calculate_all_metrics

        # Create multiple evaluation results
        results = [
            EvaluationResult(
                model_name=f"Model-{i}",
                conversation_id=f"conv-{i:03d}",
                turn_number=1,
                question=f"Question {i}",
                response=f"Response {i}",
                page_reference=0,
                latency_ms=100.0,
                success=True,
                metrics=RAGMetrics(),
            )
            for i in range(5)
        ]

        judge_model = LLMModel(
            name="Judge-Model",
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=mock_judge_response_valid))]

        call_times = []

        async def mock_create(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.05)  # Simulate API latency
            return mock_response

        with (
            patch("helpers.rag_metrics.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.rag_metrics.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_client.close = AsyncMock()  # Mock async close for cleanup
            mock_client_class.return_value = mock_client

            scored_results = await calculate_all_metrics(
                results=results,
                judge_model=judge_model,
                pages=["Ground truth content for testing."],
                max_concurrent=5,  # Allow all 5 to run in parallel
            )

            # Verify all results have metrics
            assert len(scored_results) == 5
            for result in scored_results:
                assert result.metrics is not None
                assert result.metrics.groundedness == 4

            # Verify parallel execution: calls should start nearly simultaneously
            # If sequential, time difference would be ~0.05s between each call
            # If parallel, all calls start within a small window
            assert len(call_times) == 5
            time_spread = max(call_times) - min(call_times)
            # All calls should start within 0.02s of each other (parallel)
            assert time_spread < 0.02, f"Calls not parallel: spread={time_spread}s"

    @pytest.mark.asyncio
    async def test_calculate_all_metrics_respects_max_concurrent(self, mock_judge_response_valid):
        """Test that max_concurrent limits parallel execution."""
        import asyncio

        from helpers.llm_models import EvaluationResult, LLMModel, RAGMetrics
        from helpers.rag_metrics import calculate_all_metrics

        # Create 6 evaluation results
        results = [
            EvaluationResult(
                model_name=f"Model-{i}",
                conversation_id=f"conv-{i:03d}",
                turn_number=1,
                question=f"Question {i}",
                response=f"Response {i}",
                page_reference=0,
                latency_ms=100.0,
                success=True,
                metrics=RAGMetrics(),
            )
            for i in range(6)
        ]

        judge_model = LLMModel(
            name="Judge-Model",
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=mock_judge_response_valid))]

        concurrent_count = [0]
        max_observed_concurrent = [0]

        async def mock_create(*args, **kwargs):
            concurrent_count[0] += 1
            max_observed_concurrent[0] = max(max_observed_concurrent[0], concurrent_count[0])
            await asyncio.sleep(0.05)
            concurrent_count[0] -= 1
            return mock_response

        with (
            patch("helpers.rag_metrics.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.rag_metrics.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_client.close = AsyncMock()  # Mock async close for cleanup
            mock_client_class.return_value = mock_client

            await calculate_all_metrics(
                results=results,
                judge_model=judge_model,
                pages=["Ground truth content for testing."],
                max_concurrent=2,  # Only allow 2 at a time
            )

            # Verify concurrency was limited to 2
            assert max_observed_concurrent[0] <= 2

    @pytest.mark.asyncio
    async def test_calculate_all_metrics_progress_callback(self, mock_judge_response_valid):
        """Test that progress callback is called correctly."""
        from helpers.llm_models import EvaluationResult, LLMModel, RAGMetrics
        from helpers.rag_metrics import calculate_all_metrics

        results = [
            EvaluationResult(
                model_name="Model-A",
                conversation_id="conv-001",
                turn_number=1,
                question="Q1",
                response="R1",
                page_reference=0,
                latency_ms=100.0,
                success=True,
                metrics=RAGMetrics(),
            ),
            EvaluationResult(
                model_name="Model-A",
                conversation_id="conv-002",
                turn_number=1,
                question="Q2",
                response="R2",
                page_reference=0,
                latency_ms=100.0,
                success=True,
                metrics=RAGMetrics(),
            ),
            # Failed result (not scorable)
            EvaluationResult(
                model_name="Model-A",
                conversation_id="conv-003",
                turn_number=1,
                question="Q3",
                response="",
                page_reference=0,
                latency_ms=0,
                success=False,
                error="API Error",
                metrics=RAGMetrics(),
            ),
        ]

        judge_model = LLMModel(
            name="Judge-Model",
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=mock_judge_response_valid))]

        progress_calls = []

        def progress_callback(completed, total):
            progress_calls.append((completed, total))

        with (
            patch("helpers.rag_metrics.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.rag_metrics.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()  # Mock async close for cleanup
            mock_client_class.return_value = mock_client

            await calculate_all_metrics(
                results=results,
                judge_model=judge_model,
                pages=["Ground truth content for testing."],
                progress_callback=progress_callback,
            )

            # Should have 2 progress calls (one per scorable result)
            assert len(progress_calls) == 2
            # All calls should have total=3
            assert all(total == 3 for _, total in progress_calls)
            # Final call should indicate completion
            final_completed = max(c for c, _ in progress_calls)
            assert final_completed == 3  # 1 non-scorable + 2 scorable

    @pytest.mark.asyncio
    async def test_calculate_all_metrics_client_reuse(self, mock_judge_response_valid):
        """Test that a single client is created and reused."""
        from helpers.llm_models import EvaluationResult, LLMModel, RAGMetrics
        from helpers.rag_metrics import calculate_all_metrics

        results = [
            EvaluationResult(
                model_name=f"Model-{i}",
                conversation_id=f"conv-{i:03d}",
                turn_number=1,
                question=f"Q{i}",
                response=f"R{i}",
                page_reference=0,
                latency_ms=100.0,
                success=True,
                metrics=RAGMetrics(),
            )
            for i in range(3)
        ]

        judge_model = LLMModel(
            name="Judge-Model",
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=mock_judge_response_valid))]

        with (
            patch("helpers.rag_metrics.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.rag_metrics.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()  # Mock async close for cleanup
            mock_client_class.return_value = mock_client

            await calculate_all_metrics(
                results=results,
                judge_model=judge_model,
                pages=["Ground truth content for testing."],
            )

            # Client should be created only once
            assert mock_client_class.call_count == 1
