"""
Unit tests for helpers/llm_eval.py

Tests cover:
- Single model evaluation with mocked API
- Timeout behavior
- Retry logic for 429/503 errors
- Parallel execution with asyncio.gather()
- Conversation history trimming
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
def sample_conversation():
    """Sample conversation for testing."""
    from helpers.llm_models import Conversation, ConversationTurn

    return Conversation(
        conversation_id="test-conv-001",
        page_reference=0,
        turns=[
            ConversationTurn(
                turn_number=1,
                question="What is machine learning?",
            ),
            ConversationTurn(
                turn_number=2,
                question="What are its applications?",
            ),
        ],
    )


@pytest.fixture
def sample_model():
    """Sample model for testing."""
    from helpers.llm_models import LLMModel

    return LLMModel(
        name="Test-Model",
        endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        deployment_name="gpt-4",
    )


@pytest.fixture
def sample_grounding_document():
    """Sample grounding document for testing."""
    from helpers.llm_models import GroundingDocument

    return GroundingDocument(
        filename="test.pdf",
        pages=["Machine learning is a subset of AI for data-driven learning."],
        page_count=1,
        file_size_bytes=1000,
    )


@pytest.fixture
def sample_page_content():
    """Sample page content for testing."""
    return "Machine learning is a subset of AI for data-driven learning."


# ============================================================================
# Test Cases for evaluate_single
# ============================================================================


@pytest.mark.unit
class TestEvaluateSingle:
    """Tests for evaluate_single function."""

    @pytest.mark.asyncio
    async def test_evaluate_single_success(
        self, sample_model, sample_conversation, sample_page_content
    ):
        """Test successful single evaluation."""
        from helpers.llm_eval import evaluate_single

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Machine learning enables data-driven learning."))
        ]

        # Create mock client directly (no patching needed since we pass client)
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await evaluate_single(
            client=mock_client,
            deployment_name="gpt-4",
            model=sample_model,
            conversation=sample_conversation,
            page_content=sample_page_content,
            turn_index=0,
        )

        assert result["ok"] is True
        assert "response" in result
        assert "latency_ms" in result
        assert "api_latency_ms" in result
        assert result["model_name"] == "Test-Model"

    @pytest.mark.asyncio
    async def test_evaluate_single_returns_tokens(
        self, sample_model, sample_conversation, sample_page_content
    ):
        """Test that evaluate_single returns token counts from the response."""
        from helpers.llm_eval import evaluate_single

        # Create mock response with usage data
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 500
        mock_usage.completion_tokens = 150
        mock_usage.total_tokens = 650
        mock_usage.prompt_tokens_details = MagicMock(cached_tokens=100)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response with tokens."))]
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await evaluate_single(
            client=mock_client,
            deployment_name="gpt-4",
            model=sample_model,
            conversation=sample_conversation,
            page_content=sample_page_content,
            turn_index=0,
        )

        assert result["ok"] is True
        assert result["prompt_tokens"] == 500
        assert result["completion_tokens"] == 150
        assert result["total_tokens"] == 650
        assert result["cached_tokens"] == 100

    @pytest.mark.asyncio
    async def test_evaluate_single_handles_missing_usage(
        self, sample_model, sample_conversation, sample_page_content
    ):
        """Test evaluate_single handles response without usage data."""
        from helpers.llm_eval import evaluate_single

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response without usage."))]
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await evaluate_single(
            client=mock_client,
            deployment_name="gpt-4",
            model=sample_model,
            conversation=sample_conversation,
            page_content=sample_page_content,
            turn_index=0,
        )

        assert result["ok"] is True
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0
        assert result["total_tokens"] == 0
        assert result["cached_tokens"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_single_api_error(
        self, sample_model, sample_conversation, sample_page_content
    ):
        """Test handling of API errors."""
        from helpers.llm_eval import evaluate_single

        # Create mock client that raises exception
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        result = await evaluate_single(
            client=mock_client,
            deployment_name="gpt-4",
            model=sample_model,
            conversation=sample_conversation,
            page_content=sample_page_content,
            turn_index=0,
        )

        assert result["ok"] is False
        assert "error" in result
        assert result["api_latency_ms"] == 0.0
        # Token fields should be 0 on error
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0
        assert result["total_tokens"] == 0
        assert result["cached_tokens"] == 0


@pytest.mark.unit
class TestEvaluateSingleTimeout:
    """Tests for timeout behavior."""

    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_model, sample_conversation, sample_page_content):
        """Test that timeout is handled gracefully."""
        import asyncio

        from helpers.llm_eval import evaluate_single

        async def slow_api_call(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate slow response
            return MagicMock()

        # Create mock client with slow response
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = slow_api_call

        result = await evaluate_single(
            client=mock_client,
            deployment_name="gpt-4",
            model=sample_model,
            conversation=sample_conversation,
            page_content=sample_page_content,
            turn_index=0,
            timeout_seconds=0.1,  # Very short timeout for test
        )

        assert result["ok"] is False
        assert "timeout" in result.get("error", "").lower()
        assert result["api_latency_ms"] == 0.0


@pytest.mark.unit
class TestRetryLogic:
    """Tests for retry logic on 429/503 errors."""

    @pytest.mark.asyncio
    async def test_retry_on_429(self, sample_model, sample_conversation, sample_page_content):
        """Test retry on 429 (rate limit) error."""
        from openai import RateLimitError

        from helpers.llm_eval import evaluate_single

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Success after retry"))]

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Create a proper mock response for RateLimitError
                mock_resp = MagicMock()
                mock_resp.status_code = 429
                mock_resp.headers = {}
                raise RateLimitError(
                    message="Rate limit exceeded",
                    response=mock_resp,
                    body={"error": {"message": "Rate limit exceeded"}},
                )
            return mock_response

        # Create mock client with retry behavior
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = mock_create

        result = await evaluate_single(
            client=mock_client,
            deployment_name="gpt-4",
            model=sample_model,
            conversation=sample_conversation,
            page_content=sample_page_content,
            turn_index=0,
            max_retries=3,
        )

        assert result["ok"] is True
        assert call_count == 3
        assert "api_latency_ms" in result  # Field should be present
        assert result["latency_ms"] > 0  # Total latency includes retry delays


@pytest.mark.unit
class TestParallelExecution:
    """Tests for parallel execution."""

    @pytest.mark.asyncio
    async def test_run_parallel_evaluations(
        self, sample_model, sample_conversation, sample_grounding_document
    ):
        """Test parallel execution across multiple models."""
        from helpers.llm_eval import run_parallel_evaluations
        from helpers.llm_models import LLMModel, QuestionDataset

        models = [
            sample_model,
            LLMModel(
                name="Model-2",
                endpoint="https://test2.openai.azure.com/",
                api_key="key2",
            ),
        ]

        dataset = QuestionDataset(
            dataset_id="ds-test",
            name="Test Dataset",
            source_document=sample_grounding_document,
            conversations=[sample_conversation],
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]

        with (
            patch("helpers.llm_eval.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.llm_eval.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()  # Mock async close for cleanup
            mock_client_class.return_value = mock_client

            results = await run_parallel_evaluations(
                models=models,
                dataset=dataset,
                max_concurrent=5,
            )

            assert len(results) > 0
            # Each model should have results for each turn
            assert any(r["model_name"] == "Test-Model" for r in results)


@pytest.mark.unit
class TestConversationHistory:
    """Tests for conversation history trimming."""

    def test_build_conversation_history_basic(self, sample_conversation):
        """Test building conversation history."""
        from helpers.llm_eval import build_conversation_history

        history = build_conversation_history(
            conversation=sample_conversation,
            responses=["First response"],
            current_turn_index=1,
            max_history=10,
        )

        # Should include the first turn's Q&A
        assert len(history) >= 2  # At least user + assistant messages

    def test_build_conversation_history_trimming(self):
        """Test that history is trimmed to max_history pairs."""
        from helpers.llm_eval import build_conversation_history
        from helpers.llm_models import Conversation, ConversationTurn

        # Create a conversation with many turns
        turns = [ConversationTurn(turn_number=i, question=f"Question {i}") for i in range(1, 15)]
        conversation = Conversation(conversation_id="long-conv", page_reference=0, turns=turns)

        responses = [f"Response {i}" for i in range(1, 11)]

        history = build_conversation_history(
            conversation=conversation,
            responses=responses,
            current_turn_index=10,
            max_history=5,  # Only keep 5 pairs
        )

        # Should be trimmed: 5 pairs = 10 messages + 1 system + 1 current question = 12 total
        assert len(history) <= 12


# ============================================================================
# Tests for build_evaluation_run chat_history isolation
# ============================================================================


@pytest.mark.unit
class TestBuildEvaluationRunChatHistory:
    """Tests for build_evaluation_run chat_history isolation per model."""

    def test_chat_history_isolates_per_model(self, sample_grounding_document):
        """Test that chat_history only contains responses from the same model."""
        from helpers.llm_eval import build_evaluation_run
        from helpers.llm_models import (
            Conversation,
            ConversationTurn,
            LLMModel,
            QuestionDataset,
        )

        # Create dataset with one conversation with 2 turns
        conversation = Conversation(
            conversation_id="conv-001",
            page_reference=0,
            turns=[
                ConversationTurn(turn_number=1, question="What is X?"),
                ConversationTurn(turn_number=2, question="Tell me more about X."),
            ],
        )
        dataset = QuestionDataset(
            dataset_id="ds-test",
            source_document=sample_grounding_document,
            conversations=[conversation],
        )

        models = [
            LLMModel(name="ModelA", endpoint="https://a.com", api_key="keyA"),
            LLMModel(name="ModelB", endpoint="https://b.com", api_key="keyB"),
        ]

        # Simulate results from two models
        results = [
            # ModelA Turn 1
            {
                "ok": True,
                "response": "ModelA response to turn 1",
                "model_name": "ModelA",
                "conversation_id": "conv-001",
                "turn_number": 1,
                "page_reference": 0,
                "latency_ms": 100,
                "api_latency_ms": 80,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cached_tokens": 0,
                "total_tokens": 150,
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "What is X?"},
                    {"role": "assistant", "content": "ModelA response to turn 1"},
                ],
            },
            # ModelA Turn 2
            {
                "ok": True,
                "response": "ModelA response to turn 2",
                "model_name": "ModelA",
                "conversation_id": "conv-001",
                "turn_number": 2,
                "page_reference": 0,
                "latency_ms": 120,
                "api_latency_ms": 90,
                "prompt_tokens": 150,
                "completion_tokens": 60,
                "cached_tokens": 0,
                "total_tokens": 210,
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "What is X?"},
                    {"role": "assistant", "content": "ModelA response to turn 1"},
                    {"role": "user", "content": "Tell me more about X."},
                    {"role": "assistant", "content": "ModelA response to turn 2"},
                ],
            },
            # ModelB Turn 1
            {
                "ok": True,
                "response": "ModelB response to turn 1",
                "model_name": "ModelB",
                "conversation_id": "conv-001",
                "turn_number": 1,
                "page_reference": 0,
                "latency_ms": 110,
                "api_latency_ms": 85,
                "prompt_tokens": 100,
                "completion_tokens": 55,
                "cached_tokens": 0,
                "total_tokens": 155,
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "What is X?"},
                    {"role": "assistant", "content": "ModelB response to turn 1"},
                ],
            },
            # ModelB Turn 2
            {
                "ok": True,
                "response": "ModelB response to turn 2",
                "model_name": "ModelB",
                "conversation_id": "conv-001",
                "turn_number": 2,
                "page_reference": 0,
                "latency_ms": 130,
                "api_latency_ms": 95,
                "prompt_tokens": 160,
                "completion_tokens": 65,
                "cached_tokens": 0,
                "total_tokens": 225,
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "What is X?"},
                    {"role": "assistant", "content": "ModelB response to turn 1"},
                    {"role": "user", "content": "Tell me more about X."},
                    {"role": "assistant", "content": "ModelB response to turn 2"},
                ],
            },
        ]

        run = build_evaluation_run(
            run_id="run-test",
            dataset=dataset,
            models=models,
            results=results,
        )

        # Find ModelA turn 2 result
        model_a_turn_2 = next(
            r for r in run.results if r.model_name == "ModelA" and r.turn_number == 2
        )

        # chat_history should ONLY contain ModelA's prior turns
        assert model_a_turn_2.chat_history is not None
        assert len(model_a_turn_2.chat_history) == 2  # user + assistant from turn 1
        assert model_a_turn_2.chat_history[0].role == "user"
        assert model_a_turn_2.chat_history[0].content == "What is X?"
        assert model_a_turn_2.chat_history[1].role == "assistant"
        assert "ModelA" in model_a_turn_2.chat_history[1].content
        assert "ModelB" not in model_a_turn_2.chat_history[1].content

        # Find ModelB turn 2 result
        model_b_turn_2 = next(
            r for r in run.results if r.model_name == "ModelB" and r.turn_number == 2
        )

        # chat_history should ONLY contain ModelB's prior turns
        assert model_b_turn_2.chat_history is not None
        assert len(model_b_turn_2.chat_history) == 2
        assert "ModelB" in model_b_turn_2.chat_history[1].content
        assert "ModelA" not in model_b_turn_2.chat_history[1].content

    def test_chat_history_none_for_turn_1(self, sample_grounding_document):
        """Test that turn 1 has no chat_history."""
        from helpers.llm_eval import build_evaluation_run
        from helpers.llm_models import (
            Conversation,
            ConversationTurn,
            LLMModel,
            QuestionDataset,
        )

        conversation = Conversation(
            conversation_id="conv-001",
            page_reference=0,
            turns=[ConversationTurn(turn_number=1, question="First Q?")],
        )
        dataset = QuestionDataset(
            dataset_id="ds-test",
            source_document=sample_grounding_document,
            conversations=[conversation],
        )

        models = [LLMModel(name="Model", endpoint="https://a.com", api_key="key")]

        results = [
            {
                "ok": True,
                "response": "First response",
                "model_name": "Model",
                "conversation_id": "conv-001",
                "turn_number": 1,
                "page_reference": 0,
                "latency_ms": 100,
                "api_latency_ms": 80,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cached_tokens": 0,
                "total_tokens": 150,
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "First Q?"},
                    {"role": "assistant", "content": "First response"},
                ],
            },
        ]

        run = build_evaluation_run(
            run_id="run-test",
            dataset=dataset,
            models=models,
            results=results,
        )

        # Turn 1 should have no chat_history
        turn_1_result = run.results[0]
        assert turn_1_result.chat_history is None

    def test_chat_history_isolates_by_page_reference(self):
        """Test that chat_history isolates by page_reference (the bug fix).

        This tests the specific bug where conversations with the same conversation_id
        but different page_reference values were getting their chat_history mixed.
        """
        from helpers.llm_eval import build_evaluation_run
        from helpers.llm_models import (
            Conversation,
            ConversationTurn,
            GroundingDocument,
            LLMModel,
            QuestionDataset,
        )

        # Create document with 2 pages
        grounding_doc = GroundingDocument(
            filename="multi-page.pdf",
            pages=["Page 0 content about topic A.", "Page 1 content about topic B."],
            page_count=2,
            file_size_bytes=2000,
        )

        # Create 2 conversations with SAME conversation_id but DIFFERENT page_reference
        # This simulates the scenario that caused the original bug
        conv_page_0 = Conversation(
            conversation_id="conv-001",  # Same ID
            page_reference=0,  # Different page
            turns=[
                ConversationTurn(turn_number=1, question="What is topic A?"),
                ConversationTurn(turn_number=2, question="More about A?"),
            ],
        )
        conv_page_1 = Conversation(
            conversation_id="conv-001",  # Same ID (this was causing the bug)
            page_reference=1,  # Different page
            turns=[
                ConversationTurn(turn_number=1, question="What is topic B?"),
                ConversationTurn(turn_number=2, question="More about B?"),
            ],
        )

        dataset = QuestionDataset(
            dataset_id="ds-test",
            source_document=grounding_doc,
            conversations=[conv_page_0, conv_page_1],
        )

        models = [LLMModel(name="GPT-4", endpoint="https://test.com", api_key="key")]

        # Simulate results - each page gets its own responses
        results = [
            # Page 0, Turn 1
            {
                "ok": True,
                "response": "Topic A is about page 0 content",
                "model_name": "GPT-4",
                "conversation_id": "conv-001",
                "turn_number": 1,
                "page_reference": 0,
                "latency_ms": 100,
                "api_latency_ms": 80,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cached_tokens": 0,
                "total_tokens": 150,
                "messages": [
                    {"role": "system", "content": "Context: Page 0 content"},
                    {"role": "user", "content": "What is topic A?"},
                    {"role": "assistant", "content": "Topic A is about page 0 content"},
                ],
            },
            # Page 0, Turn 2
            {
                "ok": True,
                "response": "More details about topic A from page 0",
                "model_name": "GPT-4",
                "conversation_id": "conv-001",
                "turn_number": 2,
                "page_reference": 0,
                "latency_ms": 110,
                "api_latency_ms": 85,
                "prompt_tokens": 150,
                "completion_tokens": 60,
                "cached_tokens": 0,
                "total_tokens": 210,
                "messages": [
                    {"role": "system", "content": "Context: Page 0 content"},
                    {"role": "user", "content": "What is topic A?"},
                    {"role": "assistant", "content": "Topic A is about page 0 content"},
                    {"role": "user", "content": "More about A?"},
                    {"role": "assistant", "content": "More details about topic A from page 0"},
                ],
            },
            # Page 1, Turn 1
            {
                "ok": True,
                "response": "Topic B is about page 1 content",
                "model_name": "GPT-4",
                "conversation_id": "conv-001",
                "turn_number": 1,
                "page_reference": 1,
                "latency_ms": 105,
                "api_latency_ms": 82,
                "prompt_tokens": 100,
                "completion_tokens": 52,
                "cached_tokens": 0,
                "total_tokens": 152,
                "messages": [
                    {"role": "system", "content": "Context: Page 1 content"},
                    {"role": "user", "content": "What is topic B?"},
                    {"role": "assistant", "content": "Topic B is about page 1 content"},
                ],
            },
            # Page 1, Turn 2
            {
                "ok": True,
                "response": "More details about topic B from page 1",
                "model_name": "GPT-4",
                "conversation_id": "conv-001",
                "turn_number": 2,
                "page_reference": 1,
                "latency_ms": 115,
                "api_latency_ms": 88,
                "prompt_tokens": 155,
                "completion_tokens": 62,
                "cached_tokens": 0,
                "total_tokens": 217,
                "messages": [
                    {"role": "system", "content": "Context: Page 1 content"},
                    {"role": "user", "content": "What is topic B?"},
                    {"role": "assistant", "content": "Topic B is about page 1 content"},
                    {"role": "user", "content": "More about B?"},
                    {"role": "assistant", "content": "More details about topic B from page 1"},
                ],
            },
        ]

        run = build_evaluation_run(
            run_id="run-test",
            dataset=dataset,
            models=models,
            results=results,
        )

        # Find page 0, turn 2 result
        page_0_turn_2 = next(r for r in run.results if r.page_reference == 0 and r.turn_number == 2)

        # chat_history should ONLY contain page 0's prior turns (not page 1's!)
        assert page_0_turn_2.chat_history is not None
        assert len(page_0_turn_2.chat_history) == 2
        assert page_0_turn_2.chat_history[0].content == "What is topic A?"
        assert "Topic A" in page_0_turn_2.chat_history[1].content
        assert "Topic B" not in page_0_turn_2.chat_history[1].content  # BUG FIX: no cross-page

        # Find page 1, turn 2 result
        page_1_turn_2 = next(r for r in run.results if r.page_reference == 1 and r.turn_number == 2)

        # chat_history should ONLY contain page 1's prior turns (not page 0's!)
        assert page_1_turn_2.chat_history is not None
        assert len(page_1_turn_2.chat_history) == 2
        assert page_1_turn_2.chat_history[0].content == "What is topic B?"
        assert "Topic B" in page_1_turn_2.chat_history[1].content
        assert "Topic A" not in page_1_turn_2.chat_history[1].content  # BUG FIX: no cross-page
