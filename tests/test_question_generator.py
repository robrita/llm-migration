"""
Unit tests for helpers/question_generator.py

Tests cover:
- Question generation from document content
- Multi-turn conversation generation
- JSON parsing of LLM responses
- Error handling for LLM failures
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
def sample_document_content() -> str:
    """Sample document content for testing."""
    return """
    Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without being explicitly programmed.

    There are three main types of machine learning:
    1. Supervised Learning - Uses labeled data to train models
    2. Unsupervised Learning - Finds patterns in unlabeled data
    3. Reinforcement Learning - Learns through trial and error

    Common applications include image recognition, natural language
    processing, and recommendation systems.
    """


@pytest.fixture
def mock_llm_response_valid():
    """Valid JSON response from LLM for question generation."""
    return """
    {
        "conversations": [
            {
                "conversation_id": "conv-001",
                "turns": [
                    {
                        "turn_number": 1,
                        "question": "What is machine learning?"
                    },
                    {
                        "turn_number": 2,
                        "question": "What are the three main types of machine learning?"
                    }
                ]
            },
            {
                "conversation_id": "conv-002",
                "turns": [
                    {
                        "turn_number": 1,
                        "question": "What are common applications of machine learning?"
                    }
                ]
            }
        ]
    }
    """


@pytest.fixture
def mock_llm_response_invalid():
    """Invalid JSON response from LLM."""
    return "This is not valid JSON and should cause parsing to fail."


# ============================================================================
# Test Cases
# ============================================================================


@pytest.mark.unit
class TestGenerateQuestions:
    """Tests for generate_questions function."""

    @pytest.mark.asyncio
    async def test_generate_questions_success(
        self, sample_document_content: str, mock_llm_response_valid: str
    ):
        """Test successful question generation."""
        from helpers.llm_models import GroundingDocument, LLMModel
        from helpers.question_generator import generate_questions

        document = GroundingDocument(
            filename="test.pdf",
            pages=[sample_document_content],
            page_count=1,
            file_size_bytes=1000,
        )
        model = LLMModel(
            name="Test-Generator",
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=mock_llm_response_valid))]

        with (
            patch("helpers.question_generator.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.question_generator.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await generate_questions(document, model, num_conversations=2)

            assert result["ok"] is True
            assert "dataset" in result
            dataset = result["dataset"]
            assert dataset.dataset_id is not None
            assert len(dataset.conversations) == 2
            assert dataset.conversations[0].turns[0].question == "What is machine learning?"
            assert dataset.conversations[0].page_reference == 0

    @pytest.mark.asyncio
    async def test_generate_questions_invalid_json(
        self, sample_document_content: str, mock_llm_response_invalid: str
    ):
        """Test handling of invalid JSON response from LLM."""
        from helpers.llm_models import GroundingDocument, LLMModel
        from helpers.question_generator import generate_questions

        document = GroundingDocument(
            filename="test.pdf",
            pages=[sample_document_content],
            page_count=1,
            file_size_bytes=1000,
        )
        model = LLMModel(
            name="Test-Generator",
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=mock_llm_response_invalid))]

        with (
            patch("helpers.question_generator.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.question_generator.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await generate_questions(document, model, num_conversations=2)

            assert result["ok"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_generate_questions_api_error(self, sample_document_content: str):
        """Test handling of API errors during question generation."""
        from helpers.llm_models import GroundingDocument, LLMModel
        from helpers.question_generator import generate_questions

        document = GroundingDocument(
            filename="test.pdf",
            pages=[sample_document_content],
            page_count=1,
            file_size_bytes=1000,
        )
        model = LLMModel(
            name="Test-Generator",
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

        with (
            patch("helpers.question_generator.get_bearer_token_provider") as mock_token_provider,
            patch("helpers.question_generator.AsyncAzureOpenAI") as mock_client_class,
        ):
            mock_token_provider.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client.chat = MagicMock()
            mock_client.chat.completions = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
            mock_client_class.return_value = mock_client

            result = await generate_questions(document, model, num_conversations=2)

            assert result["ok"] is False
            assert "error" in result
            assert "API Error" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_questions_empty_document(self):
        """Test handling of empty document content."""
        from helpers.llm_models import GroundingDocument, LLMModel
        from helpers.question_generator import generate_questions

        document = GroundingDocument(
            filename="empty.pdf",
            pages=["   "],  # Whitespace only - insufficient content
            page_count=1,
            file_size_bytes=100,
        )
        model = LLMModel(
            name="Test-Generator",
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
        )

        result = await generate_questions(document, model, num_conversations=2)

        assert result["ok"] is False
        assert (
            "no pages" in result["error"].lower() or "sufficient content" in result["error"].lower()
        )


@pytest.mark.unit
class TestQuestionGeneratorPrompt:
    """Tests for prompt generation."""

    def test_prompt_includes_document_content(self, sample_document_content: str):
        """Test that prompt includes the document content."""
        from helpers.question_generator import build_generation_prompt

        prompt = build_generation_prompt(
            sample_document_content, num_conversations=3, turns_per_conversation=3
        )

        assert "machine learning" in prompt.lower()
        assert "supervised" in prompt.lower()

    def test_prompt_requests_correct_format(self, sample_document_content: str):
        """Test that prompt requests JSON format."""
        from helpers.question_generator import build_generation_prompt

        prompt = build_generation_prompt(
            sample_document_content, num_conversations=3, turns_per_conversation=3
        )

        assert "json" in prompt.lower()
        assert "conversation" in prompt.lower()


@pytest.mark.unit
class TestParsePageResponseConversationId:
    """Tests for sequential conversation ID generation in parse_page_response.

    Conversation IDs are sequential (conv-001, conv-002, etc.) across all pages.
    The page isolation is handled by the page_reference field, not the ID.
    """

    def test_sequential_conv_id_ignores_llm_placeholder(self):
        """Test that LLM's placeholder conversation_id is ignored."""
        from helpers.question_generator import parse_page_response

        # LLM returns the same "conv-001" for all conversations
        llm_response = """
        {
            "conversations": [
                {
                    "conversation_id": "conv-001",
                    "turns": [{"turn_number": 1, "question": "Q1?"}]
                },
                {
                    "conversation_id": "conv-001",
                    "turns": [{"turn_number": 1, "question": "Q2?"}]
                }
            ]
        }
        """
        result = parse_page_response(llm_response, page_reference=0, start_index=0)

        # Should generate sequential IDs
        assert len(result) == 2
        assert result[0].conversation_id == "conv-001"
        assert result[1].conversation_id == "conv-002"

    def test_sequential_conv_ids_with_start_index(self):
        """Test that start_index continues sequential numbering."""
        from helpers.question_generator import parse_page_response

        llm_response = """
        {
            "conversations": [
                {
                    "conversation_id": "ignored",
                    "turns": [{"turn_number": 1, "question": "Q1?"}]
                }
            ]
        }
        """

        # Simulate: page 0 generated 2 conversations, now parsing page 1
        result = parse_page_response(llm_response, page_reference=1, start_index=2)

        assert result[0].conversation_id == "conv-003"  # Continues from 2
        assert result[0].page_reference == 1

    def test_conv_id_format_sequential(self):
        """Test conversation ID format: conv-{index:03d}."""
        from helpers.question_generator import parse_page_response

        llm_response = """
        {
            "conversations": [
                {"conversation_id": "ignored", "turns": [{"turn_number": 1, "question": "Q1?"}]},
                {"conversation_id": "ignored", "turns": [{"turn_number": 1, "question": "Q2?"}]},
                {"conversation_id": "ignored", "turns": [{"turn_number": 1, "question": "Q3?"}]}
            ]
        }
        """
        result = parse_page_response(llm_response, page_reference=5, start_index=0)

        assert len(result) == 3
        assert result[0].conversation_id == "conv-001"
        assert result[1].conversation_id == "conv-002"
        assert result[2].conversation_id == "conv-003"
        # All reference page 5
        assert all(c.page_reference == 5 for c in result)

    def test_conv_id_no_collision_with_incrementing_start_index(self):
        """Test that conversation IDs are unique when start_index increments."""
        from helpers.question_generator import parse_page_response

        llm_response = """{"conversations": [{"conversation_id": "x", "turns": [{"turn_number": 1, "question": "Q?"}]}]}"""

        all_conv_ids = []
        current_index = 0
        for page_idx in range(10):
            result = parse_page_response(
                llm_response, page_reference=page_idx, start_index=current_index
            )
            all_conv_ids.append(result[0].conversation_id)
            current_index += len(result)

        # Should have 10 unique sequential IDs
        assert len(set(all_conv_ids)) == 10
        assert all_conv_ids == [f"conv-{i:03d}" for i in range(1, 11)]
