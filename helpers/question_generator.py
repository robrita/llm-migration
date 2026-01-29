"""
Question Generator Module

Generates multi-turn Q&A conversations using LLM from grounding documents.
Questions are generated per-page to simulate RAG chunking - each conversation
references a specific page (0-indexed) for evaluation grounding.
"""

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, AsyncOpenAI

from helpers.llm_models import (
    Conversation,
    ConversationTurn,
    GroundingDocument,
    LLMModel,
    QuestionDataset,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

DEFAULT_NUM_CONVERSATIONS = 5
DEFAULT_TURNS_PER_CONVERSATION = 3
MIN_PAGE_CHARS = 500  # Minimum characters for a page to be valid for generation
MAX_RETRIES = 3


# ============================================================================
# Page Filtering
# ============================================================================


def get_valid_page_indices(pages: list[str], min_chars: int = MIN_PAGE_CHARS) -> list[int]:
    """
    Get indices of pages with sufficient content for question generation.

    Args:
        pages: List of page content strings (0-indexed).
        min_chars: Minimum character count for a page to be valid.

    Returns:
        List of valid page indices (0-indexed).
    """
    return [i for i, page in enumerate(pages) if len(page.strip()) >= min_chars]


def distribute_conversations(
    num_conversations: int, valid_page_indices: list[int]
) -> dict[int, int]:
    """
    Distribute conversations round-robin across valid pages.

    Args:
        num_conversations: Total number of conversations to generate.
        valid_page_indices: List of valid page indices (0-indexed).

    Returns:
        Dict mapping page_index -> number of conversations to generate for that page.
    """
    if not valid_page_indices:
        return {}

    distribution: dict[int, int] = dict.fromkeys(valid_page_indices, 0)
    for i in range(num_conversations):
        page_idx = valid_page_indices[i % len(valid_page_indices)]
        distribution[page_idx] += 1

    return distribution


# ============================================================================
# Prompt Generation
# ============================================================================


def build_generation_prompt(
    page_content: str,
    num_conversations: int,
    turns_per_conversation: int,
) -> str:
    """
    Build the prompt for question generation from a single page.

    Args:
        page_content: The text content from a single document page.
        num_conversations: Number of conversations to generate for this page.
        turns_per_conversation: Number of turns per conversation.

    Returns:
        The formatted prompt string.
    """
    return f"""You are an expert at generating questions for evaluating RAG (Retrieval-Augmented Generation) systems.

Based on the following document page content, generate {num_conversations} multi-turn conversations with {turns_per_conversation} turns each.

Each conversation should:
1. Focus on CORE CONCEPTS, key ideas, definitions, processes, or factual information in the content
2. Start with a clear, answerable question based ONLY on the provided page content
3. Follow-up questions should build on previous context (like a real conversation)
4. All questions must be answerable using ONLY the information in this page
5. Ask questions DIRECTLY without preamble - never start with "According to the page", "Based on the document", "As stated in the text", or similar phrases

IMPORTANT - DO NOT generate questions about:
- Document structure (e.g., "How many sections are there?", "What is the title?")
- Formatting or layout (e.g., "Is this a list or paragraph?")
- Meta-information about the document itself
- Page numbers, headers, footers, or navigation elements

Instead, focus on questions that test understanding of:
- Main concepts and their definitions
- Relationships between ideas
- Processes, procedures, or methodologies described
- Facts, figures, and specific data points
- Causes, effects, and implications

Respond ONLY with valid JSON in this exact format:
{{
    "conversations": [
        {{
            "conversation_id": "conv-001",
            "turns": [
                {{
                    "turn_number": 1,
                    "question": "Question text here"
                }},
                {{
                    "turn_number": 2,
                    "question": "Follow-up question building on turn 1"
                }}
            ]
        }}
    ]
}}

PAGE CONTENT:
---
{page_content}
---

Generate the conversations now. Respond with ONLY the JSON object, no additional text."""


# ============================================================================
# Response Parsing
# ============================================================================


def parse_page_response(
    response_text: str,
    page_reference: int,
    start_index: int = 0,
) -> list[Conversation]:
    """
    Parse the LLM response for a single page into Conversation objects.

    Args:
        response_text: The raw JSON response from the LLM.
        page_reference: The 0-indexed page number these conversations reference.
        start_index: Starting index for sequential conversation IDs (0-based).

    Returns:
        List of Conversation objects with page_reference set.

    Raises:
        ValueError: If the response cannot be parsed.
    """
    text = response_text.strip()

    # Handle markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        raise ValueError(f"Invalid JSON response: {e}") from e

    conversations = []
    raw_conversations = data.get("conversations", [])

    for idx, conv_data in enumerate(raw_conversations):
        turns = []
        for turn_data in conv_data.get("turns", []):
            turn = ConversationTurn(
                turn_number=turn_data.get("turn_number", 1),
                question=turn_data.get("question", ""),
            )
            turns.append(turn)

        if turns:
            # Sequential conversation ID: conv-001, conv-002, etc.
            conv_id = f"conv-{start_index + idx + 1:03d}"
            conversation = Conversation(
                conversation_id=conv_id,
                page_reference=page_reference,
                turns=turns,
            )
            conversations.append(conversation)

    return conversations


# ============================================================================
# Main Generation Function
# ============================================================================


async def generate_questions(
    document: GroundingDocument,
    model: LLMModel,
    num_conversations: int = DEFAULT_NUM_CONVERSATIONS,
    turns_per_conversation: int = DEFAULT_TURNS_PER_CONVERSATION,
) -> dict[str, Any]:
    """
    Generate multi-turn Q&A conversations from a document using an LLM.

    Questions are generated per-page to simulate RAG chunking. Each conversation
    references a specific page (0-indexed) that will be used as grounding context
    during evaluation instead of the full document.

    Args:
        document: The grounding document with pages array.
        model: The LLM model configuration to use.
        num_conversations: Total number of conversations to generate (distributed across pages).
        turns_per_conversation: Number of turns per conversation.

    Returns:
        Dict with:
        - ok (bool): Whether generation succeeded
        - dataset (QuestionDataset): The generated dataset (if ok)
        - error (str): Error message (if not ok)
        - warning (str): Warning message (if applicable)
        - processing_time (float): Time taken in seconds
    """
    start_time = time.time()

    # Validate document has pages
    if not document.pages:
        return {
            "ok": False,
            "error": "Document has no pages. Cannot generate questions.",
            "processing_time": time.time() - start_time,
        }

    # Find valid pages (≥500 chars)
    valid_indices = get_valid_page_indices(document.pages)
    if not valid_indices:
        return {
            "ok": False,
            "error": (
                f"No pages have sufficient content (minimum {MIN_PAGE_CHARS} characters). "
                "Please use a different PDF with more text content."
            ),
            "processing_time": time.time() - start_time,
        }

    logger.info(f"Found {len(valid_indices)} valid pages out of {len(document.pages)} total pages")

    # Distribute conversations across valid pages
    distribution = distribute_conversations(num_conversations, valid_indices)

    try:
        # Create client based on model type
        if model.endpoint:
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            client = AsyncAzureOpenAI(
                azure_endpoint=model.endpoint,
                azure_ad_token_provider=token_provider,
                api_version=model.api_version or "2024-12-01-preview",
            )
            model_name = model.deployment_name or "gpt-4o"
        else:
            client = AsyncOpenAI(
                api_key=model.api_key.get_secret_value(),
            )
            model_name = model.deployment_name or "gpt-4o"

        all_conversations: list[Conversation] = []

        # Generate conversations for each page with assigned count
        for page_idx, conv_count in distribution.items():
            if conv_count == 0:
                continue

            page_content = document.pages[page_idx]
            prompt = build_generation_prompt(
                page_content,
                num_conversations=conv_count,
                turns_per_conversation=turns_per_conversation,
            )

            logger.info(
                f"Generating {conv_count} conversations for page {page_idx} "
                f"using {model.name} ({model_name})"
            )

            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates questions in JSON format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content
            logger.debug(f"LLM response for page {page_idx}: {response_text[:300]}...")

            # Parse response with sequential conversation IDs
            page_conversations = parse_page_response(
                response_text, page_reference=page_idx, start_index=len(all_conversations)
            )
            all_conversations.extend(page_conversations)

        # Build dataset
        dataset = QuestionDataset(
            dataset_id=f"ds-{uuid.uuid4().hex[:8]}",
            source_document=document,
            conversations=all_conversations,
            created_at=datetime.now(UTC).isoformat(),
        )

        total_turns = sum(len(c.turns) for c in all_conversations)
        logger.info(
            f"Successfully generated {len(all_conversations)} conversations "
            f"with {total_turns} total turns across {len(valid_indices)} pages"
        )

        result: dict[str, Any] = {
            "ok": True,
            "dataset": dataset,
            "processing_time": time.time() - start_time,
        }

        # Add warning if we had to skip some pages
        skipped_pages = len(document.pages) - len(valid_indices)
        if skipped_pages > 0:
            result["warning"] = (
                f"Skipped {skipped_pages} pages with less than {MIN_PAGE_CHARS} characters"
            )

        return result

    except ValueError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        return {
            "ok": False,
            "error": str(e),
            "processing_time": time.time() - start_time,
        }
    except Exception as e:
        logger.error(f"Error during question generation: {e}")
        return {
            "ok": False,
            "error": str(e),
            "processing_time": time.time() - start_time,
        }
