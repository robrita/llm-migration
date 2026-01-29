"""
LLM Evaluation Module

Provides parallel evaluation of LLM models with:
- Single model evaluation with timeout
- Retry logic for transient errors (429, 503)
- Parallel execution using asyncio
- Conversation history management
"""

import asyncio
import logging
import time
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import APIStatusError, AsyncAzureOpenAI, AsyncOpenAI, RateLimitError

from helpers.llm_models import (
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_SYSTEM_PROMPT,
    ChatHistoryItem,
    Conversation,
    EvaluationResult,
    EvaluationRun,
    LLMModel,
    QuestionDataset,
    RAGMetrics,
    RunStatus,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_HISTORY = 10
RETRY_BACKOFF_BASE = 1.0  # Base delay in seconds


# ============================================================================
# Client Factory - Create once per model, reuse across evaluations
# ============================================================================


def create_client(model: LLMModel) -> tuple[AsyncAzureOpenAI | AsyncOpenAI, str]:
    """
    Create an async OpenAI client for a model.

    Args:
        model: The LLM model configuration.

    Returns:
        Tuple of (client, deployment_name)
    """
    if model.endpoint:
        # Use Entra ID authentication for Azure OpenAI
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        client = AsyncAzureOpenAI(
            azure_endpoint=model.endpoint,
            azure_ad_token_provider=token_provider,
            api_version=model.api_version or "2024-12-01-preview",
        )
        deployment_name = model.deployment_name or "gpt-4o"
    else:
        client = AsyncOpenAI(
            api_key=model.api_key.get_secret_value(),
        )
        deployment_name = model.deployment_name or "gpt-4o"

    logger.debug(f"Created client for model: {model.name}")
    return client, deployment_name


# ============================================================================
# Conversation History Builder
# ============================================================================


def build_conversation_history(
    conversation: Conversation,
    responses: list[str],
    current_turn_index: int,
    max_history: int = DEFAULT_MAX_HISTORY,
    grounding_content: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """
    Build the conversation history for the API call.

    Args:
        conversation: The conversation with turns.
        responses: List of previous responses from the model.
        current_turn_index: Index of the current turn (0-based).
        max_history: Maximum number of user/assistant pairs to retain.
        grounding_content: Optional grounding document content.
        system_prompt: Base system prompt for the assistant.

    Returns:
        List of message dicts for the API call.
    """
    messages = []

    # System message with grounding context
    system_content = system_prompt
    if grounding_content:
        system_content += f"\n\nUse the following context to answer questions:\n{grounding_content}"
    messages.append({"role": "system", "content": system_content})

    # Build history from previous turns
    start_index = max(0, current_turn_index - max_history)

    for i in range(start_index, current_turn_index):
        turn = conversation.turns[i]
        messages.append({"role": "user", "content": turn.question})

        if i < len(responses):
            messages.append({"role": "assistant", "content": responses[i]})

    # Add current turn question
    if current_turn_index < len(conversation.turns):
        messages.append(
            {"role": "user", "content": conversation.turns[current_turn_index].question}
        )

    return messages


# ============================================================================
# Single Evaluation
# ============================================================================


async def evaluate_single(
    client: AsyncAzureOpenAI | AsyncOpenAI,
    deployment_name: str,
    model: LLMModel,
    conversation: Conversation,
    page_content: str,
    turn_index: int,
    previous_responses: list[str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_history: int = DEFAULT_MAX_HISTORY,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    """
    Evaluate a single turn of a conversation with one model.

    Args:
        client: Pre-created async OpenAI client (reused across calls).
        deployment_name: Model deployment name.
        model: The LLM model configuration.
        conversation: The conversation to evaluate.
        page_content: The grounding page content for this conversation.
        turn_index: Index of the turn to evaluate (0-based).
        previous_responses: List of previous responses for context.
        timeout_seconds: Timeout for the API call.
        max_retries: Maximum number of retries for transient errors.
        max_history: Maximum conversation history pairs to retain.
        system_prompt: System prompt for the assistant.

    Returns:
        Dict with:
        - ok (bool): Whether evaluation succeeded
        - response (str): Model response (if ok)
        - latency_ms (float): Total end-to-end latency in milliseconds
        - api_latency_ms (float): Pure API call latency in milliseconds
        - model_name (str): Name of the model
        - conversation_id (str): ID of the conversation
        - turn_number (int): Turn number (1-based)
        - page_reference (int): 0-indexed page used for grounding
        - error (str): Error message (if not ok)
    """
    start_time = time.time()
    previous_responses = previous_responses or []

    # Build messages with history using page content for grounding
    messages = build_conversation_history(
        conversation=conversation,
        responses=previous_responses,
        current_turn_index=turn_index,
        max_history=max_history,
        grounding_content=page_content,
        system_prompt=system_prompt,
    )

    # Retry loop
    last_error = None
    for attempt in range(max_retries):
        try:
            # Measure pure API latency - starts here, right before the API call
            api_start_time = time.time()

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=deployment_name,
                    messages=messages,
                    temperature=0.7,
                ),
                timeout=timeout_seconds,
            )

            # Pure API latency ends here
            api_latency_ms = (time.time() - api_start_time) * 1000
            total_latency_ms = (time.time() - start_time) * 1000

            response_text = response.choices[0].message.content

            # Extract token usage from response
            prompt_tokens = 0
            completion_tokens = 0
            cached_tokens = 0
            total_tokens = 0

            if hasattr(response, "usage") and response.usage:
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                total_tokens = getattr(response.usage, "total_tokens", 0) or 0

                # Check for cached tokens (Azure OpenAI prompt caching)
                if hasattr(response.usage, "prompt_tokens_details"):
                    details = response.usage.prompt_tokens_details
                    if details and hasattr(details, "cached_tokens"):
                        cached_tokens = getattr(details, "cached_tokens", 0) or 0

            logger.debug(
                f"[{model.name}] Turn {turn_index + 1}: {response_text[:100]}... "
                f"(API: {api_latency_ms:.0f}ms, Total: {total_latency_ms:.0f}ms, "
                f"Tokens: {prompt_tokens}+{completion_tokens}={total_tokens})"
            )

            # Append assistant response to messages for complete API payload
            complete_messages = messages + [{"role": "assistant", "content": response_text}]

            return {
                "ok": True,
                "response": response_text,
                "messages": complete_messages,
                "latency_ms": total_latency_ms,
                "api_latency_ms": api_latency_ms,
                "model_name": model.name,
                "conversation_id": conversation.conversation_id,
                "turn_number": turn_index + 1,
                "page_reference": conversation.page_reference,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "total_tokens": total_tokens,
            }

        except TimeoutError:
            logger.warning(f"[{model.name}] Timeout on turn {turn_index + 1}")
            return {
                "ok": False,
                "error": f"Timeout after {timeout_seconds}s",
                "model_name": model.name,
                "conversation_id": conversation.conversation_id,
                "turn_number": turn_index + 1,
                "page_reference": conversation.page_reference,
                "latency_ms": (time.time() - start_time) * 1000,
                "api_latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "total_tokens": 0,
            }

        except RateLimitError as e:
            last_error = e
            delay = RETRY_BACKOFF_BASE * (2**attempt)
            logger.warning(
                f"[{model.name}] Rate limited (429), retry {attempt + 1}/{max_retries} in {delay}s"
            )
            await asyncio.sleep(delay)

        except APIStatusError as e:
            if e.status_code in (429, 503):
                last_error = e
                delay = RETRY_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    f"[{model.name}] Error {e.status_code}, "
                    f"retry {attempt + 1}/{max_retries} in {delay}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"[{model.name}] API error: {e}")
                return {
                    "ok": False,
                    "error": str(e),
                    "model_name": model.name,
                    "conversation_id": conversation.conversation_id,
                    "turn_number": turn_index + 1,
                    "page_reference": conversation.page_reference,
                    "latency_ms": (time.time() - start_time) * 1000,
                    "api_latency_ms": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "total_tokens": 0,
                }

        except Exception as e:
            logger.error(f"[{model.name}] Unexpected error: {e}")
            return {
                "ok": False,
                "error": str(e),
                "model_name": model.name,
                "conversation_id": conversation.conversation_id,
                "turn_number": turn_index + 1,
                "page_reference": conversation.page_reference,
                "latency_ms": (time.time() - start_time) * 1000,
                "api_latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "total_tokens": 0,
            }

    # All retries exhausted
    return {
        "ok": False,
        "error": f"Max retries ({max_retries}) exhausted: {last_error}",
        "model_name": model.name,
        "conversation_id": conversation.conversation_id,
        "turn_number": turn_index + 1,
        "page_reference": conversation.page_reference,
        "latency_ms": (time.time() - start_time) * 1000,
        "api_latency_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
    }


# ============================================================================
# Parallel Evaluation
# ============================================================================


async def run_conversation_turns(
    client: AsyncAzureOpenAI | AsyncOpenAI,
    deployment_name: str,
    model: LLMModel,
    conversation: Conversation,
    page_content: str,
    timeout_seconds: float,
    max_retries: int,
    max_history: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """
    Run all turns of a conversation sequentially to preserve multi-turn context.

    Args:
        client: Pre-created async OpenAI client.
        deployment_name: Model deployment name.
        model: The LLM model configuration.
        conversation: The conversation to evaluate.
        page_content: The grounding page content for this conversation.
        timeout_seconds: Timeout per API call.
        max_retries: Max retries for transient errors.
        max_history: Max conversation history pairs.
        system_prompt: System prompt for the assistant.

    Returns:
        List of evaluation results for each turn.
    """
    results = []
    previous_responses: list[str] = []

    for turn_idx in range(len(conversation.turns)):
        result = await evaluate_single(
            client=client,
            deployment_name=deployment_name,
            model=model,
            conversation=conversation,
            page_content=page_content,
            turn_index=turn_idx,
            previous_responses=previous_responses,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_history=max_history,
            system_prompt=system_prompt,
        )
        results.append(result)

        # Accumulate successful responses for next turn's context
        if result.get("ok") and result.get("response"):
            previous_responses.append(result["response"])
        else:
            # On failure, add placeholder to maintain indexing
            previous_responses.append("")

    return results


async def run_parallel_evaluations(
    models: list[LLMModel],
    dataset: QuestionDataset,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_history: int = DEFAULT_MAX_HISTORY,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    progress_callback: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Run evaluations across all models and conversations with proper parallel/sequential execution.

    Execution model:
    - Models: parallel (independent)
    - Conversations: parallel (independent)
    - Turns within a conversation: sequential (multi-turn context dependency)

    Args:
        models: List of LLM models to evaluate.
        dataset: The question dataset with conversations.
        max_concurrent: Maximum concurrent API calls.
        timeout_seconds: Timeout per API call.
        max_retries: Max retries for transient errors.
        max_history: Max conversation history pairs.
        system_prompt: System prompt for evaluation models.
        progress_callback: Optional callback(completed, total) for progress.

    Returns:
        List of evaluation results for all model/turn combinations.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    total_tasks = sum(len(c.turns) for c in dataset.conversations) * len(models)
    completed = [0]  # Use list for mutable counter in nested function

    # Create clients once per model (reuse across all evaluations)
    clients: dict[str, tuple[AsyncAzureOpenAI | AsyncOpenAI, str]] = {}
    for model in models:
        try:
            clients[model.name] = create_client(model)
            logger.info(f"Created client for model: {model.name}")
        except Exception as e:
            logger.error(f"Failed to create client for {model.name}: {e}")
            # Will handle missing client below

    try:

        async def run_conversation_with_semaphore(
            model: LLMModel,
            conversation: Conversation,
        ) -> list[dict[str, Any]]:
            """Run a single conversation with semaphore control."""
            if model.name not in clients:
                # Client creation failed - return error for all turns
                return [
                    {
                        "ok": False,
                        "error": "Failed to create client",
                        "model_name": model.name,
                        "conversation_id": conversation.conversation_id,
                        "turn_number": i + 1,
                        "page_reference": conversation.page_reference,
                        "latency_ms": 0.0,
                        "api_latency_ms": 0.0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cached_tokens": 0,
                        "total_tokens": 0,
                    }
                    for i in range(len(conversation.turns))
                ]

            client, deployment_name = clients[model.name]

            # Look up page content for this conversation
            page_idx = conversation.page_reference
            if page_idx < 0 or page_idx >= len(dataset.source_document.pages):
                return [
                    {
                        "ok": False,
                        "error": f"Invalid page_reference {page_idx}",
                        "model_name": model.name,
                        "conversation_id": conversation.conversation_id,
                        "turn_number": i + 1,
                        "page_reference": page_idx,
                        "latency_ms": 0.0,
                        "api_latency_ms": 0.0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cached_tokens": 0,
                        "total_tokens": 0,
                    }
                    for i in range(len(conversation.turns))
                ]
            page_content = dataset.source_document.pages[page_idx]

            # Run turns sequentially within semaphore
            async with semaphore:
                results = await run_conversation_turns(
                    client=client,
                    deployment_name=deployment_name,
                    model=model,
                    conversation=conversation,
                    page_content=page_content,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    max_history=max_history,
                    system_prompt=system_prompt,
                )

                # Update progress after each conversation completes
                completed[0] += len(results)
                if progress_callback:
                    progress_callback(completed[0], total_tasks)

                return results

        # Create tasks for each model/conversation combination (parallel)
        # Turns within each conversation run sequentially inside run_conversation_with_semaphore
        tasks = []
        for model in models:
            for conversation in dataset.conversations:
                tasks.append(run_conversation_with_semaphore(model, conversation))

        logger.info(
            f"Starting evaluations: {len(models)} models × {len(dataset.conversations)} conversations "
            f"= {len(tasks)} parallel tasks, {total_tasks} total turns"
        )

        # Run all model/conversation combinations in parallel
        conversation_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        all_results = []
        for result in conversation_results:
            if isinstance(result, Exception):
                logger.error(f"Conversation task failed with exception: {result}")
            elif isinstance(result, list):
                all_results.extend(result)

        logger.info(f"Completed {len(all_results)}/{total_tasks} evaluations")

    finally:
        # Properly close all async clients to avoid "Event loop is closed" on Windows
        for client, _ in clients.values():
            await client.close()

    return all_results


# ============================================================================
# Evaluation Run Builder
# ============================================================================


def build_evaluation_run(
    run_id: str,
    dataset: QuestionDataset,
    models: list[LLMModel],
    results: list[dict[str, Any]],
) -> EvaluationRun:
    """
    Build an EvaluationRun from raw results.

    Args:
        run_id: Unique run identifier.
        dataset: The dataset used for evaluation.
        models: List of models evaluated.
        results: Raw results from run_parallel_evaluations.

    Returns:
        EvaluationRun with structured results.
    """
    evaluation_results = []

    # Group results by (model_name, conversation_id, page_reference) to build chat history
    # Using page_reference ensures no cross-page contamination in multi-turn context
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        key = (
            result.get("model_name", ""),
            result.get("conversation_id", ""),
            result.get("page_reference", 0),
        )
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    # Sort each group by turn_number
    for key in grouped:
        grouped[key].sort(key=lambda r: r.get("turn_number", 1))

    for result in results:
        if not isinstance(result, dict):
            continue

        # Find the corresponding conversation matching both conversation_id AND page_reference
        conversation = None
        result_conv_id = result.get("conversation_id")
        result_page_ref = result.get("page_reference")
        for conv in dataset.conversations:
            if conv.conversation_id == result_conv_id and conv.page_reference == result_page_ref:
                conversation = conv
                break

        # Build chat history from prior turns (for turn > 1)
        chat_history = None
        turn_num = result.get("turn_number", 1)
        if turn_num > 1 and conversation:
            key = (
                result.get("model_name", ""),
                result.get("conversation_id", ""),
                result.get("page_reference", 0),
            )
            prior_results = grouped.get(key, [])
            chat_history = []
            for prior in prior_results:
                prior_turn = prior.get("turn_number", 1)
                if prior_turn >= turn_num:
                    break
                # Add user question from dataset
                if 0 < prior_turn <= len(conversation.turns):
                    chat_history.append(
                        ChatHistoryItem(
                            role="user",
                            content=conversation.turns[prior_turn - 1].question,
                        )
                    )
                # Add assistant response from evaluation
                if prior.get("ok") and prior.get("response"):
                    chat_history.append(
                        ChatHistoryItem(role="assistant", content=prior["response"])
                    )

        eval_result = EvaluationResult(
            model_name=result.get("model_name", "Unknown"),
            conversation_id=result.get("conversation_id", "Unknown"),
            turn_number=result.get("turn_number", 1),
            question=_find_question(dataset, result),
            page_reference=result.get("page_reference"),
            response=result.get("response", ""),
            latency_ms=result.get("latency_ms", 0.0),
            api_latency_ms=result.get("api_latency_ms", 0.0),
            success=result.get("ok", False),
            error=result.get("error"),
            metrics=RAGMetrics(),  # Placeholder - calculated later
            chat_history=chat_history,
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            cached_tokens=result.get("cached_tokens", 0),
            total_tokens=result.get("total_tokens", 0),
            messages=result.get("messages"),
        )
        evaluation_results.append(eval_result)

    # Determine overall status
    success_count = sum(1 for r in results if r.get("ok", False))
    if success_count == len(results):
        status = RunStatus.COMPLETED
    elif success_count > 0:
        status = RunStatus.PARTIAL
    else:
        status = RunStatus.FAILED

    return EvaluationRun(
        run_id=run_id,
        dataset_id=dataset.dataset_id,
        model_names=[m.name for m in models],
        results=evaluation_results,
        status=status,
    )


def _find_question(dataset: QuestionDataset, result: dict) -> str:
    """Find the question for a result matching conversation_id and page_reference."""
    result_conv_id = result.get("conversation_id")
    result_page_ref = result.get("page_reference")
    for conv in dataset.conversations:
        if conv.conversation_id == result_conv_id and conv.page_reference == result_page_ref:
            turn_num = result.get("turn_number", 1)
            if 0 < turn_num <= len(conv.turns):
                return conv.turns[turn_num - 1].question
    return ""
