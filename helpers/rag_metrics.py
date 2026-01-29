"""
RAG Metrics Module

Provides LLM-as-judge evaluation for RAG quality metrics:
- Groundedness: Is the response grounded in the source document?
- Relevance: Does the response answer the question?
- Coherence: Is the response logically structured?
- Fluency: Is the response grammatically correct and readable?
"""

import asyncio
import json
import logging
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, AsyncOpenAI

from helpers.llm_models import (
    DEFAULT_MAX_CONCURRENT,
    ChatHistoryItem,
    EvaluationResult,
    LLMModel,
    RAGMetrics,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Prompt Templates
# ============================================================================

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.

Evaluate the following response based on these criteria:

1. **Groundedness** (1-5): Is the response factually grounded in the provided reference content?
   - 1: Completely ungrounded, contradicts or ignores the source
   - 5: Fully grounded, all claims are supported by the source

2. **Relevance** (1-5): Does the response directly answer the question?
   - 1: Completely off-topic
   - 5: Directly and completely answers the question

3. **Coherence** (1-5): Is the response logically structured and easy to follow?
   - 1: Incoherent, disorganized
   - 5: Clear, well-structured, logical flow

4. **Fluency** (1-5): Is the response grammatically correct and readable?
   - 1: Major grammar issues, hard to read
   - 5: Perfect grammar, natural reading

---
{conversation_history_section}
**Question**: {question}

**Reference Content (RAG Chunk)**: {reference_content}

**Response to evaluate**: {response}

---

Respond with ONLY a JSON object in this exact format:
{{
    "groundedness": <1-5>,
    "relevance": <1-5>,
    "coherence": <1-5>,
    "fluency": <1-5>
}}
"""


# ============================================================================
# Helper Functions
# ============================================================================


def format_conversation_history(chat_history: list[ChatHistoryItem] | None) -> str:
    """
    Format conversation history for the judge prompt.

    Args:
        chat_history: List of prior conversation turns.

    Returns:
        Formatted string for insertion into the judge prompt.
    """
    if not chat_history:
        return ""

    lines = ["**Prior Conversation**:"]
    for item in chat_history:
        role_label = "User" if item.role == "user" else "Assistant"
        # Truncate long messages to avoid token bloat
        content = item.content[:500] + "..." if len(item.content) > 500 else item.content
        lines.append(f"- {role_label}: {content}")
    lines.append("")  # Blank line before current question

    return "\n".join(lines) + "\n"


# ============================================================================
# Metrics Calculation
# ============================================================================


def create_judge_client(
    judge_model: LLMModel,
) -> tuple[AsyncAzureOpenAI | AsyncOpenAI, str]:
    """
    Create an async OpenAI client for the judge model.

    Args:
        judge_model: The LLM model configuration for the judge.

    Returns:
        Tuple of (client, model_name)
    """
    if judge_model.endpoint:
        # Azure OpenAI with Entra ID authentication
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        client = AsyncAzureOpenAI(
            azure_endpoint=judge_model.endpoint,
            azure_ad_token_provider=token_provider,
            api_version=judge_model.api_version or "2024-12-01-preview",
        )
        model_name = judge_model.deployment_name or "gpt-4o"
    else:
        client = AsyncOpenAI(
            api_key=judge_model.api_key.get_secret_value(),
        )
        model_name = judge_model.deployment_name or "gpt-4o"

    logger.debug(f"Created judge client for model: {judge_model.name}")
    return client, model_name


async def calculate_metrics(
    evaluation_result: EvaluationResult,
    judge_model: LLMModel,
    reference_content: str | None = None,
    client: AsyncAzureOpenAI | AsyncOpenAI | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Calculate RAG metrics for a single evaluation result using LLM-as-judge.

    Args:
        evaluation_result: The evaluation result to score.
        judge_model: The LLM model to use as judge.
        reference_content: The page content used as RAG grounding context.
        client: Optional pre-created async client (reused for efficiency).
        model_name: Optional model/deployment name (required if client provided).

    Returns:
        Dict with:
        - ok (bool): Whether calculation succeeded
        - metrics (RAGMetrics): The calculated metrics (if ok)
        - error (str): Error message (if not ok)
    """
    # Skip failed evaluations
    if not evaluation_result.success or not evaluation_result.response:
        return {
            "ok": False,
            "error": "Cannot calculate metrics for failed evaluation",
        }

    # Format conversation history for multi-turn context
    conversation_history_section = format_conversation_history(evaluation_result.chat_history)

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        conversation_history_section=conversation_history_section,
        question=evaluation_result.question,
        reference_content=reference_content or "No reference content provided",
        response=evaluation_result.response,
    )

    try:
        # Use provided client or create a new one
        if client is not None and model_name is not None:
            _client = client
            _model_name = model_name
        else:
            _client, _model_name = create_judge_client(judge_model)

        response = await _client.chat.completions.create(
            model=_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator. Respond only with JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Low temperature for consistent scoring
            response_format={"type": "json_object"},
        )

        response_text = response.choices[0].message.content

        # Parse JSON response
        scores = json.loads(response_text)

        metrics = RAGMetrics(
            groundedness=scores.get("groundedness"),
            relevance=scores.get("relevance"),
            coherence=scores.get("coherence"),
            fluency=scores.get("fluency"),
        )

        logger.debug(
            f"Calculated metrics for {evaluation_result.model_name}: "
            f"G={metrics.groundedness} R={metrics.relevance} "
            f"C={metrics.coherence} F={metrics.fluency}"
        )

        return {"ok": True, "metrics": metrics}

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse judge response: {e}")
        return {"ok": False, "error": f"Invalid JSON response from judge: {e}"}

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {"ok": False, "error": str(e)}


# ============================================================================
# Aggregate Metrics
# ============================================================================


def aggregate_run_metrics(results: list[EvaluationResult]) -> dict[str, dict[str, Any]]:
    """
    Aggregate metrics across multiple evaluation results by model.

    Args:
        results: List of evaluation results.

    Returns:
        Dict mapping model names to aggregated metrics:
        {
            "Model-A": {
                "count": 5,
                "failed": 1,
                "avg_latency_ms": 150.0,
                "avg_groundedness": 4.2,
                "avg_relevance": 4.5,
                "avg_coherence": 4.0,
                "avg_fluency": 4.8,
            },
            ...
        }
    """
    model_stats: dict[str, dict[str, Any]] = {}

    for result in results:
        model_name = result.model_name

        if model_name not in model_stats:
            model_stats[model_name] = {
                "count": 0,
                "failed": 0,
                "total_latency_ms": 0.0,
                "total_groundedness": 0.0,
                "total_relevance": 0.0,
                "total_coherence": 0.0,
                "total_fluency": 0.0,
                "scored_count": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cached_tokens": 0,
                "total_tokens": 0,
            }

        stats = model_stats[model_name]

        if result.success:
            stats["count"] += 1
            stats["total_latency_ms"] += result.latency_ms

            # Accumulate token counts
            stats["total_prompt_tokens"] += result.prompt_tokens
            stats["total_completion_tokens"] += result.completion_tokens
            stats["total_cached_tokens"] += result.cached_tokens
            stats["total_tokens"] += result.total_tokens

            # Only count scored results for average metrics
            if result.metrics and result.metrics.groundedness is not None:
                stats["scored_count"] += 1
                stats["total_groundedness"] += result.metrics.groundedness or 0
                stats["total_relevance"] += result.metrics.relevance or 0
                stats["total_coherence"] += result.metrics.coherence or 0
                stats["total_fluency"] += result.metrics.fluency or 0
        else:
            stats["failed"] += 1

    # Calculate averages
    summary = {}
    for model_name, stats in model_stats.items():
        count = stats["count"]
        scored_count = stats["scored_count"]

        summary[model_name] = {
            "count": count,
            "failed": stats["failed"],
            "avg_latency_ms": stats["total_latency_ms"] / count if count > 0 else 0,
            "avg_groundedness": stats["total_groundedness"] / scored_count
            if scored_count > 0
            else None,
            "avg_relevance": stats["total_relevance"] / scored_count if scored_count > 0 else None,
            "avg_coherence": stats["total_coherence"] / scored_count if scored_count > 0 else None,
            "avg_fluency": stats["total_fluency"] / scored_count if scored_count > 0 else None,
            "total_prompt_tokens": stats["total_prompt_tokens"],
            "total_completion_tokens": stats["total_completion_tokens"],
            "total_cached_tokens": stats["total_cached_tokens"],
            "total_tokens": stats["total_tokens"],
        }

    return summary


# ============================================================================
# Batch Metrics Calculation
# ============================================================================


async def calculate_all_metrics(
    results: list[EvaluationResult],
    judge_model: LLMModel,
    pages: list[str],
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    progress_callback: Any | None = None,
) -> list[EvaluationResult]:
    """
    Calculate metrics for all evaluation results in parallel.

    Parallelization: Each (model, conversation, turn) combination is independent
    and can be scored concurrently, controlled by max_concurrent.

    Args:
        results: List of evaluation results to score.
        judge_model: The LLM model to use as judge.
        pages: List of page content strings for reference_content lookup.
        max_concurrent: Maximum concurrent API calls (shares limit with LLM eval).
        progress_callback: Optional callback(completed, total) for progress.

    Returns:
        List of evaluation results with metrics populated.
    """
    # Filter to scorable results only
    scorable_results = [r for r in results if r.success and r.response]
    total = len(results)
    scorable_count = len(scorable_results)

    if scorable_count == 0:
        logger.info("No scorable results to calculate metrics for")
        if progress_callback:
            progress_callback(total, total)
        return results

    # Create client once (reuse across all scoring calls)
    client, model_name = create_judge_client(judge_model)
    logger.info(f"Created judge client for {judge_model.name}")

    try:
        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = [0]  # Mutable counter via list

        async def score_with_semaphore(result: EvaluationResult) -> None:
            """Score a single result with semaphore control."""
            # Look up page content for this result
            reference_content = None
            if result.page_reference is not None and 0 <= result.page_reference < len(pages):
                reference_content = pages[result.page_reference]

            async with semaphore:
                metrics_result = await calculate_metrics(
                    evaluation_result=result,
                    judge_model=judge_model,
                    reference_content=reference_content,
                    client=client,
                    model_name=model_name,
                )
                if metrics_result["ok"]:
                    result.metrics = metrics_result["metrics"]
                else:
                    logger.warning(
                        f"Failed to calculate metrics for {result.model_name}: "
                        f"{metrics_result.get('error')}"
                    )

                # Update progress
                completed[0] += 1
                if progress_callback:
                    # Report progress as fraction of total (including non-scorable)
                    non_scorable = total - scorable_count
                    progress_callback(non_scorable + completed[0], total)

        # Create tasks for all scorable results
        tasks = [score_with_semaphore(result) for result in scorable_results]

        logger.info(
            f"Starting parallel metric scoring: {scorable_count} scorable results, "
            f"max_concurrent={max_concurrent}"
        )

        # Run all scoring in parallel
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Calculated metrics for {completed[0]}/{scorable_count} scorable results")

    finally:
        # Properly close the async client to avoid "Event loop is closed" on Windows
        await client.close()

    return results
