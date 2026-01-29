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
import re
from functools import lru_cache
from pathlib import Path
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
# Prompt Loading
# ============================================================================

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=4)
def load_prompty(metric_name: str) -> str:
    """
    Load and parse a prompty file, returning the user prompt template.

    Uses Jinja2-style {{variable}} placeholders which are converted to
    Python format-style {variable} for string formatting.

    Args:
        metric_name: Name of the metric (groundedness, relevance, coherence, fluency).

    Returns:
        The user prompt template string with Python format placeholders.

    Raises:
        FileNotFoundError: If the prompty file doesn't exist.
        ValueError: If the prompty file is malformed.
    """
    prompty_path = PROMPTS_DIR / f"{metric_name}.prompty"

    if not prompty_path.exists():
        raise FileNotFoundError(f"Prompty file not found: {prompty_path}")

    content = prompty_path.read_text(encoding="utf-8")

    # Extract user section content after "user:" marker
    user_match = re.search(r"^user:\s*\n(.+)", content, re.MULTILINE | re.DOTALL)
    if not user_match:
        raise ValueError(f"No 'user:' section found in {prompty_path}")

    user_prompt = user_match.group(1).strip()

    # Convert Jinja2 {{var}} to Python {var} for str.format()
    user_prompt = re.sub(r"\{\{(\w+)\}\}", r"{\1}", user_prompt)

    logger.debug(f"Loaded prompty: {metric_name}")
    return user_prompt


def get_metric_prompts() -> dict[str, str]:
    """
    Load all metric prompts from prompty files.

    Returns:
        Dict mapping metric names to their prompt templates.
    """
    return {
        "groundedness": load_prompty("groundedness"),
        "relevance": load_prompty("relevance"),
        "coherence": load_prompty("coherence"),
        "fluency": load_prompty("fluency"),
    }


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


async def calculate_single_metric(
    metric_name: str,
    prompt: str,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    model_name: str,
    use_text_format: bool = False,
) -> dict[str, Any]:
    """
    Calculate a single RAG metric using LLM-as-judge.

    Args:
        metric_name: Name of the metric (groundedness, relevance, coherence, fluency).
        prompt: The formatted prompt for this specific metric.
        client: Async OpenAI client.
        model_name: Model/deployment name.
        use_text_format: If True, expect text response with <S2>score</S2> tags.

    Returns:
        Dict with:
        - ok (bool): Whether calculation succeeded
        - score (int): The metric score 1-5 (if ok)
        - error (str): Error message (if not ok)
    """
    try:
        if use_text_format:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            response_text = response.choices[0].message.content

            # Extract score from <S2>score</S2> tags
            score_match = re.search(r"<S2>\s*(\d)\s*</S2>", response_text)
            if not score_match:
                return {"ok": False, "error": f"Missing <S2> tag in response for {metric_name}"}
            score = int(score_match.group(1))
        else:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Respond only with JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content
            scores = json.loads(response_text)
            score = scores.get("score")

            if score is None:
                return {"ok": False, "error": f"Missing 'score' in response for {metric_name}"}

        logger.debug(f"Calculated {metric_name}: {score}")
        return {"ok": True, "score": score}

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {metric_name} response: {e}")
        return {"ok": False, "error": f"Invalid JSON response for {metric_name}: {e}"}

    except Exception as e:
        logger.error(f"Error calculating {metric_name}: {e}")
        return {"ok": False, "error": str(e)}


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

    # Use provided client or create a new one
    if client is not None and model_name is not None:
        _client = client
        _model_name = model_name
    else:
        _client, _model_name = create_judge_client(judge_model)

    # Load prompt templates from prompty files
    metric_templates = get_metric_prompts()

    # Build metric-specific prompts with tailored context
    # Relevance uses query= for conversation history context
    conversation_query = evaluation_result.question
    if evaluation_result.chat_history:
        # Format as conversation history for relevance evaluation
        history_parts = []
        for item in evaluation_result.chat_history:
            history_parts.append(f"{item.role}: {item.content}")
        history_parts.append(f"user: {evaluation_result.question}")
        conversation_query = "\n".join(history_parts)

    prompts = {
        "groundedness": metric_templates["groundedness"].format(
            query=evaluation_result.question,
            context=reference_content or "No context provided",
            response=evaluation_result.response,
        ),
        "relevance": metric_templates["relevance"].format(
            query=conversation_query,
            response=evaluation_result.response,
        ),
        "coherence": metric_templates["coherence"].format(
            query=evaluation_result.question,
            response=evaluation_result.response,
        ),
        "fluency": metric_templates["fluency"].format(
            response=evaluation_result.response,
        ),
    }

    # Track which metrics use text format (with S2 tags) vs JSON
    text_format_metrics = {"groundedness", "coherence", "fluency"}

    # Run all 4 metric calculations in parallel
    tasks = [
        calculate_single_metric(
            metric_name,
            prompt,
            _client,
            _model_name,
            use_text_format=(metric_name in text_format_metrics),
        )
        for metric_name, prompt in prompts.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Assemble results into RAGMetrics (None for failed metrics)
    metric_scores: dict[str, float | None] = {}
    metric_names = list(prompts.keys())
    errors = []

    for i, result in enumerate(results):
        metric_name = metric_names[i]
        if isinstance(result, Exception):
            logger.error(f"Exception calculating {metric_name}: {result}")
            metric_scores[metric_name] = None
            errors.append(f"{metric_name}: {result}")
        elif result.get("ok"):
            metric_scores[metric_name] = result["score"]
        else:
            metric_scores[metric_name] = None
            errors.append(f"{metric_name}: {result.get('error')}")

    metrics = RAGMetrics(
        groundedness=metric_scores.get("groundedness"),
        relevance=metric_scores.get("relevance"),
        coherence=metric_scores.get("coherence"),
        fluency=metric_scores.get("fluency"),
    )

    logger.debug(
        f"Calculated metrics for {evaluation_result.model_name}: "
        f"G={metrics.groundedness} R={metrics.relevance} "
        f"C={metrics.coherence} F={metrics.fluency}"
    )

    # Return success if at least one metric was calculated
    if any(v is not None for v in metric_scores.values()):
        return {"ok": True, "metrics": metrics}
    return {"ok": False, "error": "; ".join(errors)}


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
