# Internal Contracts: LLM Evaluation Tool

**Feature Branch**: `001-llm-eval-tool`  
**Created**: 2026-01-28  
**Purpose**: Define function signatures and return types for helper modules

## Overview

This feature is a Streamlit application without REST APIs. Contracts define internal function signatures for helper modules following the existing handler contract pattern.

## Contract Pattern

All public functions follow a consistent return pattern:

```python
# Success case
{"ok": True, "result": <data>, "processing_time": <float>}

# Failure case  
{"ok": False, "error": "<error message>", "details": <optional dict>}
```

---

## Module: `helpers/llm_models.py`

### `load_config() -> LLMEvalConfig`
Load configuration from disk or return default.

### `save_config(config: LLMEvalConfig) -> dict[str, Any]`
```python
# Input
config: LLMEvalConfig  # Full configuration object

# Output (success)
{"ok": True, "path": str}

# Output (failure)
{"ok": False, "error": str}
```

### `validate_model_connection(model: LLMModel) -> dict[str, Any]`
```python
# Input
model: LLMModel  # Model to validate

# Output (success)
{"ok": True, "model_name": str, "latency_ms": float}

# Output (failure)
{"ok": False, "error": str, "model_name": str}
```

---

## Module: `helpers/pdf_extractor.py`

### `extract_text_from_pdf(pdf_bytes: bytes) -> dict[str, Any]`
```python
# Input
pdf_bytes: bytes  # Raw PDF file content

# Output (success)
{
    "ok": True,
    "content": str,           # Extracted text
    "page_count": int,
    "file_size_bytes": int,
    "processing_time": float  # Seconds
}

# Output (failure)
{"ok": False, "error": str}
```

---

## Module: `helpers/question_generator.py`

### `generate_questions(document: GroundingDocument, model: LLMModel, num_conversations: int = 5) -> dict[str, Any]`
```python
# Input
document: GroundingDocument   # Source document
model: LLMModel               # Generator model
num_conversations: int        # Number of conversation chains (default 5)

# Output (success)
{
    "ok": True,
    "dataset": QuestionDataset,
    "processing_time": float
}

# Output (failure)
{"ok": False, "error": str}
```

---

## Module: `helpers/llm_eval.py`

### `evaluate_single(model: LLMModel, question: str, context: str, conversation_history: list[dict]) -> dict[str, Any]`
```python
# Input
model: LLMModel                          # Model to evaluate
question: str                            # Current question
context: str                             # Grounding document text
conversation_history: list[dict]         # Prior turns [{role, content}, ...]

# Output (success)
{
    "ok": True,
    "response": str,
    "latency_ms": float
}

# Output (failure)
{"ok": False, "error": str, "latency_ms": float}
```

### `run_parallel_evaluations(models: list[LLMModel], dataset: QuestionDataset, settings: EvalSettings) -> dict[str, Any]`
```python
# Input
models: list[LLMModel]       # Models to evaluate (max 5)
dataset: QuestionDataset     # Questions to run
settings: EvalSettings       # Concurrency, timeout, retries

# Output (success)
{
    "ok": True,
    "run": EvaluationRun,    # Complete run with all results
    "processing_time": float
}

# Output (partial success)
{
    "ok": True,
    "run": EvaluationRun,    # status=PARTIAL
    "failed_count": int,
    "processing_time": float
}

# Output (failure)
{"ok": False, "error": str}
```

---

## Module: `helpers/rag_metrics.py`

### `calculate_metrics(response: str, ground_truth: str, context: str, judge_model: LLMModel) -> dict[str, Any]`
```python
# Input
response: str         # Model's response to evaluate
ground_truth: str     # Expected answer
context: str          # Source document text
judge_model: LLMModel # Model to use as judge

# Output (success)
{
    "ok": True,
    "metrics": RAGMetrics,
    "processing_time": float
}

# Output (failure)
{"ok": False, "error": str}
```

### `aggregate_run_metrics(run: EvaluationRun) -> dict[str, Any]`
```python
# Input
run: EvaluationRun  # Completed evaluation run

# Output
{
    "ok": True,
    "summary": {
        "total_evaluations": int,
        "successful": int,
        "failed": int,
        "by_model": {
            "<model_name>": {
                "count": int,
                "avg_latency_ms": float,
                "avg_groundedness": float,
                "avg_relevance": float,
                "avg_coherence": float,
                "avg_fluency": float,
                "avg_overall": float
            }
        }
    }
}
```

---

## Error Codes

| Error Pattern | Meaning |
|--------------|---------|
| `"Timeout after Xs"` | Model did not respond within timeout |
| `"Rate limited (429)"` | API rate limit exceeded after retries |
| `"Service unavailable (503)"` | API temporarily unavailable after retries |
| `"Invalid API key"` | Authentication failed |
| `"PDF extraction failed: ..."` | Could not extract text from PDF |
| `"No extractable text"` | PDF contains no text content |
| `"Model limit reached"` | Attempted to add 6th evaluation model |

---

## Session State Keys

Streamlit session state keys used by this feature:

| Key | Type | Description |
|-----|------|-------------|
| `llm_eval_config` | `dict` | Serialized LLMEvalConfig |
| `llm_eval_document` | `GroundingDocument \| None` | Current uploaded document |
| `llm_eval_dataset` | `QuestionDataset \| None` | Current question dataset |
| `llm_eval_run` | `EvaluationRun \| None` | Current/last evaluation run |
| `llm_eval_selected_run_id` | `str \| None` | Selected run for dashboard |
