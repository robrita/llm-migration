# Research: LLM Evaluation Tool

**Feature Branch**: `001-llm-eval-tool`  
**Created**: 2026-01-28  
**Purpose**: Resolve unknowns from Technical Context and establish best practices

## Research Topics

### 1. RAG Evaluation Metrics Implementation

**Context**: Need to implement 5 standard RAG metrics (groundedness, relevance, coherence, fluency, latency)

**Decision**: Use Azure AI Foundry evaluation patterns with LLM-as-judge approach

**Rationale**: 
- Azure AI Foundry provides well-documented metric definitions and prompts
- LLM-as-judge approach is industry standard for semantic evaluation
- Can use the generator model (already configured) as the judge to avoid additional dependencies

**Implementation Approach**:
| Metric | Method | Score Range |
|--------|--------|-------------|
| Latency | Direct measurement (time.perf_counter) | Milliseconds |
| Groundedness | LLM judge: "Is the response grounded in the provided context?" | 1-5 scale |
| Relevance | LLM judge: "Does the response answer the question?" | 1-5 scale |
| Coherence | LLM judge: "Is the response logically consistent and well-structured?" | 1-5 scale |
| Fluency | LLM judge: "Is the response grammatically correct and readable?" | 1-5 scale |

**Alternatives Considered**:
- External evaluation frameworks (ragas, deepeval): Rejected - adds new dependencies, violates constitution principle
- Manual scoring: Rejected - not scalable for parallel evaluation

---

### 2. Parallel Async Execution Pattern

**Context**: Need to execute evaluations across models and questions in parallel with configurable concurrency

**Decision**: Use `asyncio.gather()` with `asyncio.Semaphore` for concurrency limiting

**Rationale**:
- Already used in app.py for parallel extraction (`process_file_with_services_async`)
- Semaphore provides simple concurrency control (default 10, configurable)
- Works well with Streamlit's `asyncio.run()` pattern

**Implementation Pattern**:
```python
async def run_parallel_evaluations(
    models: list[LLMModel],
    questions: list[Question],
    max_concurrent: int = 10
) -> list[EvaluationResult]:
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_limit(model, question):
        async with semaphore:
            return await evaluate_single(model, question)
    
    tasks = [
        evaluate_with_limit(model, question)
        for model in models
        for question in questions
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

**Alternatives Considered**:
- ThreadPoolExecutor: Rejected - async is already established pattern in codebase
- multiprocessing: Rejected - overkill for I/O-bound API calls

---

### 3. PDF Text Extraction

**Context**: Need to extract text from uploaded PDFs for question generation

**Decision**: Use PyMuPDF (already in dependencies as `pymupdf>=1.26.5`)

**Rationale**:
- Already a project dependency - no new packages needed
- Fast and reliable text extraction
- Handles complex PDF layouts well
- Cross-platform (Windows + Linux)

**Implementation Pattern**:
```python
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    return "\n".join(text_parts)
```

**Alternatives Considered**:
- PyPDF2: Less reliable with complex layouts
- pdfplumber: Not in current dependencies
- Azure Document Intelligence: Overkill for simple text extraction, adds API cost

---

### 4. Multi-Turn Question Generation

**Context**: Need to generate multi-turn Q&A datasets from grounding documents

**Decision**: Use structured prompting with the generator model to create conversation chains

**Rationale**:
- Leverages existing OpenAI SDK dependency
- Generator model is configured separately from evaluation models (per clarifications)
- JSON mode ensures parseable output

**Implementation Pattern**:
```python
QUESTION_GENERATION_PROMPT = """
You are a question generator for RAG evaluation. Given the following document content,
generate {num_questions} multi-turn conversation chains.

Each chain should have 2-3 turns where follow-up questions reference previous context.

Output as JSON array:
[
  {
    "conversation_id": "conv_1",
    "turns": [
      {"turn": 1, "question": "...", "ground_truth": "..."},
      {"turn": 2, "question": "...", "ground_truth": "..."}
    ]
  }
]

Document:
{document_text}
"""
```

**Alternatives Considered**:
- Manual question creation: Rejected - not scalable
- Pre-made datasets only: Rejected - users need document-specific questions
- Azure AI Foundry SDK generators: Adds new dependency

---

### 5. Retry and Timeout Patterns

**Context**: Need 60s timeout per model, 3 retries with exponential backoff for transient errors

**Decision**: Use `asyncio.wait_for()` for timeout and custom retry decorator

**Rationale**:
- Native asyncio patterns - no new dependencies
- Exponential backoff is industry standard for rate limiting (2^n seconds)
- Matches Azure SDK retry patterns

**Implementation Pattern**:
```python
async def call_with_retry(
    func,
    max_retries: int = 3,
    timeout_seconds: float = 60.0,
    retryable_codes: set = {429, 503}
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            result = await asyncio.wait_for(func(), timeout=timeout_seconds)
            return {"ok": True, "result": result}
        except asyncio.TimeoutError:
            return {"ok": False, "error": f"Timeout after {timeout_seconds}s"}
        except Exception as e:
            if hasattr(e, 'status_code') and e.status_code in retryable_codes:
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
            return {"ok": False, "error": str(e)}
```

**Alternatives Considered**:
- tenacity library: Adds new dependency
- No retry: Poor UX for transient failures

---

### 6. Progress Display in Streamlit

**Context**: Need real-time progress during parallel evaluation

**Decision**: Use `st.progress()` with `st.empty()` placeholders

**Rationale**:
- Native Streamlit components
- Matches existing patterns in app.py (spinners, status updates)
- Progress bar updates work within async context

**Implementation Pattern**:
```python
progress_bar = st.progress(0)
status_text = st.empty()
total = len(models) * len(questions)
completed = 0

for result in results_as_completed:
    completed += 1
    progress_bar.progress(completed / total)
    status_text.text(f"Completed {completed}/{total} evaluations")
```

**Alternatives Considered**:
- st.status(): Good for sequential, less suited for parallel
- Custom JavaScript: Violates simplicity principle

---

## Dependencies Summary

| Dependency | Status | Usage |
|------------|--------|-------|
| openai | Existing | LLM API calls, question generation |
| pymupdf (fitz) | Existing | PDF text extraction |
| pydantic | Existing | Data models, validation |
| asyncio | Built-in | Parallel execution |
| plotly | Existing | Dashboard charts |
| streamlit | Existing | UI components |

**No new dependencies required** - all functionality can be implemented with existing stack.

---

## Open Questions (Resolved)

All questions from Technical Context have been resolved through research:

| Question | Resolution |
|----------|------------|
| RAG metrics implementation | LLM-as-judge with structured prompts |
| Parallel execution pattern | asyncio.gather() with Semaphore |
| PDF extraction library | PyMuPDF (already in deps) |
| Question generation approach | Structured LLM prompting with JSON output |
| Retry/timeout implementation | asyncio.wait_for() + custom retry logic |
| Progress display | st.progress() + st.empty() |
