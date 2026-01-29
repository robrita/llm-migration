# Quickstart: LLM Evaluation Tool

**Feature Branch**: `001-llm-eval-tool`  
**Created**: 2026-01-28  
**Purpose**: Developer guide to implement and validate the feature

## Prerequisites

1. **Environment Setup**
   ```bash
   # Clone and install
   git clone <repo>
   cd llm-migration
   uv sync
   
   # Activate virtual environment
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Azure OpenAI Access**
   - At least one Azure OpenAI deployment (for generator model)
   - Additional deployments for evaluation models (up to 5)
   - API keys or Azure Identity configured

3. **Environment Variables** (optional, can configure in UI)
   ```bash
   # .env file
   AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
   AZURE_OPENAI_KEY=<your-key>
   AZURE_OPENAI_DEPLOYMENT=<deployment-name>
   ```

## Implementation Order

### Phase 1: Foundation (Blocking)

```bash
# Create directories
mkdir -p outputs/llm_eval/datasets outputs/llm_eval/runs
```

1. **helpers/llm_models.py** - Model configuration and persistence
   - `LLMModel`, `LLMEvalConfig` Pydantic models
   - `load_config()`, `save_config()` functions
   - `validate_model_connection()` async function

2. **helpers/pdf_extractor.py** - PDF text extraction
   - `extract_text_from_pdf()` using PyMuPDF

### Phase 2: User Story 1 (Model Configuration)

3. **pages/pg2_LLM_Evaluation.py** - Main page skeleton
   - Section 0️⃣: Configuration
   - Model add/edit/delete UI
   - Generator model configuration

### Phase 3: User Story 2 (Question Generation)

4. **helpers/question_generator.py** - Question generation
   - `generate_questions()` using generator model
   - JSON output parsing

5. **pages/pg2_LLM_Evaluation.py** - Add sections
   - Section 1️⃣: Document Upload
   - Section 2️⃣: Question Dataset

### Phase 4: User Story 3 (Parallel Evaluation)

6. **helpers/llm_eval.py** - Evaluation execution
   - `evaluate_single()` with timeout/retry
   - `run_parallel_evaluations()` with asyncio

7. **pages/pg2_LLM_Evaluation.py** - Add section
   - Section 3️⃣: Run Evaluation
   - Progress display

### Phase 5: User Story 4 (Dashboard)

8. **helpers/rag_metrics.py** - Metric calculation
   - `calculate_metrics()` using judge model
   - `aggregate_run_metrics()` for summaries

9. **pages/pg2_LLM_Evaluation.py** - Add section
   - Section 4️⃣: Results Dashboard
   - Plotly charts for comparison

## Validation Checklist

### Unit Tests

```bash
# Run all unit tests
make test-unit

# Run specific test file
uv run pytest tests/test_llm_eval.py -v -m unit
```

**Test Coverage Requirements**:
- [ ] `test_llm_models.py`: Config save/load, model validation
- [ ] `test_pdf_extractor.py`: Text extraction, error handling
- [ ] `test_rag_metrics.py`: Metric calculation
- [ ] `test_llm_eval.py`: Parallel execution, timeout, retry

### Manual Validation

1. **Model Configuration (US1)**
   - [ ] Add 5 models → limit enforced
   - [ ] Save config → persists to `.conf/llm_eval_config.json`
   - [ ] Delete model → UI updates
   - [ ] Validate connection → shows latency or error

2. **Question Generation (US2)**
   - [ ] Upload 50MB PDF → error shown
   - [ ] Upload valid PDF → text extracted
   - [ ] Generate questions → JSON saved to `outputs/llm_eval/datasets/`
   - [ ] Select/deselect questions → UI responds

3. **Parallel Evaluation (US3)**
   - [ ] Run with 2 models, 5 questions → parallel progress
   - [ ] Timeout model → marked failed, others continue
   - [ ] Results saved to `outputs/llm_eval/runs/`

4. **Dashboard (US4)**
   - [ ] View metrics → all 5 displayed
   - [ ] Compare models → side-by-side charts
   - [ ] Filter by run → dashboard updates

### Performance Validation

```python
# Test parallel speedup (should be ~3x faster)
import time

# Sequential baseline
start = time.time()
for model in models:
    for question in questions:
        evaluate_single(model, question)
sequential_time = time.time() - start

# Parallel execution
start = time.time()
run_parallel_evaluations(models, dataset, settings)
parallel_time = time.time() - start

speedup = sequential_time / parallel_time
assert speedup >= 3.0, f"Speedup only {speedup:.1f}x, expected 3x"
```

## Code Quality

```bash
# Before committing
make format  # Ruff auto-fix + format
make lint    # Verify no issues
make test-unit  # All tests pass
```

## File Checklist

| File | Status | Description |
|------|--------|-------------|
| `pages/pg2_LLM_Evaluation.py` | NEW | Main evaluation page |
| `helpers/llm_models.py` | NEW | Model config, persistence |
| `helpers/llm_eval.py` | NEW | Parallel evaluation logic |
| `helpers/rag_metrics.py` | NEW | RAG metric calculations |
| `helpers/pdf_extractor.py` | NEW | PDF text extraction |
| `helpers/question_generator.py` | NEW | Q&A generation |
| `helpers/utils.py` | MODIFY | Add page link to sidebar |
| `tests/test_llm_eval.py` | NEW | Unit tests |
| `tests/test_llm_models.py` | NEW | Unit tests |
| `tests/test_rag_metrics.py` | NEW | Unit tests |
| `tests/test_pdf_extractor.py` | NEW | Unit tests |
| `outputs/llm_eval/` | NEW DIR | Output directory |
| `.conf/llm_eval_config.json` | NEW | Config file (git-ignored) |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found: fitz" | `uv sync` to install pymupdf |
| "Timeout after 60s" | Increase `timeout_seconds` in settings |
| "Rate limited (429)" | Reduce `max_concurrent_calls` |
| "No extractable text" | PDF may be image-based (OCR not supported) |
| Config not persisting | Check `.conf/` directory exists and is writable |

## Success Criteria Verification

| Criterion | How to Verify |
|-----------|---------------|
| SC-001: Configure model <3 min | Time manual configuration |
| SC-002: Question gen <2 min | Time with 50-page PDF |
| SC-003: 3x speedup | Performance test above |
| SC-004: Metrics in <5s | Time dashboard load |
| SC-005: 90% completion rate | Usability test with 10 users |
| SC-006: Failure isolation | Disconnect one model, verify others complete |
| SC-007: Audit logs | Check `outputs/llm_eval/runs/` for timestamps |
