````markdown
# Tasks: LLM Evaluation Tool

**Input**: Design documents from `/specs/001-llm-eval-tool/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/internal-api.md ✅, quickstart.md ✅

**Tests**: Included per user story as unit tests with mocks (following Testing Discipline principle).

**Organization**: Tasks grouped by user story (P1 → P4) for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1, US2, US3, US4)
- File paths follow existing project structure

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create directories and initialize shared infrastructure

- [X] T001 Create output directories: `outputs/llm_eval/datasets/` and `outputs/llm_eval/runs/`
- [X] T002 Verify `.conf/` directory exists and is git-ignored in `.gitignore`
- [X] T003 Add sidebar navigation link to helpers/utils.py `render_sidebar()` function

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Pydantic models and persistence functions that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Create helpers/llm_models.py with `ModelType` enum and `LLMModel` Pydantic model
- [X] T005 Add `EvalSettings` and `LLMEvalConfig` Pydantic models to helpers/llm_models.py
- [X] T006 Implement `load_config()` and `save_config()` persistence functions in helpers/llm_models.py
- [X] T007 [P] Create helpers/pdf_extractor.py with `extract_text_from_pdf()` using PyMuPDF
- [X] T008 [P] Add `GroundingDocument`, `ConversationTurn`, `Conversation` models to helpers/llm_models.py
- [X] T009 [P] Add `QuestionDataset` model with `save_dataset()`, `load_dataset()` to helpers/llm_models.py
- [X] T010 [P] Add `RunStatus` enum, `RAGMetrics`, `EvaluationResult`, `EvaluationRun` models to helpers/llm_models.py
- [X] T011 [P] Implement `save_run()`, `load_run()`, `list_runs()` persistence functions in helpers/llm_models.py
- [X] T012 [P] Create tests/test_llm_models.py with unit tests for config save/load round-trip
- [X] T013 [P] Create tests/test_pdf_extractor.py with unit tests for text extraction (mock PDF bytes)
- [X] T014 Configure module-level logger in helpers/llm_models.py and helpers/pdf_extractor.py

**Checkpoint**: Foundation ready - all Pydantic models and persistence functions available

---

## Phase 3: User Story 1 - Configure LLM Models for Evaluation (Priority: P1) 🎯 MVP

**Goal**: Allow users to configure up to 5 LLM models with endpoints and credentials for evaluation

**Independent Test**: Add 2-3 models with mock/real endpoints, verify they persist and appear in model list

### Tests for User Story 1 ⚠️

> **Write tests FIRST, ensure they FAIL before implementation**

- [X] T015 [P] [US1] Add unit test for validate_model_connection() in tests/test_llm_models.py
- [X] T016 [P] [US1] Add unit test for add/edit/delete model operations in tests/test_llm_models.py
- [X] T017 [P] [US1] Add unit test for 5-model limit enforcement in tests/test_llm_models.py

### Implementation for User Story 1

- [X] T018 [US1] Implement `validate_model_connection()` async function in helpers/llm_models.py
- [X] T019 [US1] Create pages/pg2_LLM_Evaluation.py with page config and section 0️⃣ Configuration header
- [X] T020 [US1] Add generator model configuration form (name, endpoint, API key, deployment) in pg2_LLM_Evaluation.py
- [X] T021 [US1] Add evaluation models list with "Add Model" button and 5-model limit message in pg2_LLM_Evaluation.py
- [X] T022 [US1] Add model edit/delete actions with confirmation in pg2_LLM_Evaluation.py
- [X] T023 [US1] Add "Test Connection" button with latency display in pg2_LLM_Evaluation.py
- [X] T024 [US1] Persist config on save using save_config() in pg2_LLM_Evaluation.py
- [X] T025 [US1] Load config on page load using load_config() in pg2_LLM_Evaluation.py
- [X] T026 [US1] Add logging for model add/edit/delete/validate operations in helpers/llm_models.py

**Checkpoint**: User Story 1 complete - models can be configured, validated, persisted, and listed

---

## Phase 4: User Story 2 - Generate Multi-Turn Question Dataset (Priority: P2)

**Goal**: Upload PDF, extract text, and generate multi-turn Q&A pairs using generator model

**Independent Test**: Upload sample PDF, generate questions, verify Q&A pairs display and can be selected/deselected

### Tests for User Story 2 ⚠️

> **Write tests FIRST, ensure they FAIL before implementation**

- [X] T027 [P] [US2] Add unit test for generate_questions() with mocked LLM response in tests/test_question_generator.py
- [X] T028 [P] [US2] Add unit test for PDF size validation (50MB limit) in tests/test_pdf_extractor.py
- [X] T029 [P] [US2] Add unit test for empty PDF handling in tests/test_pdf_extractor.py

### Implementation for User Story 2

- [X] T030 [US2] Create helpers/question_generator.py with `generate_questions()` function signature
- [X] T031 [US2] Implement prompt template for multi-turn Q&A generation in helpers/question_generator.py
- [X] T032 [US2] Implement JSON parsing for generated Q&A pairs in helpers/question_generator.py
- [X] T033 [US2] Add exponential backoff retry logic (3 retries) in helpers/question_generator.py
- [X] T034 [US2] Add section 1️⃣ Document Upload with file uploader (PDF only, 50MB limit) in pg2_LLM_Evaluation.py
- [X] T035 [US2] Display upload success with page count and extracted text preview in pg2_LLM_Evaluation.py
- [X] T036 [US2] Add section 2️⃣ Question Dataset with "Generate Questions" button in pg2_LLM_Evaluation.py
- [X] T037 [US2] Display generated conversations with checkboxes per conversation in pg2_LLM_Evaluation.py
- [X] T038 [US2] Add "Select All" / "Deselect All" buttons for question selection in pg2_LLM_Evaluation.py
- [X] T039 [US2] Display ground truth answers alongside questions in expandable sections in pg2_LLM_Evaluation.py
- [X] T040 [US2] Save generated dataset to outputs/llm_eval/datasets/{dataset_id}.json in pg2_LLM_Evaluation.py
- [X] T041 [US2] Add progress indicator during question generation in pg2_LLM_Evaluation.py
- [X] T042 [US2] Add logging for document upload, extraction, and generation in helpers/question_generator.py

**Checkpoint**: User Story 2 complete - can upload PDF, generate Q&A pairs, select questions, save dataset

---

## Phase 5: User Story 3 - Run Parallel Evaluation (Priority: P3)

**Goal**: Execute evaluations across all selected models and questions in parallel with progress tracking

**Independent Test**: Select 2+ models and 5+ questions, run evaluation, observe parallel progress and results

### Tests for User Story 3 ⚠️

> **Write tests FIRST, ensure they FAIL before implementation**

- [X] T043 [P] [US3] Create tests/test_llm_eval.py with unit test for evaluate_single() with mocked API
- [X] T044 [P] [US3] Add unit test for timeout behavior (60s) in tests/test_llm_eval.py
- [X] T045 [P] [US3] Add unit test for retry logic (429, 503 errors) in tests/test_llm_eval.py
- [X] T046 [P] [US3] Add unit test for parallel execution with asyncio.gather() in tests/test_llm_eval.py
- [X] T047 [P] [US3] Add unit test for conversation history trimming (max 10 pairs) in tests/test_llm_eval.py

### Implementation for User Story 3

- [X] T048 [US3] Create helpers/llm_eval.py with module-level logger and imports
- [X] T049 [US3] Implement `evaluate_single()` async function with timeout (asyncio.wait_for) in helpers/llm_eval.py
- [X] T050 [US3] Add retry logic with exponential backoff for 429/503 errors in helpers/llm_eval.py
- [X] T051 [US3] Implement conversation history builder with trimming (max N pairs) in helpers/llm_eval.py
- [X] T052 [US3] Implement `run_parallel_evaluations()` using asyncio.gather() with Semaphore in helpers/llm_eval.py
- [X] T053 [US3] Handle partial failures (some models fail, others continue) in helpers/llm_eval.py
- [X] T054 [US3] Add section 3️⃣ Run Evaluation with model/question selection summary in pg2_LLM_Evaluation.py
- [X] T055 [US3] Add "Run Evaluation" button with validation (models + questions selected) in pg2_LLM_Evaluation.py
- [X] T056 [US3] Display real-time progress per model using st.progress() and st.status() in pg2_LLM_Evaluation.py
- [X] T057 [US3] Display results table after completion (model, question, response preview, status) in pg2_LLM_Evaluation.py
- [X] T058 [US3] Save evaluation run to outputs/llm_eval/runs/{run_id}.json in pg2_LLM_Evaluation.py
- [X] T059 [US3] Add logging for evaluation start, progress, completion, and failures in helpers/llm_eval.py

**Checkpoint**: User Story 3 complete - can run parallel evaluations with progress, handle failures, save results

---

## Phase 6: User Story 4 - View RAG Evaluation Dashboard (Priority: P4)

**Goal**: Display comprehensive dashboard with RAG metrics (latency, groundedness, relevance, coherence, fluency)

**Independent Test**: Load pre-populated evaluation results, verify all metrics calculate and display correctly with charts

### Tests for User Story 4 ⚠️

> **Write tests FIRST, ensure they FAIL before implementation**

- [X] T060 [P] [US4] Create tests/test_rag_metrics.py with unit test for calculate_metrics() with mocked judge
- [X] T061 [P] [US4] Add unit test for aggregate_run_metrics() calculation in tests/test_rag_metrics.py
- [X] T062 [P] [US4] Add unit test for metric score validation (1-5 range) in tests/test_rag_metrics.py

### Implementation for User Story 4

- [X] T063 [US4] Create helpers/rag_metrics.py with module-level logger and imports
- [X] T064 [US4] Implement `calculate_metrics()` using LLM-as-judge pattern in helpers/rag_metrics.py
- [X] T065 [US4] Implement prompt template for scoring groundedness, relevance, coherence, fluency in helpers/rag_metrics.py
- [X] T066 [US4] Implement `aggregate_run_metrics()` for model-level summaries in helpers/rag_metrics.py
- [X] T067 [US4] Add section 4️⃣ Results Dashboard header in pg2_LLM_Evaluation.py
- [X] T068 [US4] Add run selector dropdown populated from list_runs() in pg2_LLM_Evaluation.py
- [X] T069 [US4] Display metrics summary cards (avg latency, avg scores per model) in pg2_LLM_Evaluation.py
- [X] T070 [US4] Create side-by-side bar chart comparing models using Plotly in pg2_LLM_Evaluation.py
- [X] T071 [US4] Create latency comparison chart (box plot or line) using Plotly in pg2_LLM_Evaluation.py
- [X] T072 [US4] Add detailed results table with expand for individual responses in pg2_LLM_Evaluation.py
- [X] T073 [US4] Add metric tooltips explaining groundedness, relevance, coherence, fluency in pg2_LLM_Evaluation.py
- [X] T074 [US4] Add logging for dashboard load, metric calculation, run selection in helpers/rag_metrics.py

**Checkpoint**: User Story 4 complete - full RAG dashboard with metrics, charts, comparisons, and filtering

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final improvements across all user stories

- [X] T075 [P] Add error boundary handling for all sections in pg2_LLM_Evaluation.py (graceful failures)
- [X] T076 [P] Add sample questions dataset to inputs/llm_eval/sample_questions.json for demo
- [X] T077 [P] Update README.md with LLM Evaluation Tool usage section
- [X] T078 Run `make format` and fix any Ruff lint issues
- [X] T079 Run `make test-unit` and ensure 100% pass rate
- [X] T080 Run quickstart.md validation checklist (manual verification)
- [X] T081 Performance test: verify parallel evaluation is 3x faster than sequential
- [X] T082 Security review: verify API keys only stored in .conf/ (git-ignored), never logged

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 - BLOCKS all user stories
- **Phase 3-6 (User Stories)**: All depend on Phase 2 completion
  - Stories can proceed sequentially (P1 → P2 → P3 → P4) for solo developer
  - Or in parallel if team capacity allows
- **Phase 7 (Polish)**: Depends on all desired user stories being complete

### User Story Dependencies

| Story | Depends On | Can Start After |
|-------|------------|-----------------|
| US1 (Model Config) | Phase 2 only | T014 |
| US2 (Question Gen) | Phase 2 only | T014 |
| US3 (Parallel Eval) | US1 (models), US2 (dataset) | T026, T042 |
| US4 (Dashboard) | US3 (results to display) | T059 |

### Within Each User Story

1. Tests written and FAIL first
2. Helper module implementation
3. UI integration in pg2_LLM_Evaluation.py
4. Logging added last

### Parallel Opportunities by Phase

**Phase 2 (Foundational)**:
```text
Can run in parallel after T006:
- T007 (pdf_extractor.py)
- T008 (Document models)
- T009 (Dataset models)
- T010 (Evaluation models)
- T011 (Run persistence)
- T012 (test_llm_models.py)
- T013 (test_pdf_extractor.py)
```

**Phase 3 (US1 Tests)**:
```text
Can run in parallel:
- T015, T016, T017 (all independent test files)
```

**Phase 4 (US2 Tests)**:
```text
Can run in parallel:
- T027, T028, T029 (all independent test functions)
```

**Phase 5 (US3 Tests)**:
```text
Can run in parallel:
- T043, T044, T045, T046, T047 (all independent test functions)
```

**Phase 6 (US4 Tests)**:
```text
Can run in parallel:
- T060, T061, T062 (all independent test functions)
```

---

## Parallel Example: Foundational Phase

```bash
# After T006 (config persistence) completes, launch all these together:
Task T007: "Create helpers/pdf_extractor.py with extract_text_from_pdf()"
Task T008: "Add GroundingDocument, ConversationTurn, Conversation models"
Task T009: "Add QuestionDataset model with save_dataset(), load_dataset()"
Task T010: "Add RunStatus, RAGMetrics, EvaluationResult, EvaluationRun models"
Task T011: "Implement save_run(), load_run(), list_runs() functions"
Task T012: "Create tests/test_llm_models.py unit tests"
Task T013: "Create tests/test_pdf_extractor.py unit tests"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T014)
3. Complete Phase 3: User Story 1 (T015-T026)
4. **STOP and VALIDATE**: Test model configuration independently
5. Deploy/demo if ready - users can configure models

### Incremental Delivery

| Increment | Stories | Value Delivered |
|-----------|---------|-----------------|
| MVP | Setup + Foundation + US1 | Model configuration |
| +1 | +US2 | PDF upload, question generation |
| +2 | +US3 | Parallel evaluation, results |
| +3 | +US4 | Full RAG dashboard |
| Final | +Polish | Production-ready |

### Suggested Order for Solo Developer

1. T001-T014 (Setup + Foundation) → Foundation checkpoint
2. T015-T026 (US1) → Can configure models
3. T027-T042 (US2) → Can generate questions
4. T043-T059 (US3) → Can run evaluations
5. T060-T074 (US4) → Full dashboard
6. T075-T082 (Polish) → Production-ready

---

## Notes

- All tests use mocks (no real Azure calls in unit tests)
- Integration tests (not included) would require live Azure credentials
- Each user story ends with a checkpoint for validation
- Use `asyncio.run()` in Streamlit for async operations (existing pattern from app.py)
- Follow existing handler contract pattern: `{"ok": True/False, "result"|"error": ...}`
- Commit after each logical task group
- Run `make format` before any commit

---

## Summary

| Metric | Count |
|--------|-------|
| Total Tasks | 82 |
| Setup Tasks | 3 |
| Foundational Tasks | 11 |
| US1 Tasks (P1) | 12 |
| US2 Tasks (P2) | 16 |
| US3 Tasks (P3) | 17 |
| US4 Tasks (P4) | 15 |
| Polish Tasks | 8 |
| Parallelizable Tasks | 28 (marked with [P]) |

**MVP Scope**: Phase 1 + Phase 2 + Phase 3 = 26 tasks

**Estimated Effort**: 
- MVP (US1): ~4 hours
- Full feature (US1-US4): ~16-20 hours
- With polish: ~20-24 hours

````