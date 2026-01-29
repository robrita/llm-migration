# Implementation Plan: LLM Evaluation Tool

**Branch**: `001-llm-eval-tool` | **Date**: 2026-01-28 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-llm-eval-tool/spec.md`

## Summary

Add a new Streamlit page for LLM Evaluation that enables parallel testing of up to 5 LLM models. Users configure models, upload PDF grounding documents, generate multi-turn question datasets, execute parallel evaluations with configurable concurrency, and view a comprehensive RAG metrics dashboard (latency, groundedness, relevance, coherence, fluency). All data persisted as separate JSON files for scalability.

## Technical Context

**Language/Version**: Python 3.11+ (enforced by pyproject.toml target-version = "py311")  
**Primary Dependencies**: Streamlit 1.50.0, OpenAI SDK 2.3.0, Azure Identity 1.25.1, PyMuPDF 1.26.5, Pydantic 2.10.6, Plotly 6.3.1  
**Storage**: Local JSON files in `outputs/` (evaluation results) and `.conf/` (model configs, git-ignored)  
**Testing**: pytest with unit (-m unit) and integration (-m integration) markers, fixtures in tests/conftest.py  
**Target Platform**: Windows and Linux (cross-platform via pathlib.Path)  
**Project Type**: Single Streamlit application with multi-page architecture  
**Performance Goals**: Parallel evaluation 3x faster than sequential; question generation <2 min for 50 pages  
**Constraints**: 60s per-model timeout, 10 concurrent API calls (configurable), 50MB PDF limit, retry 3x with exponential backoff  
**Scale/Scope**: Up to 5 evaluation models, 10 conversation history pairs, single-user sessions

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Status | Notes |
|-----------|-------------|--------|-------|
| I. Code Quality First | Ruff lint/format, Python 3.11+, 100 char lines, pathlib.Path | ✅ PASS | Will use existing pyproject.toml config |
| II. Security & Privacy | Credentials in .env/.conf, Pydantic validation, no hardcoding | ✅ PASS | Model API keys stored in .conf/llm_eval_config.json (git-ignored) |
| III. Handler Contract | Return dict[str, Any], include 'service' key, error handling | ⚠️ ADAPT | New pattern: evaluator functions return similar dict structure |
| IV. Testing Discipline | Unit tests with mocks, integration tests with credentials | ✅ PASS | Will add tests/test_llm_eval.py with fixtures |
| V. Observability | Module-level loggers, log key actions, timing | ✅ PASS | Will use logging.getLogger(__name__) throughout |
| Streamlit Conventions | width="stretch", st.subheader(), st.plotly_chart config | ✅ PASS | Will follow existing page patterns |

**Gate Result**: PASS - No violations requiring justification

## Project Structure

### Documentation (this feature)

```text
specs/001-llm-eval-tool/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (API schemas)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (brownfield - extends existing structure)

```text
# Existing structure (DO NOT MODIFY without justification)
app.py                           # Main Streamlit entry point
helpers/
├── utils.py                     # Shared utilities (render_sidebar, keep_state)
├── speech_personal_voice.py     # Personal Voice helpers
pages/
├── pg1_Voice_Gallery.py         # Voice Gallery page
├── pg3_Pricing.py               # Pricing page

# NEW FILES for LLM Evaluation feature
pages/
├── pg2_LLM_Evaluation.py        # NEW: Main evaluation page
helpers/
├── llm_eval.py                  # NEW: Evaluation logic, parallel execution
├── llm_models.py                # NEW: Model configuration, API clients
├── rag_metrics.py               # NEW: RAG metric calculations
├── pdf_extractor.py             # NEW: PDF text extraction
├── question_generator.py        # NEW: Multi-turn Q&A generation

inputs/
├── llm_eval/                    # NEW: Static data for evaluation
│   └── sample_questions.json    # Optional sample dataset

outputs/
├── llm_eval/                    # NEW: Evaluation output directory
│   ├── datasets/                # Generated question datasets (JSON per dataset)
│   └── runs/                    # Evaluation run results (JSON per run)

.conf/
├── llm_eval_config.json         # NEW: Model configs (git-ignored)

tests/
├── test_llm_eval.py             # NEW: Unit tests for evaluation
├── test_llm_models.py           # NEW: Unit tests for model config
├── test_rag_metrics.py          # NEW: Unit tests for metrics
├── test_pdf_extractor.py        # NEW: Unit tests for PDF extraction
```

**Structure Decision**: Extends existing Streamlit multi-page architecture. New page follows naming convention (pg2_*). Helper modules split by responsibility. Output files in dedicated subdirectory for isolation.

## Complexity Tracking

> **No violations identified** - Design follows existing patterns and constitution principles.

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| New helper modules | 5 new files in helpers/ | Single responsibility principle; avoids bloating existing utils.py |
| Output directory | outputs/llm_eval/ subdirectory | Isolates from existing outputs/temp/ used by Voice features |
| Config storage | .conf/llm_eval_config.json | Follows existing .conf/ pattern for git-ignored secrets |

## Constitution Check (Post-Design)

*Re-evaluation after Phase 1 design completion.*

| Principle | Requirement | Status | Design Evidence |
|-----------|-------------|--------|-----------------|
| I. Code Quality First | Ruff lint/format, Python 3.11+, pathlib.Path | ✅ PASS | data-model.md uses Path, type hints throughout |
| II. Security & Privacy | Credentials in .conf, Pydantic validation | ✅ PASS | LLMModel uses SecretStr for api_key, config in .conf/ |
| III. Handler Contract | Return dict[str, Any], error handling | ✅ PASS | contracts/internal-api.md defines {"ok": True/False, ...} pattern |
| IV. Testing Discipline | Unit tests with mocks | ✅ PASS | quickstart.md defines test files per module |
| V. Observability | Module-level loggers, timing | ✅ PASS | All contracts include processing_time field |
| Streamlit Conventions | st.subheader(), st.progress() | ✅ PASS | research.md specifies native Streamlit patterns |
| No New Dependencies | Use existing packages only | ✅ PASS | research.md confirms PyMuPDF, OpenAI SDK already in deps |

**Post-Design Gate Result**: ✅ PASS - All principles satisfied, no violations.

---

## Generated Artifacts

| Artifact | Path | Purpose |
|----------|------|---------|
| Research | [research.md](research.md) | Technology decisions and patterns |
| Data Model | [data-model.md](data-model.md) | Pydantic models and persistence |
| Internal API | [contracts/internal-api.md](contracts/internal-api.md) | Function signatures and contracts |
| Quickstart | [quickstart.md](quickstart.md) | Developer implementation guide |

**Next Step**: Run `/speckit.tasks` to generate Phase 2 task breakdown.
