# Feature Specification: LLM Evaluation Tool

**Feature Branch**: `001-llm-eval-tool`  
**Created**: 2026-01-28  
**Status**: Draft  
**Input**: User description: "Add a new page for LLM Evaluation Tool to easily test multiple LLM models in parallel"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Configure LLM Models for Evaluation (Priority: P1)

As an AI developer, I want to configure up to 5 LLM models with their endpoints and credentials so that I can compare their performance on the same evaluation dataset.

**Why this priority**: Model configuration is the prerequisite for all evaluation workflows. Without models configured, no evaluation can run.

**Independent Test**: Can be fully tested by adding 2-3 models with mock/real endpoints and verifying they appear in the model list ready for evaluation.

**Acceptance Scenarios**:

1. **Given** I am on the LLM Evaluation page, **When** I click "Add Model", **Then** I see a configuration form for model name, endpoint URL, API key, and model type
2. **Given** I have configured a model, **When** I save the configuration, **Then** the model appears in my configured models list with a status indicator
3. **Given** I have 5 models configured, **When** I try to add a 6th model, **Then** the system prevents addition and shows a limit message
4. **Given** I have configured models, **When** I revisit the page, **Then** my model configurations persist (securely stored locally)

---

### User Story 2 - Generate Multi-Turn Question Dataset (Priority: P2)

As an AI developer, I want to upload a grounding document (PDF) and generate a multi-turn question dataset so that I can evaluate how well LLMs handle contextual conversations based on the document.

**Why this priority**: The evaluation dataset is essential for running evaluations. This enables users to create test data from their own documents rather than requiring pre-made datasets.

**Independent Test**: Can be fully tested by uploading a sample PDF and generating questions, then reviewing the generated Q&A pairs without running any evaluation.

**Acceptance Scenarios**:

1. **Given** I am on the LLM Evaluation page, **When** I upload a PDF document, **Then** the system extracts text and shows upload success
2. **Given** I have uploaded a document, **When** I click "Generate Questions", **Then** the system generates multi-turn Q&A pairs with ground truth answers
3. **Given** the system has generated questions, **When** I view the dataset, **Then** I see question-answer pairs organized by conversation turns
4. **Given** I have a generated dataset, **When** I want to modify it, **Then** I can select/deselect individual questions or use "Select All"

---

### User Story 3 - Run Parallel Evaluation (Priority: P3)

As an AI developer, I want to run evaluations across all selected models and questions in parallel so that I can quickly compare model performance without waiting for sequential execution.

**Why this priority**: Parallel execution is the core differentiator of this tool, dramatically reducing evaluation time compared to sequential testing.

**Independent Test**: Can be fully tested by selecting 2+ models and 5+ questions, running evaluation, and observing parallel progress indicators.

**Acceptance Scenarios**:

1. **Given** I have models configured and questions selected, **When** I click "Run Evaluation", **Then** all selected models are queried in parallel
2. **Given** evaluation is running, **When** I view the progress, **Then** I see real-time status for each model's progress
3. **Given** evaluation completes, **When** I view results, **Then** I see responses from each model organized by question
4. **Given** a model fails during evaluation, **When** viewing results, **Then** I see the error message for that specific model while other results display normally

---

### User Story 4 - View RAG Evaluation Dashboard (Priority: P4)

As an AI developer, I want to see a comprehensive dashboard with standard RAG evaluation metrics so that I can make informed decisions about which model performs best for my use case.

**Why this priority**: The dashboard is the primary value output—without meaningful metrics visualization, the tool provides little actionable insight.

**Independent Test**: Can be fully tested with pre-populated evaluation results to verify all metrics calculate and display correctly.

**Acceptance Scenarios**:

1. **Given** evaluation has completed, **When** I view the dashboard, **Then** I see metrics including latency, groundedness, relevance, coherence, and fluency
2. **Given** I am viewing the dashboard, **When** I compare models, **Then** I see side-by-side metric comparisons in charts/tables
3. **Given** I am viewing metrics, **When** I hover over a data point, **Then** I see detailed breakdown of that metric's calculation
4. **Given** multiple evaluation runs exist, **When** I filter by run, **Then** the dashboard updates to show selected run's metrics

---

### Edge Cases

- What happens when a model endpoint is unreachable during evaluation? → Show error status for that model, continue with others
- What happens when the PDF upload exceeds size limits? → Show validation error before upload completes (assume 50MB limit based on Azure patterns)
- What happens when question generation fails for a PDF? → Show error with specific reason (e.g., "PDF contains no extractable text")
- What happens when evaluation is interrupted mid-run? → Preserve partial results, allow resume or restart
- What happens when multiple users evaluate simultaneously? → Session isolation ensures each user sees only their results
- What happens when a model takes excessively long to respond? → Per-model timeout (60 seconds default) marks that model's request as failed, other models continue unaffected
- What happens when an LLM API returns a transient error (429, 503)? → Retry up to 3 times with exponential backoff, then mark as failed
- How many concurrent API calls should the system make? → Configurable limit, default 10 concurrent calls

## Requirements *(mandatory)*

### Functional Requirements

**Model Configuration**
- **FR-001**: System MUST allow users to add up to 5 LLM models for evaluation
- **FR-002**: System MUST capture model name, endpoint URL, API key, and model type for each configured model
- **FR-003**: System MUST validate endpoint connectivity before allowing evaluation runs
- **FR-004**: System MUST persist model configurations locally in `.conf/` directory (git-ignored, following existing credential storage pattern)
- **FR-005**: System MUST allow users to edit and delete configured models
- **FR-005a**: System MUST allow configuration of a separate "generator model" for question dataset generation (distinct from evaluation models)

**Document & Dataset Management**
- **FR-006**: System MUST accept PDF file uploads as grounding documents
- **FR-007**: System MUST extract text content from uploaded PDFs for question generation
- **FR-008**: System MUST generate multi-turn question datasets from grounding documents
- **FR-008a**: System MUST save generated question datasets to local JSON files (separate file per dataset for scalability)
- **FR-009**: System MUST display generated questions with checkboxes for selection
- **FR-010**: System MUST provide "Select All" / "Deselect All" functionality for questions
- **FR-011**: System MUST show ground truth answers alongside generated questions

**Evaluation Execution**
- **FR-012**: System MUST execute evaluations across selected models in parallel
- **FR-013**: System MUST execute evaluations across selected questions in parallel within each model
- **FR-014**: System MUST display real-time progress during evaluation runs
- **FR-015**: System MUST handle model failures gracefully without blocking other models
- **FR-016**: System MUST preserve audit logs for all evaluation runs (following existing logging patterns)
- **FR-016a**: System MUST include prior conversation turns as context when evaluating multi-turn questions (true multi-turn evaluation)
- **FR-016b**: System MUST trim conversation history to keep only the latest N user/assistant pairs (default 10, configurable) to manage context window limits
- **FR-016c**: System MUST save evaluation results to local JSON files (separate file per evaluation run for scalability)

**Dashboard & Metrics**
- **FR-017**: System MUST calculate and display latency metrics (response time per query)
- **FR-018**: System MUST calculate and display groundedness scores (how well responses align with source document)
- **FR-019**: System MUST calculate and display relevance scores (how well responses answer the question)
- **FR-020**: System MUST calculate and display coherence scores (logical flow and consistency)
- **FR-021**: System MUST calculate and display fluency scores (grammatical correctness and readability)
- **FR-022**: System MUST provide side-by-side model comparison visualizations
- **FR-023**: System MUST allow filtering results by evaluation run
- **FR-023a**: System MUST load dashboard data from local JSON files (evaluation results, metrics)

### Non-Functional Requirements

- **NFR-001**: UI MUST reuse existing Streamlit patterns (two-column layout, numbered sections, container borders)
- **NFR-002**: System MUST NOT introduce new persistence technologies beyond JSON files in existing directories
- **NFR-002a**: System MUST use separate JSON files for each dataset/evaluation run to support scalability and avoid large monolithic files
- **NFR-003**: API credentials MUST be stored securely in environment variables or local config files (never hardcoded)
- **NFR-004**: System MUST follow existing code quality standards (Ruff, Python 3.11+, type hints)
- **NFR-005**: System MUST enforce a 60-second per-model timeout during evaluation; timed-out models are marked as failed without blocking other models
- **NFR-006**: System MUST retry transient errors (429 rate limit, 503 service unavailable) up to 3 times with exponential backoff before marking as failed
- **NFR-007**: System MUST limit concurrent API calls to a configurable maximum (default 10) to balance speed vs. rate limit risk

### Key Entities

- **LLMModel**: Represents a configured LLM for evaluation (name, endpoint, API key, model type, status)
- **GroundingDocument**: Represents an uploaded PDF with extracted text content (filename, content, upload timestamp)
- **QuestionDataset**: Collection of multi-turn Q&A pairs generated from a grounding document (questions, ground truth answers, conversation context)
- **EvaluationRun**: A single execution of evaluation across models and questions (run ID, timestamp, models, questions, status)
- **EvaluationResult**: Results for a single model-question pair (model, question, response, metrics, latency)
- **RAGMetrics**: Calculated metrics for evaluation (groundedness, relevance, coherence, fluency, latency)

### Assumptions

- PDF extraction uses existing Python libraries (e.g., PyPDF2, pdfplumber) without introducing new dependencies where possible
- Question generation uses a dedicated "generator model" configured separately from evaluation models (avoids circular dependency)
- Standard RAG metrics follow Azure AI Foundry evaluation patterns
- Multi-turn conversations include prior turns as context when evaluating each question (true multi-turn evaluation)
- Conversation history is trimmed to keep only the latest 10 user/assistant pairs by default (configurable) to manage context window limits
- Maximum 50MB PDF file size (consistent with Azure patterns for document processing)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can configure up to 5 LLM models in under 3 minutes each
- **SC-002**: Question generation produces results within 2 minutes for documents up to 50 pages
- **SC-003**: Parallel evaluation of 5 models with 10 questions completes at least 3x faster than sequential execution
- **SC-004**: Dashboard displays all 5 standard RAG metrics (latency, groundedness, relevance, coherence, fluency) within 5 seconds of evaluation completion
- **SC-005**: 90% of users can complete a full evaluation workflow (configure → upload → generate → evaluate → view dashboard) without external guidance
- **SC-006**: Model failures during evaluation do not block completion of other model evaluations
- **SC-007**: All evaluation runs are logged with timestamps and can be filtered in the dashboard

### Out of Scope

- Chat with data UI (conversational interface for querying documents)
- Persistent cloud storage for evaluation results (local JSON only)
- Real-time collaboration between multiple users
- Custom metric definitions (only standard RAG metrics)
- Automated model recommendations based on results

## Clarifications

### Session 2026-01-28

- Q: When an LLM model takes excessively long to respond during evaluation, what timeout behavior should apply? → A: Per-model timeout (60 seconds default) marks that model's request as failed, other models continue unaffected
- Q: When an LLM API returns a transient error (429, 503), should the system retry? → A: Retry up to 3 times with exponential backoff, then mark as failed
- Q: Which LLM should be used to generate the multi-turn question dataset from the uploaded PDF? → A: Use a dedicated "generator model" configured separately from evaluation models
- Q: What is the maximum number of concurrent API calls during parallel evaluation? → A: Configurable limit, default 10 concurrent calls (balances speed vs. rate limit risk)
- Q: How should the system handle multi-turn conversation context during evaluation? → A: Include prior turns as context (true multi-turn evaluation)
- Q: How much conversation history should be retained? → A: Trim to latest 10 user/assistant pairs by default (configurable) to manage context window limits
- Q: How should generated data be persisted? → A: Save all generated data locally as separate JSON files (datasets, evaluation outputs, dashboard data) for scalability
