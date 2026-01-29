"""
LLM Evaluation Tool

A comprehensive interface for:
- Configuring LLM models for evaluation (up to 5)
- Uploading PDF grounding documents
- Generating multi-turn question datasets
- Running parallel evaluations
- Viewing RAG metrics dashboard

Author: LLM Evaluation Tool Team
"""

import asyncio
import logging
import time

import streamlit as st

from helpers.llm_eval import build_evaluation_run, run_parallel_evaluations
from helpers.llm_models import (
    DEFAULT_SYSTEM_PROMPT,
    EvalSettings,
    GroundingDocument,
    LLMEvalConfig,
    LLMModel,
    ModelType,
    QuestionDataset,
    list_datasets,
    list_runs,
    load_config,
    load_dataset,
    load_run,
    save_config,
    save_dataset,
    save_run,
    validate_model_connection,
)
from helpers.pdf_extractor import MAX_PDF_SIZE_BYTES, extract_text_from_pdf
from helpers.question_generator import generate_questions
from helpers.rag_metrics import aggregate_run_metrics, calculate_all_metrics
from helpers.utils import (
    calculate_token_cost,
    format_cost,
    load_azure_openai_pricing,
    render_sidebar,
)

# Module-level logger
logger = logging.getLogger(__name__)

# ============================================================================
# Page Configuration
# ============================================================================

render_sidebar()

st.title("🧪 LLM Evaluation Tool")
st.markdown(
    """
    Compare multiple LLM models in parallel with standard RAG evaluation metrics.
    Configure up to 5 models, upload grounding documents, generate question datasets,
    and view comprehensive performance comparisons.
    """
)

# ============================================================================
# Session State Initialization
# ============================================================================


def init_session_state():
    """Initialize session state with config from disk."""
    if "llm_eval_config" not in st.session_state:
        config = load_config()
        st.session_state.llm_eval_config = config.model_dump(mode="json")
        logger.info("Loaded LLM eval config from disk")

    if "llm_eval_document" not in st.session_state:
        st.session_state.llm_eval_document = None

    if "llm_eval_documents_list" not in st.session_state:
        st.session_state.llm_eval_documents_list = None

    if "llm_eval_dataset" not in st.session_state:
        # Auto-load latest dataset if available locally
        available_datasets = list_datasets()
        if available_datasets:
            # Load most recent dataset (sorted, so last is newest)
            latest_id = available_datasets[-1]
            dataset = load_dataset(latest_id)
            if dataset:
                st.session_state.llm_eval_dataset = dataset.model_dump(mode="json")
                logger.info(f"Auto-loaded latest dataset: {latest_id}")
            else:
                st.session_state.llm_eval_dataset = None
        else:
            st.session_state.llm_eval_dataset = None

    if "llm_eval_run" not in st.session_state:
        st.session_state.llm_eval_run = None

    if "editing_model_index" not in st.session_state:
        st.session_state.editing_model_index = None

    if "show_add_model_form" not in st.session_state:
        st.session_state.show_add_model_form = False


init_session_state()


def get_config() -> LLMEvalConfig:
    """Get current config from session state."""
    return LLMEvalConfig.model_validate(st.session_state.llm_eval_config)


def update_config(config: LLMEvalConfig):
    """Update config in session state and persist to disk."""
    st.session_state.llm_eval_config = config.model_dump(mode="json")
    # Convert SecretStr to plain strings for storage
    if config.generator_model:
        st.session_state.llm_eval_config["generator_model"]["api_key"] = (
            config.generator_model.api_key.get_secret_value()
        )
    for i, model in enumerate(config.evaluation_models):
        st.session_state.llm_eval_config["evaluation_models"][i]["api_key"] = (
            model.api_key.get_secret_value()
        )
    result = save_config(config)
    if result["ok"]:
        logger.info("Config saved successfully")
    else:
        logger.error(f"Failed to save config: {result.get('error')}")
        st.error(f"Failed to save config: {result.get('error')}")


# ============================================================================
# Section 0️⃣: Configuration
# ============================================================================

st.subheader("0️⃣ Model Configuration")

config = get_config()

# Settings expander
with st.expander("⚙️ Evaluation Settings", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        max_concurrent = st.number_input(
            "Max Concurrent API Calls",
            min_value=1,
            max_value=50,
            value=config.settings.max_concurrent_calls,
            help="Maximum number of parallel API calls during evaluation",
        )
        timeout = st.number_input(
            "Timeout (seconds)",
            min_value=10,
            max_value=300,
            value=int(config.settings.timeout_seconds),
            help="Per-model response timeout",
        )

    with col2:
        max_retries = st.number_input(
            "Max Retries",
            min_value=0,
            max_value=10,
            value=config.settings.max_retries,
            help="Retry count for transient errors (429, 503)",
        )
        max_history = st.number_input(
            "Max Conversation History",
            min_value=1,
            max_value=50,
            value=config.settings.max_conversation_history,
            help="Maximum user/assistant pairs to retain for context",
        )

    system_prompt = st.text_area(
        "System Prompt",
        value=getattr(config.settings, "system_prompt", DEFAULT_SYSTEM_PROMPT),
        height=100,
        help="System prompt used for evaluation models. Grounding context will be appended.",
    )

    if st.button("💾 Save Settings", key="save_settings"):
        config.settings = EvalSettings(
            max_concurrent_calls=max_concurrent,
            timeout_seconds=float(timeout),
            max_retries=max_retries,
            max_conversation_history=max_history,
            system_prompt=system_prompt,
        )
        update_config(config)
        st.success("Settings saved!")
        st.rerun()

# ============================================================================
# Generator Model Configuration
# ============================================================================

with st.expander("🎯 Generator Model", expanded=False):
    st.caption("This model is used to generate question datasets from your documents.")

    with st.container(border=True):
        if config.generator_model:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{config.generator_model.name}**")
                st.caption(
                    f"{config.generator_model.model_type.value} | {config.generator_model.endpoint[:50]}..."
                )
            with col2:
                if st.button("✏️ Edit", key="edit_generator"):
                    st.session_state.editing_generator = True
                    st.rerun()
            with col3:
                if st.button("🗑️ Remove", key="remove_generator"):
                    config.generator_model = None
                    update_config(config)
                    st.rerun()
        else:
            st.info("No generator model configured. Add one below.")

    # Add/Edit generator model form
    show_generator_form = config.generator_model is None or st.session_state.get(
        "editing_generator", False
    )

    if show_generator_form:
        with st.form("generator_model_form"):
            st.markdown("**Configure Generator Model**")

            gen_name = st.text_input(
                "Model Name",
                value=config.generator_model.name if config.generator_model else "Generator-GPT-4",
                max_chars=50,
            )
            gen_type = st.selectbox(
                "Model Type",
                options=[mt.value for mt in ModelType],
                index=0,
            )
            gen_endpoint = st.text_input(
                "Endpoint URL",
                value=config.generator_model.endpoint if config.generator_model else "",
                placeholder="https://your-resource.openai.azure.com/",
            )
            gen_api_key = st.text_input(
                "API Key",
                type="password",
                value="",
                placeholder="Enter API key (leave blank to keep existing)",
            )
            gen_deployment = st.text_input(
                "Deployment Name",
                value=config.generator_model.deployment_name if config.generator_model else "gpt-4",
            )
            gen_api_version = st.text_input(
                "API Version",
                value=config.generator_model.api_version
                if config.generator_model
                else "2024-02-15-preview",
            )
            # Load pricing keys for dropdown selection
            pricing_keys = [""] + list(load_azure_openai_pricing().keys())
            current_gen_pricing = (
                config.generator_model.pricing_key if config.generator_model else ""
            )
            gen_pricing_index = (
                pricing_keys.index(current_gen_pricing)
                if current_gen_pricing in pricing_keys
                else 0
            )
            gen_pricing_key = st.selectbox(
                "Pricing Key",
                options=pricing_keys,
                index=gen_pricing_index,
                help="Select the pricing tier for cost calculation",
                format_func=lambda x: "(Select pricing key)" if x == "" else x,
            )

            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("💾 Save Generator Model", type="primary")
            with col2:
                cancelled = st.form_submit_button("Cancel")

            if submitted:
                try:
                    # Use existing key if not provided
                    api_key = gen_api_key
                    if not api_key and config.generator_model:
                        api_key = config.generator_model.api_key.get_secret_value()

                    if not api_key:
                        st.error("API key is required")
                    elif not gen_endpoint:
                        st.error("Endpoint URL is required")
                    else:
                        config.generator_model = LLMModel(
                            name=gen_name,
                            endpoint=gen_endpoint,
                            api_key=api_key,
                            model_type=ModelType(gen_type),
                            deployment_name=gen_deployment,
                            api_version=gen_api_version,
                            pricing_key=gen_pricing_key if gen_pricing_key else None,
                        )
                        update_config(config)
                        st.session_state.editing_generator = False
                        st.success("Generator model saved!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to save model: {e}")

            if cancelled:
                st.session_state.editing_generator = False
                st.rerun()

# ============================================================================
# Evaluation Models Configuration
# ============================================================================

with st.expander("📊 Evaluation Models", expanded=False):
    st.caption(
        f"Configure up to 5 models for comparison. Currently: {len(config.evaluation_models)}/5"
    )

    # Display existing models
    for idx, model in enumerate(config.evaluation_models):
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                st.markdown(f"**{model.name}**")
                st.caption(f"{model.model_type.value} | {model.endpoint[:40]}...")

            with col2:
                if st.button("🔌 Test", key=f"test_model_{idx}"):
                    with st.spinner(f"Testing {model.name}..."):
                        result = asyncio.run(validate_model_connection(model))
                        if result["ok"]:
                            st.success(f"✅ Connected ({result['latency_ms']:.0f}ms)")
                        else:
                            st.error(f"❌ {result['error'][:50]}")

            with col3:
                if st.button("✏️ Edit", key=f"edit_model_{idx}"):
                    st.session_state.editing_model_index = idx
                    st.rerun()

            with col4:
                if st.button("🗑️", key=f"delete_model_{idx}"):
                    config.remove_evaluation_model(model.name)
                    update_config(config)
                    st.rerun()

    # Add Model button
    if len(config.evaluation_models) < 5:
        if st.button("➕ Add Evaluation Model", type="secondary"):
            st.session_state.show_add_model_form = True
            st.session_state.editing_model_index = None
            st.rerun()
    else:
        st.warning("Maximum 5 evaluation models reached. Remove one to add another.")

    # Add/Edit Model Form
    editing_index = st.session_state.get("editing_model_index")
    show_form = st.session_state.get("show_add_model_form", False) or editing_index is not None

    if show_form:
        editing_model = None
        if editing_index is not None and editing_index < len(config.evaluation_models):
            editing_model = config.evaluation_models[editing_index]

        with st.container(border=True):
            st.markdown("### " + ("Edit" if editing_model else "Add") + " Evaluation Model")

            with st.form("eval_model_form"):
                model_name = st.text_input(
                    "Model Name",
                    value=editing_model.name
                    if editing_model
                    else f"Model-{len(config.evaluation_models) + 1}",
                    max_chars=50,
                )
                model_type = st.selectbox(
                    "Model Type",
                    options=[mt.value for mt in ModelType],
                    index=[mt.value for mt in ModelType].index(editing_model.model_type.value)
                    if editing_model
                    else 0,
                )
                model_endpoint = st.text_input(
                    "Endpoint URL",
                    value=editing_model.endpoint if editing_model else "",
                    placeholder="https://your-resource.openai.azure.com/",
                )
                model_api_key = st.text_input(
                    "API Key",
                    type="password",
                    value="",
                    placeholder="Enter API key"
                    + (" (leave blank to keep existing)" if editing_model else ""),
                )
                model_deployment = st.text_input(
                    "Deployment Name",
                    value=editing_model.deployment_name if editing_model else "gpt-4",
                )
                model_api_version = st.text_input(
                    "API Version",
                    value=editing_model.api_version if editing_model else "2024-02-15-preview",
                )
                # Load pricing keys for dropdown selection
                pricing_keys = [""] + list(load_azure_openai_pricing().keys())
                current_model_pricing = (
                    editing_model.pricing_key if editing_model and editing_model.pricing_key else ""
                )
                model_pricing_index = (
                    pricing_keys.index(current_model_pricing)
                    if current_model_pricing in pricing_keys
                    else 0
                )
                model_pricing_key = st.selectbox(
                    "Pricing Key",
                    options=pricing_keys,
                    index=model_pricing_index,
                    help="Select the pricing tier for cost calculation",
                    format_func=lambda x: "(Select pricing key)" if x == "" else x,
                )

                col1, col2 = st.columns(2)
                with col1:
                    submitted = st.form_submit_button("💾 Save Model", type="primary")
                with col2:
                    cancelled = st.form_submit_button("Cancel")

                if submitted:
                    try:
                        # Use existing key if not provided
                        api_key = model_api_key
                        if not api_key and editing_model:
                            api_key = editing_model.api_key.get_secret_value()

                        if not api_key:
                            st.error("API key is required")
                        elif not model_endpoint:
                            st.error("Endpoint URL is required")
                        else:
                            new_model = LLMModel(
                                name=model_name,
                                endpoint=model_endpoint,
                                api_key=api_key,
                                model_type=ModelType(model_type),
                                deployment_name=model_deployment,
                                api_version=model_api_version,
                                pricing_key=model_pricing_key if model_pricing_key else None,
                            )

                            if editing_model:
                                # Update existing model
                                config.evaluation_models[editing_index] = new_model
                                logger.info(f"Updated evaluation model: {model_name}")
                            else:
                                # Add new model
                                if config.add_evaluation_model(new_model):
                                    logger.info(f"Added evaluation model: {model_name}")
                                else:
                                    st.error("Failed to add model (limit reached)")

                            update_config(config)
                            st.session_state.show_add_model_form = False
                            st.session_state.editing_model_index = None
                            st.success("Model saved!")
                            st.rerun()

                    except Exception as e:
                        st.error(f"Failed to save model: {e}")

                if cancelled:
                    st.session_state.show_add_model_form = False
                    st.session_state.editing_model_index = None
                    st.rerun()

# ============================================================================
# Configuration Summary
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    generator_status = "✅" if config.generator_model else "❌"
    st.metric("Generator Model", generator_status)

with col2:
    st.metric("Evaluation Models", f"{len(config.evaluation_models)}/5")

with col3:
    ready = config.generator_model is not None and len(config.evaluation_models) > 0
    st.metric("Ready", "✅" if ready else "❌")

if not ready:
    st.info(
        "💡 **Next Steps**: "
        + ("Configure a generator model. " if not config.generator_model else "")
        + ("Add at least one evaluation model. " if not config.evaluation_models else "")
    )

# ============================================================================
# Section 1️⃣: Document Upload (US2)
# ============================================================================

st.divider()
st.subheader("1️⃣ Document Upload")
st.caption("Upload PDF documents to extract text for question generation.")

with st.container(border=True):
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximum file size per file: {MAX_PDF_SIZE_BYTES / (1024 * 1024):.0f} MB",
        key="pdf_uploader",
    )

    if uploaded_files:
        # Create a fingerprint of uploaded files to detect changes
        current_files_fingerprint = tuple((f.name, len(f.getvalue())) for f in uploaded_files)
        previous_fingerprint = st.session_state.get("uploaded_files_fingerprint")

        # Auto-extract if files changed
        if current_files_fingerprint != previous_fingerprint:
            # Validate file sizes
            valid_files = []
            for uploaded_file in uploaded_files:
                file_size = len(uploaded_file.getvalue())
                if file_size > MAX_PDF_SIZE_BYTES:
                    st.error(
                        f"❌ **{uploaded_file.name}** ({file_size / (1024 * 1024):.1f} MB) exceeds "
                        f"maximum allowed ({MAX_PDF_SIZE_BYTES / (1024 * 1024):.0f} MB). Skipping."
                    )
                else:
                    valid_files.append((uploaded_file, file_size))

            if valid_files:
                extracted_documents = []
                progress_bar = st.progress(0, text="Extracting text from PDFs...")

                for idx, (uploaded_file, file_size) in enumerate(valid_files):
                    progress_bar.progress(
                        (idx) / len(valid_files), text=f"Extracting: {uploaded_file.name}..."
                    )

                    result = extract_text_from_pdf(uploaded_file.getvalue())

                    if result["ok"]:
                        document = GroundingDocument(
                            filename=uploaded_file.name,
                            pages=result["pages"],
                            page_count=result["page_count"],
                            file_size_bytes=file_size,
                        )
                        extracted_documents.append(document.model_dump(mode="json"))

                        logger.info(
                            f"Extracted text from {uploaded_file.name}: "
                            f"{len(result['content'])} chars, {result['page_count']} pages"
                        )
                    else:
                        st.error(
                            f"❌ Extraction failed for {uploaded_file.name}: {result['error']}"
                        )
                        logger.error(
                            f"PDF extraction failed for {uploaded_file.name}: {result['error']}"
                        )

                progress_bar.progress(1.0, text="Extraction complete!")

                if extracted_documents:
                    # Merge all documents into a single grounding document
                    # Concatenate pages arrays from all documents
                    merged_pages: list[str] = []
                    for doc in extracted_documents:
                        merged_pages.extend(doc["pages"])
                    total_pages = sum(doc["page_count"] for doc in extracted_documents)
                    total_size = sum(doc["file_size_bytes"] for doc in extracted_documents)
                    filenames = ", ".join(doc["filename"] for doc in extracted_documents)

                    merged_document = GroundingDocument(
                        filename=filenames
                        if len(extracted_documents) == 1
                        else f"{len(extracted_documents)} files merged",
                        pages=merged_pages,
                        page_count=total_pages,
                        file_size_bytes=total_size,
                    )
                    st.session_state.llm_eval_document = merged_document.model_dump(mode="json")
                    # Store individual documents for reference
                    st.session_state.llm_eval_documents_list = extracted_documents

                    # Calculate total content length for display
                    total_content_len = sum(len(p) for p in merged_pages)
                    st.success(
                        f"✅ Extracted {total_content_len:,} characters "
                        f"from {total_pages} page(s) across {len(extracted_documents)} file(s)"
                    )

            # Store fingerprint to prevent re-extraction on rerun
            st.session_state.uploaded_files_fingerprint = current_files_fingerprint

    # Show current document
    if st.session_state.llm_eval_document:
        doc_data = st.session_state.llm_eval_document
        docs_list = st.session_state.get("llm_eval_documents_list", [])
        st.markdown("### 📋 Current Document(s)")

        # Compute content from pages for display
        doc_content = "\n\n".join(p for p in doc_data.get("pages", []) if p)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Files", len(docs_list) if docs_list else 1)
        with col2:
            st.metric("Total Pages", doc_data["page_count"])
        with col3:
            st.metric("Total Characters", f"{len(doc_content):,}")
        with col4:
            st.metric("Total Size", f"{doc_data['file_size_bytes'] / 1024:.1f} KB")

        # Show individual file details if multiple files
        if docs_list and len(docs_list) > 1:
            with st.expander("📁 Individual Files", expanded=False):
                for idx, doc in enumerate(docs_list):
                    file_chars = sum(len(p) for p in doc.get("pages", []) if p)
                    st.markdown(
                        f"**{idx + 1}. {doc['filename']}** - "
                        f"{doc['page_count']} pages, {file_chars:,} chars"
                    )

        with st.expander("📖 Preview Content", expanded=False):
            preview_length = min(2000, len(doc_content))
            st.text_area(
                "Document Text (preview)",
                value=doc_content[:preview_length]
                + ("..." if len(doc_content) > preview_length else ""),
                height=300,
                disabled=True,
            )

        if st.button("🗑️ Clear Document(s)", key="clear_document"):
            st.session_state.llm_eval_document = None
            st.session_state.llm_eval_documents_list = None
            st.session_state.uploaded_files_fingerprint = None
            st.rerun()

# ============================================================================
# Section 2️⃣: Question Dataset (US2)
# ============================================================================

st.divider()
st.subheader("2️⃣ Question Dataset")
st.caption("Generate multi-turn Q&A conversations from your document.")

with st.container(border=True):
    # Check prerequisites
    has_document = st.session_state.llm_eval_document is not None
    has_generator = config.generator_model is not None

    # Load existing dataset section (always visible)
    available_datasets = list_datasets()
    if available_datasets:
        current_dataset_id = (
            st.session_state.llm_eval_dataset.get("dataset_id", "")
            if st.session_state.llm_eval_dataset
            else None
        )
        try:
            current_index = (
                available_datasets.index(current_dataset_id) if current_dataset_id else 0
            )
        except ValueError:
            current_index = 0

        selected_dataset_id = st.selectbox(
            "Select Dataset",
            options=available_datasets,
            index=current_index,
            key="dataset_selector",
        )

        # Auto-load when selection changes
        if selected_dataset_id and selected_dataset_id != current_dataset_id:
            loaded = load_dataset(selected_dataset_id)
            if loaded:
                st.session_state.llm_eval_dataset = loaded.model_dump(mode="json")
                st.rerun()
            else:
                st.error(f"❌ Failed to load: {selected_dataset_id}")

    if not has_document and not st.session_state.llm_eval_dataset:
        st.warning(
            "⚠️ Upload and extract a document first (Section 1️⃣), or load an existing dataset above."
        )
    elif not has_generator and not st.session_state.llm_eval_dataset:
        st.warning(
            "⚠️ Configure a generator model first (Section 0️⃣), or load an existing dataset above."
        )

    # Show generation form only when both document and generator are available
    if has_document and has_generator:
        st.markdown("### 🎲 Generate New Dataset")
        doc_data = st.session_state.llm_eval_document

        col1, col2 = st.columns(2)
        with col1:
            num_conversations = st.number_input(
                "Number of Conversations",
                min_value=1,
                max_value=20,
                value=5,
                help="How many separate conversations to generate",
            )
        with col2:
            turns_per_conv = st.selectbox(
                "Turns per Conversation",
                options=[1, 2, 3],
                index=2,  # Default to 3
                help="Number of question-answer turns in each conversation",
            )

        if st.button("🎲 Generate Questions", type="primary", key="generate_questions"):
            with st.spinner(
                f"Generating {num_conversations} conversations with {config.generator_model.name}..."
            ):
                document = GroundingDocument.model_validate(doc_data)

                result = asyncio.run(
                    generate_questions(
                        document=document,
                        model=config.generator_model,
                        num_conversations=num_conversations,
                        turns_per_conversation=turns_per_conv,
                    )
                )

                if result["ok"]:
                    dataset = result["dataset"]

                    # Save to disk
                    save_result = save_dataset(dataset)
                    if save_result["ok"]:
                        logger.info(f"Dataset saved: {save_result['path']}")
                    else:
                        st.error(f"❌ Failed to save dataset: {save_result.get('error')}")
                        logger.error(f"Failed to save dataset: {save_result.get('error')}")

                    # Store in session
                    st.session_state.llm_eval_dataset = dataset.model_dump(mode="json")

                    total_turns = sum(len(c.turns) for c in dataset.conversations)
                    save_path = save_result.get("path", "N/A")
                    st.success(
                        f"✅ Generated {len(dataset.conversations)} conversations "
                        f"with {total_turns} total turns "
                        f"in {result['processing_time']:.2f}s\n\n"
                        f"📁 Saved to: `{save_path}`"
                    )

                    # Show warning if pages were skipped
                    if result.get("warning"):
                        st.warning(f"⚠️ {result['warning']}")

                    logger.info(
                        f"Generated dataset {dataset.dataset_id}: "
                        f"{len(dataset.conversations)} conversations, {total_turns} turns"
                    )
                    st.rerun()
                else:
                    st.error(f"❌ Generation failed: {result['error']}")
                    logger.error(f"Question generation failed: {result['error']}")

    # Display current dataset
    if st.session_state.llm_eval_dataset:
        dataset_data = st.session_state.llm_eval_dataset
        conversations = dataset_data.get("conversations", [])
        total_turns = sum(len(c.get("turns", [])) for c in conversations)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset ID", dataset_data.get("dataset_id", "Unknown"))
        with col2:
            st.metric("Conversations", len(conversations))
        with col3:
            st.metric("Total Turns", total_turns)

        # Display conversations
        for idx, conv in enumerate(conversations):
            page_ref = conv.get("page_reference", 0)
            with st.expander(
                f"💬 Conversation {idx + 1}: {conv.get('conversation_id', 'Unknown')} "
                f"(Page {page_ref + 1})"
            ):
                st.caption(f"📄 References page {page_ref + 1} for RAG grounding")
                for turn in conv.get("turns", []):
                    st.markdown(f"**Turn {turn.get('turn_number', '?')}**")
                    st.markdown(f"🙋 **Q:** {turn.get('question', '')}")

        if st.button("🗑️ Clear Dataset", key="clear_dataset"):
            st.session_state.llm_eval_dataset = None
            st.rerun()

# ============================================================================
# Section 3️⃣: Run Evaluation (US3)
# ============================================================================

st.divider()
st.subheader("3️⃣ Run Evaluation")
st.caption("Execute parallel evaluations across all configured models.")

with st.container(border=True):
    # Check prerequisites
    has_dataset = st.session_state.llm_eval_dataset is not None
    has_models = len(config.evaluation_models) > 0

    if not has_dataset:
        st.warning("⚠️ Generate a question dataset first (Section 2️⃣)")
    elif not has_models:
        st.warning("⚠️ Configure at least one evaluation model (Section 0️⃣)")
    else:
        # Display evaluation summary
        dataset_data = st.session_state.llm_eval_dataset
        conversations = dataset_data.get("conversations", [])
        total_turns = sum(len(c.get("turns", [])) for c in conversations)
        total_evals = total_turns * len(config.evaluation_models)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models", len(config.evaluation_models))
        with col2:
            st.metric("Turns", total_turns)
        with col3:
            st.metric("Total Evaluations", total_evals)

        st.markdown("**Models to evaluate:**")
        for model in config.evaluation_models:
            st.markdown(f"- {model.name} ({model.model_type.value})")

        if st.button("🚀 Run Evaluation", type="primary", key="run_evaluation"):
            # Prepare dataset
            dataset = QuestionDataset.model_validate(dataset_data)
            if st.session_state.llm_eval_document:
                dataset.source_document = GroundingDocument.model_validate(
                    st.session_state.llm_eval_document
                )

            # Progress tracking
            progress_bar = st.progress(0, text="Starting evaluation...")
            status_container = st.empty()

            def update_progress(completed: int, total: int):
                pct = completed / total if total > 0 else 0
                progress_bar.progress(pct, text=f"Evaluating... {completed}/{total}")

            # Run parallel evaluations
            with st.spinner(f"Running {total_evals} evaluations..."):
                start_time = time.time()
                results = asyncio.run(
                    run_parallel_evaluations(
                        models=config.evaluation_models,
                        dataset=dataset,
                        max_concurrent=config.settings.max_concurrent_calls,
                        timeout_seconds=config.settings.timeout_seconds,
                        max_retries=config.settings.max_retries,
                        max_history=config.settings.max_conversation_history,
                        system_prompt=getattr(
                            config.settings, "system_prompt", DEFAULT_SYSTEM_PROMPT
                        ),
                        progress_callback=update_progress,
                    )
                )

            progress_bar.progress(1.0, text="Evaluation complete!")

            # Build and save evaluation run
            import uuid

            run_id = f"run-{uuid.uuid4().hex[:8]}"
            eval_run = build_evaluation_run(
                run_id=run_id,
                dataset=dataset,
                models=config.evaluation_models,
                results=results,
            )

            # Calculate RAG metrics using generator model as judge
            if config.generator_model:
                progress_bar.progress(0, text="Calculating RAG metrics...")

                def metrics_progress(completed: int, total: int):
                    pct = completed / total if total > 0 else 0
                    progress_bar.progress(pct, text=f"Scoring... {completed}/{total}")

                eval_run.results = asyncio.run(
                    calculate_all_metrics(
                        results=eval_run.results,
                        judge_model=config.generator_model,
                        pages=dataset.source_document.pages,
                        progress_callback=metrics_progress,
                    )
                )
                progress_bar.progress(1.0, text="Metrics calculated!")

            save_result = save_run(eval_run)
            if save_result["ok"]:
                logger.info(f"Saved evaluation run: {save_result['path']}")
            else:
                logger.error(f"Failed to save run: {save_result.get('error')}")

            st.session_state.llm_eval_run = eval_run.model_dump(mode="json")

            # Display summary
            success_count = sum(1 for r in results if r.get("ok", False))
            st.success(
                f"✅ Completed {success_count}/{len(results)} evaluations | Run ID: `{run_id}`"
            )

            # Results table
            st.markdown("### 📊 Results Preview")

            # Group by model
            model_results: dict = {}
            for r in results:
                model_name = r.get("model_name", "Unknown")
                if model_name not in model_results:
                    model_results[model_name] = {"success": 0, "failed": 0, "total_api_ms": 0}
                if r.get("ok"):
                    model_results[model_name]["success"] += 1
                    # Use api_latency_ms (pure API time) if available
                    api_ms = r.get("api_latency_ms", 0) or r.get("latency_ms", 0)
                    model_results[model_name]["total_api_ms"] += api_ms
                else:
                    model_results[model_name]["failed"] += 1

            # Display summary table
            for model_name, stats in model_results.items():
                total = stats["success"] + stats["failed"]
                avg_latency = (
                    stats["total_api_ms"] / stats["success"] if stats["success"] > 0 else 0
                )
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"**{model_name}**")
                with col2:
                    st.markdown(f"✅ {stats['success']}/{total}")
                with col3:
                    st.markdown(f"⏱️ {avg_latency:.0f}ms avg")
                with col4:
                    if stats["failed"] > 0:
                        st.markdown(f"❌ {stats['failed']} failed")

            st.rerun()

# ============================================================================
# Section 4️⃣: Results Dashboard (US4)
# ============================================================================

st.divider()
st.subheader("4️⃣ Results Dashboard")
st.caption("View comprehensive RAG metrics and model comparisons.")

with st.container(border=True):
    # Load available runs (returns list of run ID strings)
    available_run_ids = list_runs()

    if not available_run_ids:
        st.info("📊 No evaluation runs found. Complete an evaluation in Section 3️⃣ to see results.")
    else:
        # Run selector - use run IDs directly
        selected_run_id = st.selectbox(
            "Select Evaluation Run",
            options=available_run_ids,
        )

        if selected_run_id:
            selected_run = load_run(selected_run_id)

            if selected_run:
                # Display run metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Run ID", selected_run.run_id)
                with col2:
                    st.metric("Status", selected_run.status.value)
                with col3:
                    st.metric("Models", len(selected_run.model_names))
                with col4:
                    success_pct = selected_run.success_rate * 100
                    st.metric("Success Rate", f"{success_pct:.1f}%")

                st.divider()

                # Aggregate metrics by model
                summary = aggregate_run_metrics(selected_run.results)

                if summary:
                    st.markdown("### 📊 Model Comparison")

                    # Create comparison table
                    import plotly.graph_objects as go

                    model_names = list(summary.keys())

                    # Latency comparison chart (convert ms to seconds)
                    latencies = [summary[m].get("avg_latency_ms", 0) / 1000 for m in model_names]

                    fig_latency = go.Figure(
                        data=[
                            go.Bar(
                                x=model_names,
                                y=latencies,
                                marker_color=[
                                    "#1f77b4",
                                    "#ff7f0e",
                                    "#2ca02c",
                                    "#d62728",
                                    "#9467bd",
                                ][: len(model_names)],
                            )
                        ]
                    )
                    fig_latency.update_layout(
                        title="Average Latency (s)",
                        xaxis_title="Model",
                        yaxis_title="Latency (s)",
                        height=350,
                    )
                    st.plotly_chart(fig_latency, config={"responsive": True})

                    # RAG Metrics comparison
                    metrics_data = []
                    cost_warnings = []

                    for model in model_names:
                        model_summary = summary[model]

                        # Get pricing_key from config for this model
                        model_config = config.get_evaluation_model(model)
                        pricing_key = model_config.pricing_key if model_config else None

                        # Calculate cost dynamically
                        total_prompt = model_summary.get("total_prompt_tokens", 0)
                        total_completion = model_summary.get("total_completion_tokens", 0)
                        total_cached = model_summary.get("total_cached_tokens", 0)
                        total_tokens = model_summary.get("total_tokens", 0)

                        cost, warning = calculate_token_cost(
                            pricing_key, total_prompt, total_completion, total_cached
                        )
                        if warning:
                            cost_warnings.append(f"{model}: {warning}")

                        metrics_data.append(
                            {
                                "Model": model,
                                "Count": model_summary.get("count", 0),
                                "Failed": model_summary.get("failed", 0),
                                "Avg Latency (s)": f"{model_summary.get('avg_latency_ms', 0) / 1000:.2f}",
                                "Groundedness": f"{model_summary.get('avg_groundedness'):.2f}"
                                if model_summary.get("avg_groundedness") is not None
                                else "-",
                                "Relevance": f"{model_summary.get('avg_relevance'):.2f}"
                                if model_summary.get("avg_relevance") is not None
                                else "-",
                                "Coherence": f"{model_summary.get('avg_coherence'):.2f}"
                                if model_summary.get("avg_coherence") is not None
                                else "-",
                                "Fluency": f"{model_summary.get('avg_fluency'):.2f}"
                                if model_summary.get("avg_fluency") is not None
                                else "-",
                                "Total Tokens": f"{total_tokens:,}",
                                "Total Cost": format_cost(cost)[0],
                            }
                        )

                    # Show cost warnings if any
                    if cost_warnings:
                        st.warning(
                            "⚠️ Cost calculation warnings:\n"
                            + "\n".join(f"• {w}" for w in cost_warnings)
                        )

                    import pandas as pd

                    df = pd.DataFrame(metrics_data)
                    st.dataframe(df, width="stretch", hide_index=True)

                    # Detailed results
                    st.divider()
                    st.markdown("### 📝 Detailed Results")

                    # Extract unique values for filters
                    all_models = sorted({r.model_name for r in selected_run.results})
                    all_conversations = sorted({r.conversation_id for r in selected_run.results})
                    all_turns = sorted({r.turn_number for r in selected_run.results})

                    # Initialize session state for filters (before widgets)
                    if "detail_model_filter" not in st.session_state:
                        st.session_state.detail_model_filter = all_models
                    if "detail_conv_filter" not in st.session_state:
                        st.session_state.detail_conv_filter = all_conversations
                    if "detail_turn_filter" not in st.session_state:
                        st.session_state.detail_turn_filter = all_turns

                    # Select All button
                    if st.button("✅ Select All", key="select_all_filters"):
                        st.session_state.detail_model_filter = all_models
                        st.session_state.detail_conv_filter = all_conversations
                        st.session_state.detail_turn_filter = all_turns
                        st.rerun()

                    # Filter controls
                    filter_col1, filter_col2, filter_col3 = st.columns(3)
                    with filter_col1:
                        selected_models = st.multiselect(
                            "Filter by Model",
                            options=all_models,
                            key="detail_model_filter",
                        )
                    with filter_col2:
                        selected_conversations = st.multiselect(
                            "Filter by Conversation",
                            options=all_conversations,
                            key="detail_conv_filter",
                        )
                    with filter_col3:
                        selected_turns = st.multiselect(
                            "Filter by Turn",
                            options=all_turns,
                            key="detail_turn_filter",
                        )

                    # Apply filters
                    filtered_results = [
                        r
                        for r in selected_run.results
                        if r.model_name in selected_models
                        and r.conversation_id in selected_conversations
                        and r.turn_number in selected_turns
                    ]

                    st.caption(
                        f"Showing {len(filtered_results)} of {len(selected_run.results)} results"
                    )

                    # Chart visualization of filtered data
                    if filtered_results:
                        import plotly.express as px

                        # Build DataFrame for charts
                        chart_data = []
                        for r in filtered_results:
                            latency_val = r.api_latency_ms if r.api_latency_ms > 0 else r.latency_ms
                            row = {
                                "Turn": r.turn_number,
                                "Conversation": r.conversation_id,
                                "Model": r.model_name,
                                "Latency (s)": latency_val / 1000,
                                "Groundedness": r.metrics.groundedness
                                if r.metrics and r.metrics.groundedness is not None
                                else None,
                                "Relevance": r.metrics.relevance
                                if r.metrics and r.metrics.relevance is not None
                                else None,
                                "Coherence": r.metrics.coherence
                                if r.metrics and r.metrics.coherence is not None
                                else None,
                                "Fluency": r.metrics.fluency
                                if r.metrics and r.metrics.fluency is not None
                                else None,
                            }
                            chart_data.append(row)

                        chart_df = pd.DataFrame(chart_data)

                        # Turn selector
                        available_turns = sorted(chart_df["Turn"].unique())
                        selected_chart_turn = st.selectbox(
                            "Select Turn to Visualize",
                            options=available_turns,
                            key="chart_turn_select",
                        )

                        # Filter by selected turn
                        turn_df = chart_df[chart_df["Turn"] == selected_chart_turn].copy()

                        # Create short conversation labels (C1, C2, etc.)
                        conv_map = {
                            c: f"C{i + 1}"
                            for i, c in enumerate(sorted(turn_df["Conversation"].unique()))
                        }
                        turn_df["Conv"] = turn_df["Conversation"].map(conv_map)

                        # Latency grouped bar chart
                        fig_latency = px.bar(
                            turn_df,
                            x="Conv",
                            y="Latency (s)",
                            color="Model",
                            barmode="group",
                            title=f"Latency by Conversation (Turn {selected_chart_turn})",
                            hover_data=["Conversation"],
                        )
                        fig_latency.update_layout(
                            xaxis_title="Conversation",
                            yaxis_title="Latency (s)",
                            height=350,
                            legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
                        )
                        st.plotly_chart(fig_latency, config={"responsive": True})

                        # RAG Metrics chart (if metrics exist)
                        has_metrics = (
                            chart_df[["Groundedness", "Relevance", "Coherence", "Fluency"]]
                            .notna()
                            .any()
                            .any()
                        )

                        if has_metrics:
                            metric_choice = st.selectbox(
                                "Select Metric to Visualize",
                                options=["Groundedness", "Relevance", "Coherence", "Fluency"],
                                key="metric_line_chart_select",
                            )

                            fig_metrics = px.bar(
                                turn_df,
                                x="Conv",
                                y=metric_choice,
                                color="Model",
                                barmode="group",
                                title=f"{metric_choice} by Conversation (Turn {selected_chart_turn})",
                                hover_data=["Conversation"],
                            )
                            fig_metrics.update_layout(
                                xaxis_title="Conversation",
                                yaxis_title=metric_choice,
                                height=350,
                                yaxis={"range": [0, 5.5]},
                                legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
                            )
                            st.plotly_chart(fig_metrics, config={"responsive": True})

                    st.divider()

                    for result_idx, result in enumerate(filtered_results):
                        status_icon = "✅" if result.success else "❌"
                        with st.expander(
                            f"{status_icon} {result.model_name} | Turn {result.turn_number} | {result.conversation_id}"
                        ):
                            # Show API messages payload (request + response)
                            with st.expander("📨 Messages Payload", expanded=False):
                                if result.messages:
                                    st.json(result.messages)
                                else:
                                    st.warning("No messages payload available")

                            # Calculate per-turn cost
                            model_config = config.get_evaluation_model(result.model_name)
                            pricing_key = model_config.pricing_key if model_config else None
                            turn_cost, _ = calculate_token_cost(
                                pricing_key,
                                result.prompt_tokens,
                                result.completion_tokens,
                                result.cached_tokens,
                            )

                            if result.metrics:
                                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                                with (
                                    col1,
                                    st.container(
                                        key=f"metric_latency_{result.model_name}_{result.conversation_id}_{result.turn_number}_{result_idx}"
                                    ),
                                ):
                                    # Use api_latency_ms (pure API time) if available, else fall back to total
                                    latency_val = (
                                        result.api_latency_ms
                                        if result.api_latency_ms > 0
                                        else result.latency_ms
                                    )
                                    st.metric(
                                        "Latency",
                                        f"{latency_val / 1000:.2f}s",
                                    )
                                with (
                                    col2,
                                    st.container(
                                        key=f"metric_ground_{result.model_name}_{result.conversation_id}_{result.turn_number}_{result_idx}"
                                    ),
                                ):
                                    st.metric(
                                        "Groundedness",
                                        f"{result.metrics.groundedness:.2f}"
                                        if result.metrics.groundedness is not None
                                        else "-",
                                    )
                                with (
                                    col3,
                                    st.container(
                                        key=f"metric_relev_{result.model_name}_{result.conversation_id}_{result.turn_number}_{result_idx}"
                                    ),
                                ):
                                    st.metric(
                                        "Relevance",
                                        f"{result.metrics.relevance:.2f}"
                                        if result.metrics.relevance is not None
                                        else "-",
                                    )
                                with (
                                    col4,
                                    st.container(
                                        key=f"metric_coher_{result.model_name}_{result.conversation_id}_{result.turn_number}_{result_idx}"
                                    ),
                                ):
                                    st.metric(
                                        "Coherence",
                                        f"{result.metrics.coherence:.2f}"
                                        if result.metrics.coherence is not None
                                        else "-",
                                    )
                                with (
                                    col5,
                                    st.container(
                                        key=f"metric_fluen_{result.model_name}_{result.conversation_id}_{result.turn_number}_{result_idx}"
                                    ),
                                ):
                                    st.metric(
                                        "Fluency",
                                        f"{result.metrics.fluency:.2f}"
                                        if result.metrics.fluency is not None
                                        else "-",
                                    )
                                with (
                                    col6,
                                    st.container(
                                        key=f"metric_tokens_{result.model_name}_{result.conversation_id}_{result.turn_number}_{result_idx}"
                                    ),
                                ):
                                    st.metric("Tokens", f"{result.total_tokens:,}")
                                with (
                                    col7,
                                    st.container(
                                        key=f"metric_cost_{result.model_name}_{result.conversation_id}_{result.turn_number}_{result_idx}"
                                    ),
                                ):
                                    cost_val, cost_label = format_cost(turn_cost)
                                    st.metric(cost_label, cost_val)
                            else:
                                # Show latency, tokens, cost even without metrics
                                col1, col2, col3 = st.columns(3)
                                with (
                                    col1,
                                    st.container(
                                        key=f"metric_latency_{result.model_name}_{result.conversation_id}_{result.turn_number}_{result_idx}_solo"
                                    ),
                                ):
                                    # Use api_latency_ms (pure API time) if available
                                    latency_val = (
                                        result.api_latency_ms
                                        if result.api_latency_ms > 0
                                        else result.latency_ms
                                    )
                                    st.metric("Latency", f"{latency_val / 1000:.2f}s")
                                with (
                                    col2,
                                    st.container(
                                        key=f"metric_tokens_{result.model_name}_{result.conversation_id}_{result.turn_number}_{result_idx}_solo"
                                    ),
                                ):
                                    st.metric("Tokens", f"{result.total_tokens:,}")
                                with (
                                    col3,
                                    st.container(
                                        key=f"metric_cost_{result.model_name}_{result.conversation_id}_{result.turn_number}_{result_idx}_solo"
                                    ),
                                ):
                                    cost_val, cost_label = format_cost(turn_cost)
                                    st.metric(cost_label, cost_val)

                            if result.error:
                                st.error(f"Error: {result.error}")
            else:
                st.error(f"Failed to load run: {selected_run_id}")
