import json
import logging
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.append("..")
from helpers.utils import render_sidebar

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Path to pricing data
PRICING_JSON_PATH = Path(__file__).parent.parent / "inputs" / "azure_openai_pricing.json"


@lru_cache(maxsize=1)
def load_azure_openai_pricing() -> dict:
    """Load Azure OpenAI pricing data from JSON file."""
    try:
        with open(PRICING_JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Loaded {len(data)} models from {PRICING_JSON_PATH}")
            return data
    except FileNotFoundError:
        logger.error(f"Pricing file not found: {PRICING_JSON_PATH}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in pricing file: {e}")
        return {}


def main():
    # Render shared sidebar navigation
    render_sidebar()

    # Load pricing data
    azure_openai_pricing = load_azure_openai_pricing()

    # Page Title
    st.title("💰 Azure OpenAI Pricing")
    st.markdown(
        """
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: var(--text-secondary);">
            Official pricing information for Azure OpenAI Service models (per 1M tokens, USD)
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.subheader("Language Model Pricing")
    st.caption("Prices are per 1 million tokens in USD")

    # Group models by category
    categories = ["Flagship", "Reasoning"]

    for category in categories:
        st.markdown(f"#### {category} Models")
        category_models = {
            k: v for k, v in azure_openai_pricing.items() if v["category"] == category
        }

        if category_models:
            # Create DataFrame for display
            data = []
            for model, pricing in category_models.items():
                cached = f"${pricing['cached_input']:.2f}" if pricing["cached_input"] else "N/A"
                data.append(
                    {
                        "Model": model,
                        "Input": f"${pricing['input']:.2f}",
                        "Cached Input": cached,
                        "Output": f"${pricing['output']:.2f}",
                        "Context Window": pricing["context"],
                        "Max Output": pricing["max_output"],
                    }
                )

            df = pd.DataFrame(data)
            st.dataframe(df, hide_index=True, use_container_width=True)

    # Cost Calculator
    st.markdown("---")
    st.subheader("💵 Cost Calculator")

    col1, col2 = st.columns(2)
    with col1:
        selected_models = st.multiselect(
            "Select Models to Compare",
            options=list(azure_openai_pricing.keys()),
            default=[list(azure_openai_pricing.keys())[0]] if azure_openai_pricing else [],
            max_selections=5,
        )
        input_tokens = st.number_input("Input Tokens (thousands)", min_value=0, value=100, step=10)

    with col2:
        output_tokens = st.number_input("Output Tokens (thousands)", min_value=0, value=50, step=10)
        st.write("")  # Spacer for alignment
        st.write("")  # Spacer for alignment
        use_cached = st.checkbox("Use Cached Input", value=False)

    if selected_models:
        # Calculate costs for all selected models
        model_costs = {}
        for model in selected_models:
            pricing = azure_openai_pricing[model]
            input_rate = (
                pricing["cached_input"]
                if use_cached and pricing["cached_input"]
                else pricing["input"]
            )
            output_rate = pricing["output"]
            input_cost = (input_tokens * 1000 / 1_000_000) * input_rate
            output_cost = (output_tokens * 1000 / 1_000_000) * output_rate
            total_cost = input_cost + output_cost
            # Cost per 1K tokens (rate is per 1M, so divide by 1000)
            input_per_1k = input_rate / 1000
            output_per_1k = output_rate / 1000
            model_costs[model] = {
                "input": input_cost,
                "output": output_cost,
                "total": total_cost,
                "input_per_1k": input_per_1k,
                "output_per_1k": output_per_1k,
            }

        # Use first model as baseline for percentage comparisons
        baseline_model = selected_models[0]
        baseline_costs = model_costs[baseline_model]

        for model in selected_models:
            costs = model_costs[model]
            st.markdown(f"**{model}**")

            # Calculate percentage differences from baseline
            if model == baseline_model:
                input_delta = None
                output_delta = None
                total_delta = None
            else:
                input_delta = (
                    f"{((costs['input'] - baseline_costs['input']) / baseline_costs['input'] * 100):+.1f}%"
                    if baseline_costs["input"] > 0
                    else None
                )
                output_delta = (
                    f"{((costs['output'] - baseline_costs['output']) / baseline_costs['output'] * 100):+.1f}%"
                    if baseline_costs["output"] > 0
                    else None
                )
                total_delta = (
                    f"{((costs['total'] - baseline_costs['total']) / baseline_costs['total'] * 100):+.1f}%"
                    if baseline_costs["total"] > 0
                    else None
                )

            metric_cols = st.columns(3)
            # Sanitize model name for key (remove special chars)
            model_key = model.replace(" ", "_").replace("-", "_").replace(".", "_")
            with metric_cols[0], st.container(key=f"pricing_input_{model_key}"):
                st.metric(
                    label=f"Input Cost (${costs['input_per_1k']:.6f}/1K)",
                    value=f"${costs['input']:.3f}",
                    delta=input_delta,
                    delta_color="inverse",
                )
            with metric_cols[1], st.container(key=f"pricing_output_{model_key}"):
                st.metric(
                    label=f"Output Cost (${costs['output_per_1k']:.6f}/1K)",
                    value=f"${costs['output']:.3f}",
                    delta=output_delta,
                    delta_color="inverse",
                )
            with metric_cols[2], st.container(key=f"pricing_total_{model_key}"):
                st.metric(
                    label="Total Cost",
                    value=f"${costs['total']:.3f}",
                    delta=total_delta,
                    delta_color="inverse",
                )

    else:
        st.info("Select at least one model to calculate costs.")

    # Additional Information Section
    st.markdown("---")
    st.subheader("📊 Model Comparison Guide")

    col_left, col_right = st.columns(2)

    with col_left:
        st.info("""
        **Flagship Models** (GPT-4.1, GPT-4o):
        - Best overall performance
        - Multimodal capabilities
        - Up to 1M token context
        - Production workloads
        """)

        st.info("""
        **Mini/Nano Variants**:
        - Excellent price/performance
        - Fast inference
        - Good for high-volume tasks
        - Chat and summarization
        """)

    with col_right:
        st.info("""
        **Reasoning Models** (GPT-5, GPT-5.1, GPT-5.2):
        - Advanced reasoning
        - 128K token max output
        - Math and coding tasks
        - Multi-step analysis
        """)

        st.info("""
        **Codex Variants** (GPT-5.1-codex-mini):
        - Optimized for code generation
        - Large context window
        - Cost-effective reasoning
        - Developer workflows
        """)

    # Footer with helpful links
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; margin-top: 2rem;">
        <p style="color: var(--text-secondary);">
            💡 <strong>Tip:</strong> Use cached input pricing for repeated prompts to reduce costs by up to 75%<br>
            📚 Learn more: <a href="https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/" target="_blank">Azure OpenAI Pricing</a> |
            <a href="https://learn.microsoft.com/azure/ai-services/openai/concepts/models" target="_blank">Model Documentation</a> |
            <a href="https://azure.microsoft.com/en-us/pricing/calculator/" target="_blank">Azure Pricing Calculator</a>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
