"""
Analysis Page - Analīzes lapa
Author: Agate Jarmakoviča
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path

# Import components
sys.path.insert(0, str(Path(__file__).parent.parent))
from components import show_overall_score, show_dimension_scores

st.set_page_config(
    page_title="Analysis - healthdq-ai",
    page_icon="📊",
    layout="wide",
)

# Initialize session state
if "quality_results" not in st.session_state:
    st.session_state.quality_results = None

st.title("📊 Datu Kvalitātes Analīze")

# Check if data is loaded
if "data" not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ Nav ielādētu datu. Lūdzu, ejiet uz Upload lapu.")
    st.stop()

data = st.session_state.data

st.success(f"✅ Dati ielādēti: {data.shape[0]} rindas, {data.shape[1]} kolonnas")

st.markdown("""
**AI aģenti analizē jūsu datus pēc vairākām dimensijām:**
- 🎯 **Precision** - Format consistency, type validation
- ✅ **Completeness** - Missing value detection
- ♻️ **Reusability** - FAIR principles compliance
""")

# Configuration
with st.expander("⚙️ Analīzes Konfigurācija"):
    dimensions = st.multiselect(
        "Kvalitātes dimensijas",
        ["precision", "completeness", "reusability"],
        default=["precision", "completeness", "reusability"],
    )

if st.button("🚀 Sākt Analīzi", type="primary", use_container_width=True):
    with st.spinner("🤖 AI aģenti analizē datus..."):
        try:
            from healthdq.agents.coordinator import CoordinatorAgent
            from healthdq.config import get_config

            config = get_config()
            coordinator = CoordinatorAgent(config)

            # Run async analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                coordinator.analyze(data, dimensions=dimensions)
            )
            loop.close()

            st.session_state.quality_results = results

            st.success("✅ Analīze pabeigta!")

            # Show results
            st.markdown("---")

            # Overall score
            show_overall_score(results.get("overall_score", 0.0))

            st.markdown("---")

            # Dimension scores
            dimension_results = results.get("dimension_results", {})
            show_dimension_scores(dimension_results)

            st.markdown("---")
            st.info("✨ Analīze pabeigta! Ejiet uz 'Review' lapu, lai pārskatītu rezultātus.")

        except Exception as e:
            st.error(f"❌ Kļūda analīzē: {str(e)}")
            st.exception(e)

# Show previous results if available
elif st.session_state.quality_results:
    st.info("ℹ️ Analīze jau ir veikta. Rezultāti ir pieejami.")

    results = st.session_state.quality_results

    # Overall score
    show_overall_score(results.get("overall_score", 0.0))

    st.markdown("---")

    # Dimension scores
    dimension_results = results.get("dimension_results", {})
    show_dimension_scores(dimension_results)

    st.markdown("---")
    st.info("✨ Ejiet uz 'Review' lapu, lai pārskatītu detalizētus rezultātus.")
