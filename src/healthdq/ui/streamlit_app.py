"""
Streamlit Web UI for healthdq-ai with Human-in-the-Loop
Author: Agate Jarmakoviča

Interaktīvs interfeiss ar pilnu HITL workflow:
1. Datu augšupielāde
2. Sākotnējais novērtējums (PIRMS izmaiņām)
3. Cilvēka apstiprināšana
4. Datu transformācija
5. Rezultātu pārskats
"""

import streamlit as st
import pandas as pd
import asyncio
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="healthdq-ai - Data Quality Framework",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "quality_results" not in st.session_state:
    st.session_state.quality_results = None
if "review_session" not in st.session_state:
    st.session_state.review_session = None
if "approval_request" not in st.session_state:
    st.session_state.approval_request = None
if "improved_data" not in st.session_state:
    st.session_state.improved_data = None
if "workflow_stage" not in st.session_state:
    st.session_state.workflow_stage = "upload"


def main():
    """Main application."""

    # Sidebar
    with st.sidebar:
        st.title("🏥 healthdq-ai")
        st.markdown("**AI Agent-Based Data Quality Framework**")
        st.markdown("---")

        # Workflow stages
        st.subheader("📋 Workflow")

        stages = {
            "upload": "1️⃣ Augšupielāde",
            "assessment": "2️⃣ Novērtējums",
            "review": "3️⃣ Pārskatīšana",
            "approval": "4️⃣ Apstiprināšana",
            "transformation": "5️⃣ Transformācija",
            "results": "6️⃣ Rezultāti",
        }

        current_stage = st.session_state.workflow_stage

        for stage_id, stage_name in stages.items():
            if stage_id == current_stage:
                st.markdown(f"**→ {stage_name}** ✓")
            else:
                st.markdown(f"   {stage_name}")

        st.markdown("---")

        # Settings
        with st.expander("⚙️ Iestatījumi"):
            st.session_state.auto_approve_threshold = st.slider(
                "Auto-approve slieksnis",
                0.0, 1.0, 0.95,
                help="AI automātiski apstiprina izmaiņas ar šo vai augstāku pārliecību"
            )

            st.session_state.require_all_approvals = st.checkbox(
                "Prasīt katras izmaiņas apstiprināšanu",
                value=False,
                help="Ja ieslēgts, katrai izmaiņai jāapstiprina atsevišķi"
            )

        st.markdown("---")

        # About
        with st.expander("ℹ️ Par"):
            st.markdown("""
            **healthdq-ai v2.0**

            Multi-agent data quality framework ar Human-in-the-Loop.

            - 🤖 AI aģenti
            - 🔄 HITL validācija
            - 📊 FAIR principi
            - 🧠 Adaptive learning
            """)

    # Main content
    st.title("🏥 healthdq-ai: Data Quality Framework")

    # Display current stage
    if current_stage == "upload":
        show_upload_stage()
    elif current_stage == "assessment":
        show_assessment_stage()
    elif current_stage == "review":
        show_review_stage()
    elif current_stage == "approval":
        show_approval_stage()
    elif current_stage == "transformation":
        show_transformation_stage()
    elif current_stage == "results":
        show_results_stage()


def show_upload_stage():
    """Stage 1: Data upload."""
    st.header("1️⃣ Datu Augšupielāde")

    st.markdown("""
    Augšupielādējiet savu datu kopu analīzei un kvalitātes uzlabošanai.

    **Atbalstītie formāti:** CSV, Excel, JSON, Parquet
    """)

    uploaded_file = st.file_uploader(
        "Izvēlieties failu",
        type=["csv", "xlsx", "xls", "json", "parquet"],
    )

    if uploaded_file:
        try:
            # Load data based on file type
            file_extension = Path(uploaded_file.name).suffix.lower()

            with st.spinner("Ielādē datus..."):
                if file_extension == ".csv":
                    data = pd.read_csv(uploaded_file)
                elif file_extension in [".xlsx", ".xls"]:
                    data = pd.read_excel(uploaded_file)
                elif file_extension == ".json":
                    data = pd.read_json(uploaded_file)
                elif file_extension == ".parquet":
                    data = pd.read_parquet(uploaded_file)
                else:
                    st.error(f"Neatbalstīts formāts: {file_extension}")
                    return

            st.session_state.data = data
            st.success(f"✅ Dati ielādēti: {data.shape[0]} rindas, {data.shape[1]} kolonnas")

            # Show data preview
            st.subheader("📊 Datu Priekšskatījums")
            st.dataframe(data.head(10), use_container_width=True)

            # Show basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rindas", data.shape[0])
            with col2:
                st.metric("Kolonnas", data.shape[1])
            with col3:
                missing_pct = (data.isna().sum().sum() / (data.shape[0] * data.shape[1])) * 100
                st.metric("Trūkstoši %", f"{missing_pct:.1f}%")
            with col4:
                duplicates = data.duplicated().sum()
                st.metric("Dublikāti", duplicates)

            # Next button
            if st.button("▶️ Turpināt uz Novērtējumu", type="primary", use_container_width=True):
                st.session_state.workflow_stage = "assessment"
                st.rerun()

        except Exception as e:
            st.error(f"❌ Kļūda ielādējot datus: {str(e)}")


def show_assessment_stage():
    """Stage 2: Initial assessment."""
    st.header("2️⃣ Datu Kvalitātes Novērtējums")

    if st.session_state.data is None:
        st.warning("⚠️ Nav ielādētu datu. Atgriezieties uz augšupielādes posmu.")
        return

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

    if st.button("🚀 Sākt Analīzi", type="primary"):
        with st.spinner("🤖 AI aģenti analizē datus..."):
            # Run analysis
            try:
                from healthdq.agents.coordinator import CoordinatorAgent
                from healthdq.config import get_config

                config = get_config()
                coordinator = CoordinatorAgent(config)

                # Run async analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(
                    coordinator.analyze(st.session_state.data, dimensions=dimensions)
                )
                loop.close()

                st.session_state.quality_results = results
                st.session_state.workflow_stage = "review"
                st.rerun()

            except Exception as e:
                st.error(f"❌ Kļūda analīzē: {str(e)}")
                st.exception(e)


def show_review_stage():
    """Stage 3: Review results BEFORE changes."""
    st.header("3️⃣ Rezultātu Pārskatīšana")

    if st.session_state.quality_results is None:
        st.warning("⚠️ Nav analīzes rezultātu.")
        return

    results = st.session_state.quality_results

    # Overall score
    st.subheader("📊 Kopējais Kvalitātes Vērtējums")

    overall_score = results.get("overall_score", 0.0)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Score gauge
        score_pct = overall_score * 100
        score_color = "🟢" if score_pct >= 80 else "🟡" if score_pct >= 60 else "🔴"

        st.metric(
            "Kvalitātes Rezultāts",
            f"{score_pct:.1f}%",
            help="Kopējais datu kvalitātes vērtējums"
        )

        st.progress(overall_score)

    with col2:
        total_issues = results.get("metadata", {}).get("total_issues", 0)
        st.metric("Konstatētas Problēmas", total_issues)

    with col3:
        confidence = results.get("metadata", {}).get("confidence", 0.0)
        st.metric("AI Pārliecība", f"{confidence * 100:.0f}%")

    st.markdown("---")

    # Dimension scores
    st.subheader("📈 Kvalitātes Dimensijas")

    dimension_results = results.get("dimension_results", {})

    cols = st.columns(len(dimension_results))

    for i, (dimension, dim_results) in enumerate(dimension_results.items()):
        with cols[i]:
            score = dim_results.get("score", 0.0)
            issues_count = len(dim_results.get("issues", []))

            st.metric(
                dimension.title(),
                f"{score * 100:.1f}%",
                f"{issues_count} problēmas"
            )

            st.progress(score)

    st.markdown("---")

    # Issues by dimension
    st.subheader("🔍 Konstatētās Problēmas")

    for dimension, dim_results in dimension_results.items():
        issues = dim_results.get("issues", [])

        if issues:
            with st.expander(f"**{dimension.title()}** ({len(issues)} problēmas)"):
                for issue in issues:
                    severity = issue.get("severity", "medium")
                    severity_emoji = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "⚪"
                    }.get(severity, "⚪")

                    st.markdown(f"{severity_emoji} **{issue.get('type')}** - {issue.get('description')}")
                    if "column" in issue:
                        st.markdown(f"   📍 Kolonna: `{issue['column']}`")

    st.markdown("---")

    # Proposed changes
    st.subheader("💡 Ieteiktie Uzlabojumi")

    improvement_plan = results.get("improvement_plan", {})
    actions = improvement_plan.get("actions", [])

    if actions:
        st.info(f"ℹ️ Sistēma iesaka **{len(actions)}** izmaiņas datu kvalitātes uzlabošanai.")

        # Show actions as table
        actions_df = pd.DataFrame([
            {
                "Nr.": i + 1,
                "Darbība": action.get("recommended_action", "N/A"),
                "Kolonna": action.get("column", "N/A"),
                "Svarīgums": action.get("severity", "medium"),
            }
            for i, action in enumerate(actions)
        ])

        st.dataframe(actions_df, use_container_width=True, hide_index=True)

        # Next button
        col1, col2 = st.columns(2)

        with col1:
            if st.button("◀️ Atpakaļ", use_container_width=True):
                st.session_state.workflow_stage = "assessment"
                st.rerun()

        with col2:
            if st.button("✅ Pārskatīt un Apstiprināt", type="primary", use_container_width=True):
                # Create review session
                from healthdq.hitl.review import DataReview

                reviewer = DataReview()
                review_session = reviewer.create_review_session(
                    st.session_state.data,
                    results,
                    improvement_plan
                )

                st.session_state.review_session = review_session
                st.session_state.workflow_stage = "approval"
                st.rerun()
    else:
        st.success("✅ Nav konstatētas problēmas! Jūsu dati ir augstas kvalitātes.")


def show_approval_stage():
    """Stage 4: Human approval."""
    st.header("4️⃣ Izmaiņu Apstiprināšana")

    if st.session_state.review_session is None:
        st.warning("⚠️ Nav review sesijas.")
        return

    review = st.session_state.review_session
    proposed_changes = review["proposed_changes"]

    st.markdown("""
    **Lūdzu, pārskatiet un apstipriniet vai noraidiet katru ieteikto izmaiņu.**

    Jūsu feedback palīdz sistēmai mācīties un uzlaboties!
    """)

    # Create approval manager
    from healthdq.hitl.approval import ApprovalManager

    if "approval_manager" not in st.session_state:
        st.session_state.approval_manager = ApprovalManager()
        st.session_state.approval_request = st.session_state.approval_manager.create_approval_request(
            review["session_id"],
            proposed_changes
        )

    approval_manager = st.session_state.approval_manager
    approval_request = st.session_state.approval_request

    # Show each change for approval
    for i, change in enumerate(proposed_changes):
        with st.container():
            st.markdown(f"### Izmaiņa #{i + 1}")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Darbība:** {change['action_type']}")
                st.markdown(f"**Mērķis:** {change['target']}")
                st.markdown(f"**Apraksts:** {change['description']}")
                st.markdown(f"**Svarīgums:** {change['severity']}")
                st.markdown(f"**Ietekme:** {change['estimated_impact']}")

            with col2:
                approval_status = approval_request["change_approvals"][i]["approved"]

                if approval_status is None:
                    if st.button(f"✅ Apstiprināt", key=f"approve_{i}"):
                        approval_manager.approve_change(approval_request["approval_id"], i)
                        st.rerun()

                    if st.button(f"❌ Noraidīt", key=f"reject_{i}"):
                        approval_manager.reject_change(
                            approval_request["approval_id"],
                            i,
                            "Noraidīts lietotāja"
                        )
                        st.rerun()
                elif approval_status:
                    st.success("✅ Apstiprināts")
                else:
                    st.error("❌ Noraidīts")

            st.markdown("---")

    # Bulk actions
    st.subheader("⚡ Masveida Darbības")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Apstiprināt Visas", use_container_width=True):
            approval_manager.bulk_approve(
                approval_request["approval_id"],
                list(range(len(proposed_changes)))
            )
            st.rerun()

    with col2:
        if st.button("❌ Noraidīt Visas", use_container_width=True):
            approval_manager.bulk_reject(
                approval_request["approval_id"],
                list(range(len(proposed_changes))),
                "Masveida noraidīšana"
            )
            st.rerun()

    # Finalize
    st.markdown("---")

    if st.button("🚀 Pabeigt un Piemērot Izmaiņas", type="primary", use_container_width=True):
        # Finalize approval
        final_decision = approval_manager.finalize_approval(
            approval_request["approval_id"],
            reviewer_info={"user": "streamlit_user"},
            decision_rationale="Reviewed in Streamlit UI"
        )

        st.session_state.final_decision = final_decision
        st.session_state.workflow_stage = "transformation"
        st.rerun()


def show_transformation_stage():
    """Stage 5: Apply approved transformations."""
    st.header("5️⃣ Datu Transformācija")

    if st.session_state.final_decision is None:
        st.warning("⚠️ Nav apstiprinātu izmaiņu.")
        return

    final_decision = st.session_state.final_decision
    approved_changes = final_decision["approved_changes"]

    st.info(f"ℹ️ Piemēro {len(approved_changes)} apstiprinātās izmaiņas...")

    with st.spinner("🔄 Transformē datus..."):
        try:
            from healthdq.rules.transform import DataTransformer

            transformer = DataTransformer()

            # Apply transformations
            improved_data = st.session_state.data.copy()

            for change in approved_changes:
                action_type = change.get("action_type")
                target = change.get("target")

                if action_type == "impute_missing_values":
                    improved_data = transformer.impute_missing(improved_data, column=target)
                elif action_type == "handle_outliers":
                    improved_data = transformer.handle_outliers(improved_data, column=target)
                elif action_type == "standardize_data_types":
                    improved_data = transformer.standardize_types(improved_data, column=target)
                elif action_type == "normalize_column_names":
                    from healthdq.utils.helpers import normalize_column_names
                    improved_data = normalize_column_names(improved_data)

            st.session_state.improved_data = improved_data
            st.success("✅ Transformācija pabeigta!")

            st.session_state.workflow_stage = "results"
            st.rerun()

        except Exception as e:
            st.error(f"❌ Kļūda transformācijā: {str(e)}")


def show_results_stage():
    """Stage 6: Show results and comparison."""
    st.header("6️⃣ Rezultāti")

    if st.session_state.improved_data is None:
        st.warning("⚠️ Nav uzlabotu datu.")
        return

    original_data = st.session_state.data
    improved_data = st.session_state.improved_data

    st.success("✅ Datu kvalitāte uzlabota!")

    # Comparison
    st.subheader("📊 Salīdzinājums: Pirms ↔️ Pēc")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Pirms")
        st.metric("Rindas", original_data.shape[0])
        st.metric("Kolonnas", original_data.shape[1])
        missing_before = original_data.isna().sum().sum()
        st.metric("Trūkstošas vērtības", missing_before)

    with col2:
        st.markdown("### ✨ Pēc")
        st.metric("Rindas", improved_data.shape[0])
        st.metric("Kolonnas", improved_data.shape[1])
        missing_after = improved_data.isna().sum().sum()
        filled = missing_before - missing_after
        st.metric("Trūkstošas vērtības", missing_after, f"-{filled}")

    # Data preview
    st.subheader("👀 Uzlaboto Datu Priekšskatījums")
    st.dataframe(improved_data.head(10), use_container_width=True)

    # Download
    st.subheader("💾 Lejupielāde")

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df_to_csv(improved_data)

    st.download_button(
        label="📥 Lejupielādēt CSV",
        data=csv,
        file_name="improved_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Start over
    if st.button("🔄 Sākt No Jauna", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()
