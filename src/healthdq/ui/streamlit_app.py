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
import sys
import tempfile
import os
from pathlib import Path

# Add src directory to Python path for local imports
# File is at: .../src/healthdq/ui/streamlit_app.py
# We need to add: .../src/ to sys.path
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Check for healthdq package and ML dependencies
ML_FEATURES_AVAILABLE = True
MISSING_DEPENDENCIES = []

# Check if healthdq package is available
try:
    import healthdq
except ImportError:
    ML_FEATURES_AVAILABLE = False
    MISSING_DEPENDENCIES.append("healthdq package (run: pip install -e .)")

# Check for optional ML dependencies
if ML_FEATURES_AVAILABLE:
    try:
        import langchain
    except ImportError:
        ML_FEATURES_AVAILABLE = False
        MISSING_DEPENDENCIES.append("langchain")

    try:
        import chromadb
    except ImportError:
        ML_FEATURES_AVAILABLE = False
        MISSING_DEPENDENCIES.append("chromadb")

    try:
        import torch
    except ImportError:
        # Torch is optional, warn but don't disable features
        MISSING_DEPENDENCIES.append("torch")

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
if "temp_file_path" not in st.session_state:
    st.session_state.temp_file_path = None
if "file_size_mb" not in st.session_state:
    st.session_state.file_size_mb = 0


def _simulate_analysis(data: pd.DataFrame, dimensions: list) -> dict:
    """
    Simulate quality analysis when ML features are not available.
    Provides basic rule-based analysis without LLM/vector DB.
    """
    import numpy as np

    results = {
        "overall_score": 0.0,
        "dimension_results": {},
        "improvement_plan": {"actions": []},
        "metadata": {
            "total_issues": 0,
            "confidence": 0.7,  # Lower confidence for simulated results
            "mode": "simulated"
        }
    }

    total_score = 0

    # Analyze each dimension with basic rules
    for dimension in dimensions:
        dim_results = {
            "score": 0.0,
            "issues": [],
            "suggestions": []
        }

        if dimension == "completeness":
            # Check for missing values
            missing_count = data.isna().sum()
            missing_pct = (data.isna().sum() / len(data)) * 100

            for col in data.columns:
                if missing_count[col] > 0:
                    severity = "critical" if missing_pct[col] > 50 else "high" if missing_pct[col] > 20 else "medium"
                    dim_results["issues"].append({
                        "type": "missing_values",
                        "column": col,
                        "description": f"Kolonnā '{col}' trūkst {missing_count[col]} vērtības ({missing_pct[col]:.1f}%)",
                        "severity": severity,
                        "count": int(missing_count[col])
                    })

                    results["improvement_plan"]["actions"].append({
                        "column": col,
                        "recommended_action": f"Aizpildīt trūkstošās vērtības kolonnā '{col}'",
                        "severity": severity
                    })

            # Calculate score
            overall_missing_pct = (data.isna().sum().sum() / (len(data) * len(data.columns))) * 100
            dim_results["score"] = max(0, 1 - (overall_missing_pct / 100))

        elif dimension == "precision":
            # Check data types and format consistency
            for col in data.columns:
                if data[col].dtype == 'object':
                    # Check for inconsistent formats
                    unique_count = data[col].nunique()
                    if unique_count > len(data) * 0.5:
                        dim_results["issues"].append({
                            "type": "format_inconsistency",
                            "column": col,
                            "description": f"Kolonnā '{col}' ir daudz unikālu vērtību ({unique_count}), iespējams formāta nekonsekvence",
                            "severity": "medium"
                        })

            # Simple score based on numeric vs object columns
            numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
            total_cols = len(data.columns)
            dim_results["score"] = 0.7 + (numeric_cols / total_cols) * 0.3

        elif dimension == "reusability":
            # Basic FAIR principles check
            issues_count = 0

            # Check column names (should be descriptive)
            for col in data.columns:
                if len(col) < 2 or col.isdigit():
                    dim_results["issues"].append({
                        "type": "poor_naming",
                        "column": col,
                        "description": f"Kolonnai '{col}' ir nepietiekoši aprakstošs nosaukums",
                        "severity": "low"
                    })
                    issues_count += 1

            dim_results["score"] = max(0.5, 1 - (issues_count / len(data.columns)))

        total_score += dim_results["score"]
        results["dimension_results"][dimension] = dim_results
        results["metadata"]["total_issues"] += len(dim_results["issues"])

    # Calculate overall score
    results["overall_score"] = total_score / len(dimensions) if dimensions else 0.5

    return results


def _create_simulated_review_session(data: pd.DataFrame, quality_results: dict, improvement_plan: dict) -> dict:
    """
    Create a simulated review session when healthdq package is not available.
    """
    import uuid
    from datetime import datetime

    actions = improvement_plan.get("actions", [])

    return {
        "session_id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "data_info": {
            "rows": len(data),
            "columns": len(data.columns),
            "missing_values": int(data.isna().sum().sum())
        },
        "quality_summary": quality_results,
        "proposed_changes": actions,
        "mode": "simulated"
    }


def _send_feedback_to_coordinator(change: dict, approved: bool, coordinator=None) -> None:
    """
    Send feedback to coordinator agent for adaptive learning.

    Args:
        change: The change that was approved/rejected
        approved: Whether the change was approved
        coordinator: CoordinatorAgent instance (optional)
    """
    if coordinator is None:
        # Try to import and create coordinator
        try:
            from healthdq.agents.coordinator import CoordinatorAgent
            from healthdq.config import get_config

            config = get_config()
            coordinator = CoordinatorAgent(config)
        except (ImportError, ModuleNotFoundError):
            # Coordinator not available, skip feedback
            return

    # Check if coordinator has learning method
    if hasattr(coordinator, 'learn_from_feedback'):
        try:
            asyncio.run(
                coordinator.learn_from_feedback(change, approved=approved)
            )
        except Exception as e:
            # Silent fail - don't disrupt user experience if feedback fails
            import logging
            logging.warning(f"Failed to send feedback to coordinator: {e}")


def _apply_simulated_transformations(data: pd.DataFrame, approved_changes: list) -> pd.DataFrame:
    """
    Apply basic transformations when healthdq package is not available.
    """
    import numpy as np

    improved_data = data.copy()

    for change in approved_changes:
        if not change.get("approved", False):
            continue

        column = change.get("column")
        action = change.get("recommended_action", "")

        if not column or column not in improved_data.columns:
            continue

        # Basic imputation for missing values
        if "aizpildīt" in action.lower() or "missing" in action.lower():
            if improved_data[column].dtype in ['float64', 'int64']:
                # Fill numeric with median
                improved_data[column].fillna(improved_data[column].median(), inplace=True)
            else:
                # Fill categorical with mode
                mode_value = improved_data[column].mode()
                if len(mode_value) > 0:
                    improved_data[column].fillna(mode_value[0], inplace=True)

    return improved_data


def main():
    """Main application."""

    # Show ML features warning if dependencies are missing
    if not ML_FEATURES_AVAILABLE:
        st.warning(
            f"⚠️ **Demo režīms**: Daži ML funkcionalitāte nav pieejama. "
            f"Trūkstošie paketes: {', '.join(MISSING_DEPENDENCIES)}. "
            f"Lai iespējotu pilnu funkcionalitāti, instalējiet: `pip install -r requirements.txt`"
        )
    elif MISSING_DEPENDENCIES:
        st.info(
            f"ℹ️ Daži neobligātie komponenti nav instalēti: {', '.join(MISSING_DEPENDENCIES)}. "
            f"Pamata funkcionalitāte ir pieejama."
        )

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
            # Check file size
            file_size_bytes = uploaded_file.size
            file_size_mb = file_size_bytes / (1024 * 1024)
            st.session_state.file_size_mb = file_size_mb

            # Load data based on file type
            file_extension = Path(uploaded_file.name).suffix.lower()

            # For large files (>50 MB), use temp file on disk to optimize memory
            use_temp_file = file_size_mb > 50

            if use_temp_file:
                st.info(f"ℹ️ Liels fails ({file_size_mb:.1f} MB) - izmanto optimizētu režīmu ar failu uz diska")

            with st.spinner("Ielādē datus..."):
                if use_temp_file:
                    # Save to temp file for large files
                    temp_dir = tempfile.gettempdir()
                    temp_filename = f"healthdq_upload_{uploaded_file.name}"
                    temp_path = os.path.join(temp_dir, temp_filename)

                    # Write uploaded file to disk
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.session_state.temp_file_path = temp_path

                    # Read from disk
                    if file_extension == ".csv":
                        data = pd.read_csv(temp_path)
                    elif file_extension in [".xlsx", ".xls"]:
                        data = pd.read_excel(temp_path)
                    elif file_extension == ".json":
                        data = pd.read_json(temp_path)
                    elif file_extension == ".parquet":
                        data = pd.read_parquet(temp_path)
                    else:
                        st.error(f"Neatbalstīts formāts: {file_extension}")
                        return

                    st.toast(f"✅ Fails saglabāts uz diska: {temp_path}", icon="💾")
                else:
                    # Load directly into memory for smaller files
                    st.session_state.temp_file_path = None

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
            st.success(f"✅ Dati ielādēti: {data.shape[0]} rindas, {data.shape[1]} kolonnas ({file_size_mb:.1f} MB)")

            # Show data preview with editor option
            st.subheader("📊 Datu Priekšskatījums")

            preview_mode = st.radio(
                "Priekšskatījuma režīms:",
                ["Tikai skatīšanās", "Rediģēšanas režīms"],
                horizontal=True,
                help="Rediģēšanas režīms ļauj veikt ātras izmaiņas tieši tabulā"
            )

            if preview_mode == "Rediģēšanas režīms":
                edited_data = st.data_editor(
                    data.head(20),
                    use_container_width=True,
                    num_rows="dynamic",
                    key="upload_editor"
                )

                if st.button("💾 Saglabāt Izmaiņas", key="save_edits"):
                    # Update full data with edits
                    st.session_state.data.iloc[:20] = edited_data.values
                    st.toast("✅ Izmaiņas saglabātas", icon="💾")
            else:
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
                # Try to use real AI coordinator
                from healthdq.agents.coordinator import CoordinatorAgent
                from healthdq.config import get_config

                config = get_config()
                coordinator = CoordinatorAgent(config)

                # Run async analysis (modern approach with asyncio.run())
                results = asyncio.run(
                    coordinator.analyze(st.session_state.data, dimensions=dimensions)
                )

                st.session_state.quality_results = results
                st.session_state.workflow_stage = "review"
                st.toast("✅ Analīze pabeigta! Pārskatiet rezultātus.", icon="🤖")
                st.rerun()

            except (ImportError, ModuleNotFoundError) as e:
                # Fall back to simulated analysis if imports fail
                st.info("ℹ️ Demo režīms: Izmanto vienkāršu uz noteikumiem balstītu analīzi")
                try:
                    results = _simulate_analysis(st.session_state.data, dimensions)
                    st.session_state.quality_results = results
                    st.session_state.workflow_stage = "review"
                    st.rerun()
                except Exception as sim_error:
                    st.error(f"❌ Kļūda simulētajā analīzē: {str(sim_error)}")
                    st.exception(sim_error)

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
                try:
                    from healthdq.hitl.review import DataReview
                    reviewer = DataReview()
                    review_session = reviewer.create_review_session(
                        st.session_state.data,
                        results,
                        improvement_plan
                    )
                except (ImportError, ModuleNotFoundError):
                    # Use simulated review session if import fails
                    review_session = _create_simulated_review_session(
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
    """Stage 4: Human approval with auto-approve support."""
    st.header("4️⃣ Izmaiņu Apstiprināšana")

    if st.session_state.review_session is None:
        st.warning("⚠️ Nav review sesijas.")
        return

    review = st.session_state.review_session
    proposed_changes = review["proposed_changes"]

    # Get auto-approve threshold
    auto_approve_threshold = st.session_state.get("auto_approve_threshold", 0.95)

    st.markdown(f"""
    **Lūdzu, pārskatiet un apstipriniet vai noraidiet katru ieteikto izmaiņu.**

    Jūsu feedback palīdz sistēmai mācīties un uzlaboties!

    ℹ️ **Auto-approve slieksnis:** {auto_approve_threshold * 100:.0f}% - Izmaiņas ar augstāku pārliecību tiks automātiski apstiprinātas.
    """)

    # Try to use full approval system, fall back to demo mode if needed
    use_demo_mode = False

    try:
        from healthdq.hitl.approval import ApprovalManager
        # If import succeeds, check if we should use it
        if "approval_manager" not in st.session_state:
            st.session_state.approval_manager = ApprovalManager()
            st.session_state.approval_request = st.session_state.approval_manager.create_approval_request(
                review["session_id"],
                proposed_changes
            )
    except (ImportError, ModuleNotFoundError):
        use_demo_mode = True

    # Handle demo mode with simplified approval
    if use_demo_mode:
        st.info("ℹ️ Demo režīmā: Vienkāršots apstiprināšanas process")

        # Initialize simple approval tracking in session state
        if "simple_approvals" not in st.session_state:
            st.session_state.simple_approvals = {i: None for i in range(len(proposed_changes))}

            # Auto-approve changes with high confidence
            auto_approved_count = 0
            for i, change in enumerate(proposed_changes):
                confidence = change.get("estimated_confidence", 0.5)
                if confidence >= auto_approve_threshold:
                    st.session_state.simple_approvals[i] = True
                    auto_approved_count += 1

            if auto_approved_count > 0:
                st.toast(f"✅ {auto_approved_count} izmaiņas automātiski apstiprinātas (pārliecība ≥ {auto_approve_threshold * 100:.0f}%)", icon="✅")

        # Show each change for simple approval
        for i, change in enumerate(proposed_changes):
            with st.container():
                confidence = change.get("estimated_confidence", 0.5)
                was_auto_approved = confidence >= auto_approve_threshold

                st.markdown(f"### Izmaiņa #{i + 1}")

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**Kolonna:** {change.get('column', 'N/A')}")
                    st.markdown(f"**Darbība:** {change.get('recommended_action', 'N/A')}")
                    st.markdown(f"**Svarīgums:** {change.get('severity', 'medium')}")
                    st.markdown(f"**Pārliecība:** {confidence * 100:.1f}%")

                    if was_auto_approved and st.session_state.simple_approvals[i] is True:
                        st.info("🤖 Automātiski apstiprināts (augsta pārliecība)")

                with col2:
                    approval_status = st.session_state.simple_approvals[i]

                    if approval_status is None:
                        if st.button(f"✅ Apstiprināt", key=f"approve_{i}"):
                            st.session_state.simple_approvals[i] = True
                            # Send feedback to coordinator for learning
                            _send_feedback_to_coordinator(change, approved=True)
                            st.toast("✅ Izmaiņa apstiprināta", icon="✅")
                            st.rerun()

                        if st.button(f"❌ Noraidīt", key=f"reject_{i}"):
                            st.session_state.simple_approvals[i] = False
                            # Send feedback to coordinator for learning
                            _send_feedback_to_coordinator(change, approved=False)
                            st.toast("❌ Izmaiņa noraidīta", icon="❌")
                            st.rerun()
                    elif approval_status:
                        st.success("✅ Apstiprināts")
                    else:
                        st.error("❌ Noraidīts")

                st.markdown("---")

        # Bulk actions for demo mode
        st.subheader("⚡ Masveida Darbības")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Apstiprināt Visas", use_container_width=True):
                for i in range(len(proposed_changes)):
                    st.session_state.simple_approvals[i] = True
                st.rerun()

        with col2:
            if st.button("❌ Noraidīt Visas", use_container_width=True):
                for i in range(len(proposed_changes)):
                    st.session_state.simple_approvals[i] = False
                st.rerun()

        # Finalize button
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("◀️ Atpakaļ", use_container_width=True):
                st.session_state.workflow_stage = "review"
                st.rerun()

        with col2:
            if st.button("🔄 Turpināt uz Transformāciju", type="primary", use_container_width=True):
                # Mark approved changes
                for i, change in enumerate(proposed_changes):
                    change["approved"] = st.session_state.simple_approvals.get(i, False)

                st.session_state.workflow_stage = "transformation"
                st.rerun()

        return

    # Full mode with ApprovalManager (already imported and initialized above)
    approval_manager = st.session_state.approval_manager
    approval_request = st.session_state.approval_request

    # Auto-approve high-confidence changes (run once on initialization)
    if "auto_approve_applied" not in st.session_state:
        st.session_state.auto_approve_applied = True
        auto_approved_count = 0

        for i, change in enumerate(proposed_changes):
            confidence = change.get("estimated_confidence", 0.5)
            if confidence >= auto_approve_threshold:
                approval_manager.approve_change(approval_request["approval_id"], i)
                auto_approved_count += 1

        if auto_approved_count > 0:
            st.toast(f"✅ {auto_approved_count} izmaiņas automātiski apstiprinātas (pārliecība ≥ {auto_approve_threshold * 100:.0f}%)", icon="✅")

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

                confidence = change.get("estimated_confidence", 0.5)
                st.markdown(f"**Pārliecība:** {confidence * 100:.1f}%")

                if confidence >= auto_approve_threshold and approval_request["change_approvals"][i]["approved"]:
                    st.info("🤖 Automātiski apstiprināts (augsta pārliecība)")

            with col2:
                approval_status = approval_request["change_approvals"][i]["approved"]

                if approval_status is None:
                    if st.button(f"✅ Apstiprināt", key=f"approve_{i}"):
                        approval_manager.approve_change(approval_request["approval_id"], i)
                        # Send feedback to coordinator for learning
                        _send_feedback_to_coordinator(change, approved=True)
                        st.toast("✅ Izmaiņa apstiprināta", icon="✅")
                        st.rerun()

                    if st.button(f"❌ Noraidīt", key=f"reject_{i}"):
                        approval_manager.reject_change(
                            approval_request["approval_id"],
                            i,
                            "Noraidīts lietotāja"
                        )
                        # Send feedback to coordinator for learning
                        _send_feedback_to_coordinator(change, approved=False)
                        st.toast("❌ Izmaiņa noraidīta", icon="❌")
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
            st.toast(f"✅ Visas {len(proposed_changes)} izmaiņas apstiprinātas", icon="✅")
            st.rerun()

    with col2:
        if st.button("❌ Noraidīt Visas", use_container_width=True):
            approval_manager.bulk_reject(
                approval_request["approval_id"],
                list(range(len(proposed_changes))),
                "Masveida noraidīšana"
            )
            st.toast(f"❌ Visas {len(proposed_changes)} izmaiņas noraidītas", icon="❌")
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

    # Determine what data we have to work with
    # Try full ML mode first, fall back to demo mode
    use_demo_mode = False
    approved_changes = []

    if st.session_state.final_decision is not None:
        # Full mode data available
        final_decision = st.session_state.final_decision
        approved_changes = final_decision["approved_changes"]
    elif st.session_state.review_session is not None:
        # Demo mode data available
        use_demo_mode = True
        review = st.session_state.review_session
        proposed_changes = review["proposed_changes"]
        approved_changes = [change for change in proposed_changes if change.get("approved", False)]
    else:
        st.warning("⚠️ Nav apstiprinātu izmaiņu.")
        return

    st.info(f"ℹ️ Piemēro {len(approved_changes)} apstiprinātās izmaiņas...")

    with st.spinner("🔄 Transformē datus..."):
        try:
            # Try to use full DataTransformer
            from healthdq.rules.transform import DataTransformer
            from healthdq.utils.helpers import normalize_column_names

            transformer = DataTransformer()
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
                    improved_data = normalize_column_names(improved_data)

            st.session_state.improved_data = improved_data
            st.success("✅ Transformācija pabeigta!")
            st.toast("🎉 Datu kvalitāte uzlabota! Skatiet rezultātus.", icon="✨")
            st.session_state.workflow_stage = "results"
            st.rerun()

        except (ImportError, ModuleNotFoundError) as e:
            # Fall back to simulated transformation
            st.info("ℹ️ Demo režīms: Izmanto vienkāršu transformāciju")
            try:
                if use_demo_mode and st.session_state.review_session is not None:
                    proposed_changes = st.session_state.review_session["proposed_changes"]
                    improved_data = _apply_simulated_transformations(
                        st.session_state.data,
                        proposed_changes
                    )
                else:
                    # Simple fallback for non-demo mode data
                    improved_data = st.session_state.data.copy()

                st.session_state.improved_data = improved_data
                st.success("✅ Transformācija pabeigta!")
                st.session_state.workflow_stage = "results"
                st.rerun()

            except Exception as sim_error:
                st.error(f"❌ Kļūda simulētajā transformācijā: {str(sim_error)}")
                st.exception(sim_error)

        except Exception as e:
            st.error(f"❌ Kļūda transformācijā: {str(e)}")
            st.exception(e)


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

    # Data preview with editor
    st.subheader("👀 Uzlaboto Datu Priekšskatījums")

    result_preview_mode = st.radio(
        "Priekšskatījuma režīms:",
        ["Tikai skatīšanās", "Rediģēšanas režīms"],
        horizontal=True,
        key="result_preview_mode",
        help="Rediģēšanas režīms ļauj veikt finālās korekcijas"
    )

    if result_preview_mode == "Rediģēšanas režīms":
        edited_improved = st.data_editor(
            improved_data.head(20),
            use_container_width=True,
            num_rows="dynamic",
            key="result_editor"
        )

        if st.button("💾 Saglabāt Finālās Izmaiņas", key="save_final_edits"):
            # Update improved data with final edits
            st.session_state.improved_data.iloc[:20] = edited_improved.values
            st.toast("✅ Finālās izmaiņas saglabātas", icon="💾")
    else:
        st.dataframe(improved_data.head(10), use_container_width=True)

    # Download
    st.subheader("💾 Lejupielāde")

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df_to_csv(improved_data)

    if st.download_button(
        label="📥 Lejupielādēt CSV",
        data=csv,
        file_name="improved_data.csv",
        mime="text/csv",
        use_container_width=True,
    ):
        st.toast("📥 Uzlabotie dati lejupielādēti!", icon="💾")

    # Start over
    if st.button("🔄 Sākt No Jauna", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()
