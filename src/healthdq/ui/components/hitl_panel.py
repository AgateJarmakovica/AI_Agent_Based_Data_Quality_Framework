"""
HITL Panel Component - Human-in-the-Loop panelis
Author: Agate Jarmakoviča
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Callable


def show_approval_request(
    change: Dict[str, Any],
    change_index: int,
    on_approve: Optional[Callable] = None,
    on_reject: Optional[Callable] = None,
) -> Optional[str]:
    """
    Parāda apstiprināšanas pieprasījumu vienai izmaiņai.

    Args:
        change: Izmaiņas informācija
        change_index: Izmaiņas indekss
        on_approve: Callback kad apstiprina
        on_reject: Callback kad noraida

    Returns:
        "approved", "rejected", vai None
    """
    with st.container():
        st.markdown(f"### Izmaiņa #{change_index + 1}")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**Darbība:** {change.get('action_type', 'N/A')}")
            st.markdown(f"**Mērķis:** {change.get('target', 'N/A')}")
            st.markdown(f"**Apraksts:** {change.get('description', 'N/A')}")

            severity = change.get('severity', 'medium')
            severity_emoji = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '⚪'
            }.get(severity, '⚪')

            st.markdown(f"**Svarīgums:** {severity_emoji} {severity}")
            st.markdown(f"**Ietekme:** {change.get('estimated_impact', 'Nezināma')}")

        with col2:
            approve_btn = st.button(
                "✅ Apstiprināt",
                key=f"approve_{change_index}",
                use_container_width=True
            )

            reject_btn = st.button(
                "❌ Noraidīt",
                key=f"reject_{change_index}",
                use_container_width=True
            )

            if approve_btn:
                if on_approve:
                    on_approve(change_index)
                return "approved"

            if reject_btn:
                if on_reject:
                    on_reject(change_index)
                return "rejected"

        st.markdown("---")

    return None


def show_bulk_actions(
    total_changes: int,
    on_approve_all: Optional[Callable] = None,
    on_reject_all: Optional[Callable] = None,
) -> Optional[str]:
    """
    Parāda masveida apstiprināšanas pogas.

    Returns:
        "approve_all", "reject_all", vai None
    """
    st.subheader("⚡ Masveida Darbības")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Apstiprināt Visas", use_container_width=True):
            if on_approve_all:
                on_approve_all()
            return "approve_all"

    with col2:
        if st.button("❌ Noraidīt Visas", use_container_width=True):
            if on_reject_all:
                on_reject_all()
            return "reject_all"

    return None


def show_approval_summary(
    approval_stats: Dict[str, int],
    title: str = "📊 Apstiprināšanas Kopsavilkums"
) -> None:
    """
    Parāda apstiprināšanas statistiku.

    Args:
        approval_stats: Dict ar statistiku
        title: Virsraksts
    """
    st.subheader(title)

    total = approval_stats.get("total", 0)
    approved = approval_stats.get("approved", 0)
    rejected = approval_stats.get("rejected", 0)
    pending = total - approved - rejected

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Kopā", total)

    with col2:
        st.metric("✅ Apstiprināts", approved)

    with col3:
        st.metric("❌ Noraidīts", rejected)

    with col4:
        st.metric("⏳ Gaida", pending)

    # Progress bar
    if total > 0:
        progress = (approved + rejected) / total
        st.progress(progress)
        st.caption(f"Progress: {progress * 100:.0f}%")


def show_feedback_form(
    item_type: str = "change",
    on_submit: Optional[Callable] = None
) -> Optional[Dict[str, Any]]:
    """
    Parāda feedback veidlapu.

    Args:
        item_type: Elementa tips
        on_submit: Callback kad iesniedz

    Returns:
        Feedback data vai None
    """
    st.subheader("💬 Jūsu Feedback")

    with st.form("feedback_form"):
        rating = st.slider(
            "Novērtējums (1-5)",
            min_value=1,
            max_value=5,
            value=3,
            help="Cik apmierināts esat ar šo ieteikumu?"
        )

        confidence = st.slider(
            "Jūsu pārliecība (0-100%)",
            min_value=0,
            max_value=100,
            value=80,
            help="Cik pārliecināts esat par savu lēmumu?"
        )

        comment = st.text_area(
            "Komentārs (neobligāts)",
            placeholder="Paskaidrojiet savu lēmumu...",
            help="Jūsu komentāri palīdz sistēmai uzlaboties"
        )

        submitted = st.form_submit_button("📤 Iesniegt Feedback")

        if submitted:
            feedback_data = {
                "rating": rating,
                "confidence": confidence / 100,
                "comment": comment if comment else None,
            }

            if on_submit:
                on_submit(feedback_data)

            return feedback_data

    return None


def show_change_details(
    change: Dict[str, Any],
    expanded: bool = False
) -> None:
    """
    Parāda detalizētu informāciju par izmaiņu.

    Args:
        change: Izmaiņas informācija
        expanded: Vai rādīt expanded
    """
    with st.expander("🔍 Detalizēta Informācija", expanded=expanded):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Pamata Informācija:**")
            st.markdown(f"- Darbība: `{change.get('action_type')}`")
            st.markdown(f"- Dimensija: `{change.get('dimension', 'N/A')}`")
            st.markdown(f"- Kolonna: `{change.get('target')}`")

        with col2:
            st.markdown("**Ietekmes Analīze:**")
            st.markdown(f"- Svarīgums: `{change.get('severity')}`")
            st.markdown(f"- Ietekme: {change.get('estimated_impact')}")

        if "rationale" in change:
            st.markdown("**Pamatojums:**")
            st.info(change["rationale"])


def show_review_checklist(
    items: List[str],
    title: str = "✅ Pārbaudes Saraksts"
) -> Dict[str, bool]:
    """
    Parāda pārbaudes sarakstu.

    Args:
        items: Saraksts ar pārbaudes punktiem
        title: Virsraksts

    Returns:
        Dict ar checkbox stāvokļiem
    """
    st.subheader(title)

    checklist = {}

    for i, item in enumerate(items):
        checklist[item] = st.checkbox(
            item,
            key=f"checklist_{i}"
        )

    # Check if all checked
    all_checked = all(checklist.values())

    if all_checked:
        st.success("✅ Visi punkti atzīmēti!")
    else:
        unchecked = len([v for v in checklist.values() if not v])
        st.warning(f"⚠️ Vēl {unchecked} punkti nav atzīmēti")

    return checklist


def show_confidence_indicator(
    confidence: float,
    label: str = "AI Pārliecība"
) -> None:
    """
    Parāda pārliecības indikatoru.

    Args:
        confidence: Pārliecība (0.0-1.0)
        label: Label
    """
    confidence_pct = confidence * 100

    # Determine color
    if confidence_pct >= 90:
        color = "🟢"
        level = "Ļoti augsta"
    elif confidence_pct >= 75:
        color = "🟡"
        level = "Augsta"
    elif confidence_pct >= 50:
        color = "🟠"
        level = "Vidēja"
    else:
        color = "🔴"
        level = "Zema"

    col1, col2 = st.columns([3, 1])

    with col1:
        st.metric(label, f"{confidence_pct:.0f}%")
        st.progress(confidence)

    with col2:
        st.markdown(f"### {color}")
        st.caption(level)


__all__ = [
    "show_approval_request",
    "show_bulk_actions",
    "show_approval_summary",
    "show_feedback_form",
    "show_change_details",
    "show_review_checklist",
    "show_confidence_indicator",
]
