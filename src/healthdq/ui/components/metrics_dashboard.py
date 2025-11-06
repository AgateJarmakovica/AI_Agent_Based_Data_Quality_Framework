"""
Metrics Dashboard Component - Metriku panelis
Author: Agate Jarmakoviča
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional


def show_overall_score(
    score: float,
    title: str = "Kopējais Rezultāts"
) -> None:
    """
    Parāda kopējo kvalitātes rezultātu.

    Args:
        score: Rezultāts (0.0-1.0)
        title: Virsraksts
    """
    score_pct = score * 100

    # Determine color
    if score_pct >= 80:
        emoji = "🟢"
        color = "green"
    elif score_pct >= 60:
        emoji = "🟡"
        color = "orange"
    else:
        emoji = "🔴"
        color = "red"

    col1, col2 = st.columns([2, 1])

    with col1:
        st.metric(title, f"{score_pct:.1f}%")
        st.progress(score)

    with col2:
        st.markdown(f"### {emoji}")
        st.caption(f"Statuss: {color.upper()}")


def show_dimension_scores(
    dimension_results: Dict[str, Any],
    title: str = "📈 Kvalitātes Dimensijas"
) -> None:
    """
    Parāda dimensiju rezultātus.

    Args:
        dimension_results: Dict ar dimensiju rezultātiem
        title: Virsraksts
    """
    st.subheader(title)

    if not dimension_results:
        st.warning("Nav dimensiju rezultātu")
        return

    # Create columns based on number of dimensions
    cols = st.columns(len(dimension_results))

    for i, (dimension, dim_results) in enumerate(dimension_results.items()):
        with cols[i]:
            score = dim_results.get("score", 0.0)
            issues_count = len(dim_results.get("issues", []))

            # Emoji based on score
            if score >= 0.8:
                emoji = "✅"
            elif score >= 0.6:
                emoji = "⚠️"
            else:
                emoji = "❌"

            st.markdown(f"### {emoji} {dimension.title()}")
            st.metric(
                "Rezultāts",
                f"{score * 100:.1f}%",
                help=f"Kvalitātes rezultāts dimensijā {dimension}"
            )
            st.progress(score)

            if issues_count > 0:
                st.caption(f"🔍 {issues_count} problēmas")


def show_issues_summary(
    issues_by_severity: Dict[str, list],
    title: str = "🚨 Problēmu Kopsavilkums"
) -> None:
    """
    Parāda problēmu kopsavilkumu pēc svarīguma.

    Args:
        issues_by_severity: Dict ar problēmām pēc severity
        title: Virsraksts
    """
    st.subheader(title)

    severity_info = {
        "critical": {"emoji": "🔴", "label": "Kritiski", "color": "#FF0000"},
        "high": {"emoji": "🟠", "label": "Augsts", "color": "#FF8800"},
        "medium": {"emoji": "🟡", "label": "Vidējs", "color": "#FFCC00"},
        "low": {"emoji": "⚪", "label": "Zems", "color": "#CCCCCC"},
    }

    # Count by severity
    cols = st.columns(4)

    for i, (severity, info) in enumerate(severity_info.items()):
        with cols[i]:
            count = len(issues_by_severity.get(severity, []))
            st.metric(
                f"{info['emoji']} {info['label']}",
                count,
                help=f"{info['label']} prioritātes problēmas"
            )


def show_metrics_comparison(
    original_metrics: Dict[str, Any],
    improved_metrics: Dict[str, Any],
    title: str = "📊 Metriku Salīdzinājums"
) -> None:
    """
    Parāda metriku salīdzinājumu pirms/pēc.

    Args:
        original_metrics: Oriģinālie metrikas
        improved_metrics: Uzlabotie metrikas
        title: Virsraksts
    """
    st.subheader(title)

    # Overall scores
    orig_overall = original_metrics.get("overall_score", 0.0)
    imp_overall = improved_metrics.get("overall_score", 0.0)
    improvement = imp_overall - orig_overall

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Pirms", f"{orig_overall * 100:.1f}%")

    with col2:
        st.metric("Pēc", f"{imp_overall * 100:.1f}%")

    with col3:
        st.metric(
            "Uzlabojums",
            f"{improvement * 100:+.1f}%",
            delta=f"{improvement * 100:.1f}%"
        )

    # Dimension comparison
    dimensions = ["completeness", "accuracy", "consistency", "uniqueness", "validity"]

    comparison_data = []
    for dim in dimensions:
        if dim in original_metrics and dim in improved_metrics:
            orig_score = original_metrics[dim].get("overall_score", 0.0)
            imp_score = improved_metrics[dim].get("overall_score", 0.0)

            comparison_data.append({
                "Dimensija": dim.title(),
                "Pirms": f"{orig_score * 100:.1f}%",
                "Pēc": f"{imp_score * 100:.1f}%",
                "Izmaiņa": f"{(imp_score - orig_score) * 100:+.1f}%"
            })

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)


def show_fair_metrics(
    fair_metrics: Dict[str, Any],
    title: str = "🎯 FAIR Principi"
) -> None:
    """
    Parāda FAIR principu metriku.

    Args:
        fair_metrics: FAIR metrikas
        title: Virsraksts
    """
    st.subheader(title)

    fair_dimensions = {
        "findable": "🔍 Findable",
        "accessible": "🔓 Accessible",
        "interoperable": "🔄 Interoperable",
        "reusable": "♻️ Reusable",
    }

    cols = st.columns(4)

    for i, (key, label) in enumerate(fair_dimensions.items()):
        if key in fair_metrics:
            with cols[i]:
                score = fair_metrics[key].get("score", 0.0)
                st.metric(label, f"{score * 100:.0f}%")
                st.progress(score)

    # Overall FAIR score
    overall_fair = fair_metrics.get("overall_fair_score", 0.0)
    st.metric("FAIR Kopējais", f"{overall_fair * 100:.1f}%")
    st.progress(overall_fair)


def show_quality_gauge(
    score: float,
    thresholds: Optional[Dict[str, float]] = None
) -> None:
    """
    Parāda kvalitātes "gauge" ar krāsu kodēšanu.

    Args:
        score: Rezultāts (0.0-1.0)
        thresholds: Sliekšņi (optional)
    """
    if thresholds is None:
        thresholds = {
            "excellent": 0.9,
            "good": 0.8,
            "acceptable": 0.6,
            "poor": 0.4,
        }

    score_pct = score * 100

    # Determine level
    if score >= thresholds["excellent"]:
        level = "Izcili"
        color = "🟢"
    elif score >= thresholds["good"]:
        level = "Labs"
        color = "🟢"
    elif score >= thresholds["acceptable"]:
        level = "Pieņemams"
        color = "🟡"
    elif score >= thresholds["poor"]:
        level = "Vājš"
        color = "🟠"
    else:
        level = "Kritisks"
        color = "🔴"

    col1, col2 = st.columns([3, 1])

    with col1:
        st.progress(score)

    with col2:
        st.markdown(f"### {color}")
        st.caption(level)


__all__ = [
    "show_overall_score",
    "show_dimension_scores",
    "show_issues_summary",
    "show_metrics_comparison",
    "show_fair_metrics",
    "show_quality_gauge",
]
