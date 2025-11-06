"""
Data Viewer Component - Datu apskates komponente
Author: Agate Jarmakoviča
"""

import streamlit as st
import pandas as pd
from typing import Optional


def show_data_preview(
    data: pd.DataFrame,
    title: str = "📊 Datu Priekšskatījums",
    num_rows: int = 10,
    show_stats: bool = True,
) -> None:
    """
    Parāda datu priekšskatījumu ar statistiku.

    Args:
        data: DataFrame
        title: Virsraksts
        num_rows: Cik rindas rādīt
        show_stats: Vai rādīt statistiku
    """
    st.subheader(title)

    if show_stats:
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rindas", f"{data.shape[0]:,}")

        with col2:
            st.metric("Kolonnas", data.shape[1])

        with col3:
            missing_pct = (data.isna().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            st.metric("Trūkstoši %", f"{missing_pct:.1f}%")

        with col4:
            duplicates = data.duplicated().sum()
            st.metric("Dublikāti", duplicates)

    # Data preview
    st.dataframe(data.head(num_rows), use_container_width=True)


def show_column_info(data: pd.DataFrame) -> None:
    """Parāda detalizētu informāciju par kolonnām."""
    st.subheader("📋 Kolonnu Informācija")

    column_info = []
    for col in data.columns:
        info = {
            "Kolonna": col,
            "Tips": str(data[col].dtype),
            "Unikālas": data[col].nunique(),
            "Trūkstoši": data[col].isna().sum(),
            "Trūkstoši %": f"{(data[col].isna().sum() / len(data)) * 100:.1f}%",
        }
        column_info.append(info)

    df_info = pd.DataFrame(column_info)
    st.dataframe(df_info, use_container_width=True, hide_index=True)


def show_data_quality_summary(data: pd.DataFrame) -> None:
    """Parāda datu kvalitātes kopsavilkumu."""
    st.subheader("✅ Kvalitātes Kopsavilkums")

    # Calculate metrics
    total_cells = data.shape[0] * data.shape[1]
    missing_cells = data.isna().sum().sum()
    completeness = 1 - (missing_cells / total_cells)

    duplicates = data.duplicated().sum()
    duplicate_ratio = duplicates / len(data)

    # Display
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Pilnīgums",
            f"{completeness * 100:.1f}%",
            help="Procentuālā daļa no netrūkstošām vērtībām"
        )
        st.progress(completeness)

    with col2:
        st.metric(
            "Unikālums",
            f"{(1 - duplicate_ratio) * 100:.1f}%",
            help="Procentuālā daļa no unikālām rindām"
        )
        st.progress(1 - duplicate_ratio)


def show_missing_values_heatmap(data: pd.DataFrame) -> None:
    """Parāda trūkstošo vērtību heatmap."""
    st.subheader("🔥 Trūkstošo Vērtību Karte")

    # Calculate missing percentages per column
    missing_pct = (data.isna().sum() / len(data) * 100).sort_values(ascending=False)

    if missing_pct.sum() == 0:
        st.success("✅ Nav trūkstošu vērtību!")
        return

    # Show only columns with missing values
    missing_cols = missing_pct[missing_pct > 0]

    if len(missing_cols) > 0:
        chart_data = pd.DataFrame({
            "Kolonna": missing_cols.index,
            "Trūkstoši %": missing_cols.values
        })

        st.bar_chart(chart_data.set_index("Kolonna"))

        # Show table
        st.dataframe(
            chart_data,
            use_container_width=True,
            hide_index=True
        )


def show_data_sample(
    data: pd.DataFrame,
    sample_size: int = 5,
    random: bool = False
) -> None:
    """
    Parāda datu paraugu.

    Args:
        data: DataFrame
        sample_size: Parauga lielums
        random: Vai ņemt nejaušu paraugu
    """
    st.subheader("🔍 Datu Paraugs")

    if random:
        if len(data) > sample_size:
            sample = data.sample(n=sample_size)
        else:
            sample = data
    else:
        sample = data.head(sample_size)

    st.dataframe(sample, use_container_width=True)


__all__ = [
    "show_data_preview",
    "show_column_info",
    "show_data_quality_summary",
    "show_missing_values_heatmap",
    "show_data_sample",
]
