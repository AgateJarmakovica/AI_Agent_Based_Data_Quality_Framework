"""
Upload Page - Datu augšupielādes lapa
Author: Agate Jarmakoviča

Multipage Streamlit versija.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Import components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from components import show_data_preview

st.set_page_config(
    page_title="Upload Data - healthdq-ai",
    page_icon="📤",
    layout="wide",
)

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None

st.title("📤 Datu Augšupielāde")

st.markdown("""
Augšupielādējiet savu datu kopu analīzei un kvalitātes uzlabošanai.

**Atbalstītie formāti:** CSV, Excel, JSON, Parquet
""")

# File uploader
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
                st.stop()

        st.session_state.data = data
        st.success(f"✅ Dati ielādēti: {data.shape[0]} rindas, {data.shape[1]} kolonnas")

        # Show data preview using component
        show_data_preview(data, num_rows=10, show_stats=True)

        # Next button
        st.markdown("---")
        st.info("✨ Dati veiksmīgi ielādēti! Ejiet uz 'Analysis' lapu, lai sāktu analīzi.")

    except Exception as e:
        st.error(f"❌ Kļūda ielādējot datus: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Lūdzu, augšupielādējiet failu, lai turpinātu.")

    # Show example
    with st.expander("📝 Piemērs: Kādi dati ir piemēroti?"):
        st.markdown("""
        **Labi piemēroti dati:**
        - CSV ar galvenēm
        - Excel ar vienu lapu
        - JSON ar ierakstu masīvu
        - Parquet faili

        **Datu lielums:**
        - Ieteicams: < 10 MB
        - Maksimālais: 100 MB

        **Kolonna tips:**
        - Skaitļi (age, salary, amount)
        - Teksts (name, address, diagnosis)
        - Datumi (date_of_birth, created_at)
        """)
