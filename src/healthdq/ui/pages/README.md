# 📄 Streamlit Pages

Šī mape satur atsevišķas Streamlit lapas multipage aplikācijas versijai.

## 📂 Struktūra

```
pages/
├── 1_📤_Upload.py        # Datu augšupielāde
├── 2_📊_Analysis.py      # AI analīze
├── 3_🔍_Review.py        # (TODO) Rezultātu pārskats
├── 4_✅_Approval.py      # (TODO) HITL apstiprināšana
└── 5_📥_Results.py       # (TODO) Rezultāti un lejupielāde
```

## 🚀 Kā lietot

### Variants 1: Vienas lapas versija (ieteicams)

```bash
streamlit run src/healthdq/ui/streamlit_app.py
```

Visa funkcionalitāte vienā failā ar integrated workflow.

### Variants 2: Multipage versija

```bash
streamlit run src/healthdq/ui/pages/1_📤_Upload.py
```

Streamlit automātiski atpazīs visas lapas un izveidos sidebar navigāciju.

## 📝 Piezīmes

- **Pašlaik implementētas:** Upload, Analysis
- **TODO:** Review, Approval, Results

Pilna funkcionalitāte ir pieejama `streamlit_app.py` failā.

## 🔄 Session State

Lapas izmanto `st.session_state` lai saglabātu datus starp lapām:

- `st.session_state.data` - Ielādētie dati
- `st.session_state.quality_results` - Analīzes rezultāti
- `st.session_state.approval_request` - Apstiprināšanas pieprasījums
- `st.session_state.improved_data` - Uzlabotie dati

## 🎨 Komponenti

Lapas izmanto atkārtojamus komponentus no `components/` mapes:

```python
from components import show_data_preview
show_data_preview(data)
```
