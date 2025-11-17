# 🚀 healthdq-ai Streamlit Quick Start

Šis ir īss pamācība, kā palaist **healthdq-ai** Streamlit lietotāja saskarni.

---

## ⚡ Ātrā Palaišana (3 Soļi)

### 1. Instalēt Dependencies

```bash
# Minimālā instalācija (tikai UI)
pip install streamlit pandas pyyaml numpy scikit-learn
```

**Vai pilna instalācija:**

```bash
# Pilnas iespējas (ar AI/ML)
pip install -r requirements.txt
```

### 2. Palaist Streamlit

**Linux/Mac:**
```bash
# Izmantot gatavo skriptu
bash scripts/start_streamlit.sh

# Vai manuāli
streamlit run src/healthdq/ui/streamlit_app.py
```

**Windows:**
```cmd
REM Izmantot gatavo skriptu
scripts\start_streamlit.bat

REM Vai manuāli
streamlit run src\healthdq\ui\streamlit_app.py
```

### 3. Atvērt Pārlūkprogrammā

Automātiski atvērsies: **http://localhost:8501**

---

## 📋 Pilns Copy-Paste Piemērs

```bash
# 1. Pāriet uz projekta direktoriju
cd /home/user/AI_Agent_Based_Data_Quality_Framework

# 2. Instalēt minimālās dependencies
pip install streamlit pandas pyyaml numpy

# 3. Palaist
streamlit run src/healthdq/ui/streamlit_app.py
```

**Rezultāts:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.X:8501
```

---

## 🎨 Ko Var Darīt ar Streamlit UI?

### 1. **Augšupielādēt Datus** 📤
- CSV, Excel, JSON, Parquet failus
- Drag & drop vai file picker

### 2. **Veikt Datu Kvalitātes Analīzi** 📊
- **Precision** - Format consistency, outliers
- **Completeness** - Missing values, duplicates
- **Reusability** - FAIR compliance, metadata

### 3. **Healthcare Model Detection** 🏥
- Automātiska FHIR/HL7/OMOP atpazīšana
- Medical coding detection (SNOMED, LOINC, ICD-10)

### 4. **Human-in-the-Loop (HITL)** ✅
- Pārskatīt AI ieteiktos uzlabojumus
- Apstiprināt vai noraidīt izmaiņas
- Auto-approve funkcija (configurable threshold)

### 5. **Interaktīvā Data Editing** ✏️
- Labošana tiešsaistē
- Real-time validation
- Undo/Redo

### 6. **Metriku Vizualizācija** 📈
- DQ Score dashboard
- Interactive charts (Plotly)
- Dimension breakdown

### 7. **Eksportēt Rezultātus** 💾
- Uzlabotus datus (CSV, Excel, JSON)
- Kvalitātes pārskatus
- Transformation history

---

## ⚠️ Problēmu Risināšana

### ❌ "ModuleNotFoundError: No module named 'streamlit'"

**Risinājums:**
```bash
pip install streamlit
```

### ❌ "ModuleNotFoundError: No module named 'healthdq'"

**Risinājums:** Pievienot src to PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%cd%\src           # Windows
```

### ❌ Port 8501 jau aizņemts

**Risinājums:** Izmantot citu portu
```bash
streamlit run src/healthdq/ui/streamlit_app.py --server.port 8502
```

### ❌ "TypeError" vai citas kļūdas palaišanas laikā

**Risinājums:** Instalēt pilnas dependencies
```bash
pip install -r requirements.txt
```

---

## 🔧 Papildu Opcijas

### Debug Režīms
```bash
streamlit run src/healthdq/ui/streamlit_app.py --logger.level=debug
```

### Tumšā Tēma
```bash
streamlit run src/healthdq/ui/streamlit_app.py --theme.base=dark
```

### Cits Ports
```bash
streamlit run src/healthdq/ui/streamlit_app.py --server.port 8080
```

### Bez Auto-Reload
```bash
streamlit run src/healthdq/ui/streamlit_app.py --server.runOnSave=false
```

---

## 📂 Streamlit Failu Struktūra

```
src/healthdq/ui/
├── streamlit_app.py          # Galvenā aplikācija (1,057 lines)
├── components/                # UI komponentes
│   ├── data_viewer.py         # Data display widget
│   ├── hitl_panel.py          # HITL approval panel
│   └── metrics_dashboard.py  # Metrics visualization
└── pages/                     # Multi-page app
    ├── 1_📤_Upload.py         # File upload page
    └── 2_📊_Analysis.py       # Analysis page
```

---

## 💡 Padomi

### 1. **Izmantot Sample Data**
Pievienojiet sample CSV failu `data/sample/` direktorijā un augšupielādējiet caur UI.

### 2. **HITL Workflow**
- Augšupielādējiet datus
- Izvēlieties quality dimensions
- Pārskatiet AI ieteikumus
- Apstipriniet vai noraidiet
- Eksportējiet uzlabotus datus

### 3. **Auto-Approve Threshold**
Pielāgojiet `configs/hitl.yml`:
```yaml
workflow:
  auto_approve_threshold: 0.95  # 95% agreement
```

### 4. **Konfigurācija**
Rediģējiet `configs/` failus:
- `agents.yml` - Agent settings
- `rules.yml` - Quality rules
- `hitl.yml` - HITL settings

---

## 📚 Vairāk Informācijas

- **Streamlit Dokumentācija:** https://docs.streamlit.io
- **healthdq-ai UI Kods:** `src/healthdq/ui/streamlit_app.py`
- **HITL Dokumentācija:** `docs/human_in_the_loop.md`
- **Projekta README:** `README.md`
- **Projekta Analīze:** `PROJECT_STRUCTURE_ANALYSIS.md`

---

## 🆘 Palīdzība

Ja rodas problēmas:

1. Pārbaudiet, vai esat projekta saknes direktorijā
2. Pārbaudiet Python versiju (vajag 3.10+)
3. Instalējiet pilnas dependencies: `pip install -r requirements.txt`
4. Skatiet Streamlit logus terminālī
5. Mēģiniet restartēt ar `Ctrl+C` un palaist vēlreiz

---

**Versija:** healthdq-ai v2.1
**Autors:** Agate Jarmakoviča
**Datums:** 2025-11-17

Lai veicas! 🚀
