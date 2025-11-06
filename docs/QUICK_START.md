# 🚀 Quick Start Guide

Ātrā sākuma instrukcija healthdq-ai lietošanai ar HITL workflow.

## ⏱️ 5-Minute Setup

### 1. Instalācija

```bash
# Clone repository
git clone https://github.com/AgateJarmakovica/AI_Agent_Based_Data_Quality_Framework.git
cd AI_Agent_Based_Data_Quality_Framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -e .
```

### 2. Konfigurācija

```bash
# Copy environment file
cp .env.example .env

# Edit (optional - Streamlit works without API keys for basic features)
nano .env
```

### 3. Palaist Streamlit UI

```bash
streamlit run src/healthdq/ui/streamlit_app.py
```

Atvērsies: `http://localhost:8501`

---

## 📝 Pirmā Lietošana (10 minūtes)

### Solis 1: Augšupielādēt Testdatus

1. Atveriet Streamlit UI
2. Klikšķiniet "Choose file"
3. Izvēlieties CSV/Excel failu
4. Redzēsiet datu priekšskatījumu

**Test data:** Varat izmantot sample data no `data/sample/`

### Solis 2: Sākt Analīzi

1. Klikšķiniet "▶️ Turpināt uz Novērtējumu"
2. Izvēlaties dimensijas (vai atstājiet default)
3. Klikšķiniet "🚀 Sākt Analīzi"
4. Gaidiet 30-60 sekundes

### Solis 3: Pārskatīt Rezultātus

Jūs redzēsiet:
- 📊 Kopējo kvalitātes rezultātu (%)
- 📈 Kvalitātes dimensijas
- 🔍 Konstatētas problēmas
- 💡 Ieteiktos uzlabojumus

**SVARĪGI:** Šajā posmā nekādas izmaiņas vēl nav piemērotas!

### Solis 4: Apstiprināt Izmaiņas

1. Klikšķiniet "✅ Pārskatīt un Apstiprināt"
2. Pārskatiet katru izmaiņu:
   - Izlasiet aprakstu
   - Sapratiet ietekmi
   - Pieņemiet lēmumu
3. Klikšķiniet:
   - "✅ Apstiprināt" - ja piekrītat
   - "❌ Noraidīt" - ja nepiekrītat
   - VAI "✅ Apstiprināt Visas" - ja uzticaties AI

### Solis 5: Piemērot un Lejupielādēt

1. Klikšķiniet "🚀 Pabeigt un Piemērot Izmaiņas"
2. Gaidiet transformāciju
3. Redzēsiet rezultātus un salīdzinājumu
4. Klikšķiniet "📥 Lejupielādēt CSV"

**Gatavs!** Jums ir uzlaboti dati.

---

## 🔄 Tipisks Workflow

```
📤 Upload Data
    ↓
📊 AI Analysis (30-60s)
    ↓
🔍 Review Results (PIRMS izmaiņām!)
    ├─ Redzēt problēmas
    ├─ Redzēt ieteikumus
    └─ Saprast ietekmi
    ↓
✅ Approve/Reject (JŪSU lēmums!)
    ├─ Katru izmaiņu atsevišķi
    └─ Vai visas uzreiz
    ↓
🔄 Apply Changes (automātiski)
    ├─ Piemēro apstiprinātās
    └─ Izlaiž noraidītās
    ↓
📥 Download Results
```

**Laika patēriņš:**
- Upload: 10s
- Analysis: 30-60s
- Review: 2-5 min (atkarībā no problēmu skaita)
- Apply: 10-30s
- **Total: ~5-10 minūtes**

---

## 💡 Tips & Tricks

### Tip 1: Sāciet ar Mazu Failu

Pirmoreiz testējot:
- Izmantojiet < 1000 rindas
- Sapratiet, kā sistēma strādā
- Tad lietojiet uz lielākiem datiem

### Tip 2: Izmantojiet Sample Data

```bash
# Repository ietver sample data
ls data/sample/
# healthcare_500.csv
```

### Tip 3: Backup ir King

```bash
# Pirms apstrādes
cp my_data.csv my_data_backup.csv
```

### Tip 4: Pārbaudiet Rezultātus

Pēc lejupielādes:
1. Atveriet Excel/CSV viewer
2. Salīdziniet ar oriģinālu
3. Pārliecinaties, ka viss OK

---

## 🐛 Troubleshooting

### Problēma: Streamlit nepalaiž

```bash
# Pārbaudiet instalāciju
pip list | grep streamlit

# Ja nav, instalējiet
pip install streamlit
```

### Problēma: Analīze "karajas"

- Pārbaudiet, vai dati ir pārāk lieli (> 10MB)
- Mēģiniet ar mazāku failu
- Pārstartējiet Streamlit

### Problēma: Kļūda "No module named 'healthdq'"

```bash
# Instalējiet package
pip install -e .
```

### Problēma: Encoding error CSV failam

- Mēģiniet saglabāt CSV kā UTF-8
- Vai izmantojiet Excel formātu

---

## 📚 Nākamie Soļi

Pēc pirmā mēģinājuma:

1. **Izlasiet HITL Guide:**
   ```
   docs/HITL_GUIDE.md
   ```

2. **Apskatiet README:**
   ```
   README.md
   ```

3. **Izmēģiniet Python API:**
   ```python
   from healthdq.pipeline import DataQualityPipeline
   # ... skatīt README
   ```

4. **Konfigurējiet pēc vajadzības:**
   ```
   configs/agents.yml
   configs/rules.yml
   ```

---

## 🎓 Video Tutorials (Coming Soon)

- [ ] Basic Usage (5 min)
- [ ] HITL Workflow (10 min)
- [ ] Advanced Configuration (15 min)

---

## 💬 Jautājumi?

- 📖 Dokumentācija: `docs/`
- 🐛 Issues: [GitHub Issues]
- 📧 Email: [kontakts]

---

**Lai veicas ar datu kvalitātes uzlabošanu!** 🎉
