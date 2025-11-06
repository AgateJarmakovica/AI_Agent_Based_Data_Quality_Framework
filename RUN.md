# 🚀 Kā Palaist healthdq-ai

## Ātri Sākt

### 1. Minimālā instalācija (ja nav laika)

```bash
# Instalē pamata pakotnes
pip install streamlit pandas pyyaml

# Palaiž aplikāciju
streamlit run src/healthdq/ui/streamlit_app.py
```

### 2. Pilna instalācija (ieteicams)

```bash
# Instalē visu projektu ar visām atkarībām
pip install -e .

# Palaiž aplikāciju
streamlit run src/healthdq/ui/streamlit_app.py
```

---

## 📍 Fails, ko palaist

**Galvenais fails:** `src/healthdq/ui/streamlit_app.py`

---

## 🌐 Piekļuve

Pēc palaišanas atver pārlūkā:

**http://localhost:8501**

---

## ⚙️ Pārbaudes

### Vai streamlit ir instalēts?

```bash
streamlit --version
# Vajadzētu redzēt: Streamlit, version 1.37.0
```

### Vai fails eksistē?

```bash
ls -lh src/healthdq/ui/streamlit_app.py
# Vajadzētu redzēt: -rw-r--r-- ... 19K ... streamlit_app.py
```

---

## 🐛 Ja kaut kas nestrādā

### Kļūda: "streamlit: command not found"

```bash
# Instalē streamlit
pip install streamlit
```

### Kļūda: "No module named 'healthdq'"

```bash
# Instalē projektu
pip install -e .
```

### Kļūda: "ModuleNotFoundError: No module named 'pandas'"

```bash
# Instalē trūkstošās pakotnes
pip install pandas pyyaml
```

### Port jau aizņemts (8501)

```bash
# Izmanto citu portu
streamlit run src/healthdq/ui/streamlit_app.py --server.port 8502
```

---

## 🎯 Pilna Komanda ar Opcijām

```bash
streamlit run src/healthdq/ui/streamlit_app.py \
  --server.port 8501 \
  --server.address localhost \
  --browser.gatherUsageStats false
```

---

## 📱 Piekļuve no citas ierīces (tīklā)

```bash
# Palaiž ar network access
streamlit run src/healthdq/ui/streamlit_app.py \
  --server.address 0.0.0.0

# Pēc tam vari piekļūt no citas ierīces:
# http://[tavs-ip]:8501
```

---

## 🛑 Apturēt Aplikāciju

Terminālī spied: **`Ctrl + C`**

---

## 💡 Tips

Ja izmanto virtuālo vidi:

```bash
# Aktivizē venv
source venv/bin/activate  # Linux/Mac
# VAI
venv\Scripts\activate     # Windows

# Pēc tam palaiž
streamlit run src/healthdq/ui/streamlit_app.py
```

---

## 📖 Pēc palaišanas

1. Atver http://localhost:8501
2. Seko 6-posmu workflow:
   - 📤 Upload data
   - 📊 Run analysis
   - 🔍 Review results
   - ✅ Approve changes
   - 🔄 Apply transformations
   - 📥 Download results

Vairāk info: `docs/QUICK_START.md`

---

**Lai veicas!** 🎉
