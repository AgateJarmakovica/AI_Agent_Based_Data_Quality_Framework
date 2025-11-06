# 🚀 Kā Palaist healthdq-ai

> **⚠️ SVARĪGI:** Vienmēr palaiž aplikāciju no projekta saknes direktorijas (`AI_Agent_Based_Data_Quality_Framework/`), nevis no `src/` vai citām apakšdirektorijām.

## Ātri Sākt

### 1. Streamlit UI režīms (Ātrākais - Ieteicams sākumam)

Izmanto minimālas atkarības tikai UI funkcionalitātei ar simulētu analīzi:

```bash
# Pārliecinies, ka esi projekta saknes direktorijā
cd /path/to/AI_Agent_Based_Data_Quality_Framework

# Instalē tikai UI pakotnes (ātri, ~50MB)
pip install -r requirements-streamlit.txt

# Palaiž aplikāciju NO PROJEKTA SAKNES
streamlit run src/healthdq/ui/streamlit_app.py
```

**✅ Priekšrocības:**
- Ātra instalācija (dažas sekundes)
- Maza izmēra (~50MB)
- Ideāli demo/testēšanai
- Darbosies ar simulētu AI analīzi
- Nav nepieciešama `pip install -e .` instalācija

**⚠️ Ierobežojumi:**
- Nav pieejama pilna AI/LLM funkcionalitāte
- Izmanto vienkāršu uz noteikumiem balstītu analīzi

**📍 Svarīgi:**
- VIENMĒR palaiž no projekta saknes direktorijas
- Aplikācija automātiski pievieno projektu Python ceļam

### 2. Pilna AI instalācija (Prasa vairāk laika)

Pilna funkcionalitāte ar AI aģentiem, LLM un vektoru datubāzi:

```bash
# Instalē visas atkarības (var ilgt ~5-10 min, ~3GB)
pip install -r requirements.txt

# VAI instalē kā pakotni
pip install -e .

# Palaiž aplikāciju
streamlit run src/healthdq/ui/streamlit_app.py
```

**✅ Priekšrocības:**
- Pilna AI funkcionalitāte
- Multi-agent analīze
- LangChain/LangGraph integrācija
- ChromaDB vektoru atmiņa
- Transformers un torch atbalsts

**⚠️ Prasības:**
- Lielāks lejupielādes izmērs (~3GB)
- Ilgāka instalācija
- Vairāk RAM (~4GB+)

### 3. Ātras pārbaudes instalācija (tikai UI bez instalācijas)

```bash
# Instalē tikai 3 pamata pakotnes
pip install streamlit pandas pyyaml

# Palaiž aplikāciju
streamlit run src/healthdq/ui/streamlit_app.py
```

> **Piezīme:** Ja ML pakotnes nav instalētas, aplikācija automātiski pārslēdzas uz demo režīmu un parādīs brīdinājumu.

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

### Kļūda: "Error installing requirements"

Ja instalējot `requirements.txt` rodas kļūdas (torch, transformers, utt.):

```bash
# Risinājums 1: Izmanto minimālās atkarības (IETEICAMS)
pip install -r requirements-streamlit.txt

# Risinājums 2: Instalē pakāpeniski
pip install streamlit pandas pyyaml
pip install -e .  # Pēc tam pārējās

# Risinājums 3: Tikai pamatpakotnes
pip install streamlit pandas pyyaml python-dotenv loguru pydantic
```

**Piezīme:** Aplikācija automātiski noteiks, kuras pakotnes trūkst un strādās demo režīmā.

### Kļūda: "streamlit: command not found"

```bash
# Instalē streamlit
pip install streamlit
```

### Kļūda: "No module named 'healthdq'"

**Problēma:** Aplikācija nevar atrast healthdq moduli

**Risinājumi:**
```bash
# Risinājums 1: Palaiž no projekta saknes (IETEICAMS)
cd /path/to/AI_Agent_Based_Data_Quality_Framework
streamlit run src/healthdq/ui/streamlit_app.py

# Risinājums 2: Instalē kā pakotni
pip install -e .

# Risinājums 3: Iestatīt PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/AI_Agent_Based_Data_Quality_Framework"
```

**Piezīme:** Ja palaiž no projekta saknes, aplikācija automātiski pievieno projektu Python ceļam.

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

### Brīdinājums: "⚠️ Demo režīms"

Ja aplikācija parāda šo brīdinājumu:

```
⚠️ Demo režīms: Daži ML funkcionalitāte nav pieejama.
```

**Iemesls:** Nav instalētas visas ML pakotnes (langchain, chromadb, torch)

**Risinājums:**
- Ja vēlies pilnu funkcionalitāti: `pip install -r requirements.txt`
- Ja vēlies tikai testēt UI: turpini izmantot demo režīmu (darbosies ar vienkāršu analīzi)

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
