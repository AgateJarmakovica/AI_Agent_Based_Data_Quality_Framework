# 🛠️ Utility Scripts

Helper scripts priekš projekta setup, testēšanas un development.

## 📜 Skriti

### 1. **setup.sh** - Project Setup

Uzstāda visu projektu no nulles.

```bash
./scripts/setup.sh
```

**Kas notiek:**
- ✅ Pārbauda Python versiju (3.10+)
- ✅ Izveido virtual environment
- ✅ Instalē pakotni un dependencies
- ✅ Izveido .env failu no .env.example
- ✅ Izveido nepieciešamās mapes
- ✅ Instalē pre-commit hooks

**Kad lietot:**
- Pirmoreiz klonējot projektu
- Pēc clean install
- Jaunam komandas loceklim

---

### 2. **run_tests.sh** - Test Runner

Palaiž projektam testus.

```bash
# Visi testi
./scripts/run_tests.sh --all

# Tikai unit testi
./scripts/run_tests.sh --unit

# Ar coverage
./scripts/run_tests.sh --coverage

# Verbose output
./scripts/run_tests.sh --all --verbose
```

**Opcijas:**
- `--all` - Visi testi (default)
- `--unit` - Tikai unit testi
- `--integration` - Tikai integration testi
- `--coverage` - Ar coverage report
- `--verbose, -v` - Detalizēts output

**Output:**
- Terminal: Testa rezultāti
- `htmlcov/` - HTML coverage report (ja --coverage)

---

### 3. **clean.sh** - Cleanup

Iztīra build artifacts, cache, logs.

```bash
# Iztīrīt cache (default)
./scripts/clean.sh

# Iztīrīt visu
./scripts/clean.sh --all

# Specific cleanup
./scripts/clean.sh --cache --build --logs
```

**Opcijas:**
- `--all` - Iztīra visu
- `--cache` - Python cache (`__pycache__`, `*.pyc`)
- `--build` - Build artifacts (`dist/`, `*.egg-info`)
- `--logs` - Log files
- `--data` - Generated data (prasa apstiprinājumu!)

**Brīdinājums:** `--data` dzēš:
- `data/feedback/*.json`
- `output/*`
- `chroma_db/`
- `*.db`

---

### 4. **run_streamlit.sh** - Streamlit Launcher

Ērti palaiž Streamlit UI.

```bash
# Default (port 8501)
./scripts/run_streamlit.sh

# Custom port
./scripts/run_streamlit.sh --port 8502

# Network access
./scripts/run_streamlit.sh --host 0.0.0.0

# Multipage version
./scripts/run_streamlit.sh --multipage
```

**Opcijas:**
- `--port, -p PORT` - Ports (default: 8501)
- `--host HOST` - Host (default: localhost)
- `--multipage, -m` - Multipage versija
- `--help, -h` - Palīdzība

**Atver:**
- Single-page: `src/healthdq/ui/streamlit_app.py`
- Multipage: `src/healthdq/ui/pages/1_📤_Upload.py`

---

## 🚀 Quick Workflows

### First Time Setup
```bash
# 1. Setup
./scripts/setup.sh

# 2. Run tests
./scripts/run_tests.sh --all

# 3. Start UI
./scripts/run_streamlit.sh
```

### Development Workflow
```bash
# 1. Pull latest
git pull

# 2. Run tests
./scripts/run_tests.sh --coverage

# 3. Clean up
./scripts/clean.sh --cache

# 4. Develop...
./scripts/run_streamlit.sh
```

### Before Commit
```bash
# 1. Run tests
./scripts/run_tests.sh --all

# 2. Clean cache
./scripts/clean.sh --cache

# 3. Commit
git add .
git commit -m "Your message"
```

---

## 🔧 Make Scripts Executable

Ja skriti nav executable:

```bash
chmod +x scripts/*.sh
```

Vai izpildi caur bash:

```bash
bash scripts/setup.sh
bash scripts/run_tests.sh
```

---

## 🪟 Windows

Scripts ir Unix/Linux/Mac. Windows lietotājiem:

### Option 1: Git Bash
```bash
# Instalē Git for Windows
# Tad lieto scripts kā parasti
./scripts/setup.sh
```

### Option 2: WSL (Windows Subsystem for Linux)
```bash
# WSL terminal
./scripts/setup.sh
```

### Option 3: Manual Commands
```cmd
REM Setup
python -m venv venv
venv\Scripts\activate
pip install -e .

REM Tests
pytest tests/ -v

REM Streamlit
streamlit run src/healthdq/ui/streamlit_app.py
```

---

## 📝 Piezīmes

- Visi skriti sākas ar `set -e` (exit on error)
- Krāsains output priekš labākas lasāmības
- Help pieejama ar `--help` vai `-h`
- Droši lietot CI/CD pipeline

---

## 🤝 Contributing

Ja pievieno jaunu skriptu:

1. Pievieno shebang: `#!/bin/bash`
2. Pievieno help: `--help` opcija
3. Pievieno error handling: `set -e`
4. Dokumentē šajā README
5. Make executable: `chmod +x`

---

**Happy scripting!** 🎉
