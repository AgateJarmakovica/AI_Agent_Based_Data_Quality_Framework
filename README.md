# 🎯 healthdq-ai v2.0 - AI Agent-Based Data Quality Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**AI Agent-Based Data Quality Framework for Healthcare Data**

Promocijas darba prototips - Agate Jarmakoviča

## 📋 Saturs

- [Par Projektu](#par-projektu)
- [Galvenās Funkcijas](#galvenās-funkcijas)
- [Arhitektūra](#arhitektūra)
- [Uzstādīšana](#uzstādīšana)
- [Lietošana](#lietošana)
- [Dokumentācija](#dokumentācija)
- [Contributing](#contributing)
- [Licence](#licence)

## 🎓 Par Projektu

**healthdq-ai** ir uz MI aģentiem balstīta datu kvalitātes novērtēšanas un uzlabošanas sistēma, īpaši izstrādāta veselības aprūpes datiem. Sistēma implementē:

- 🤖 **Multi-Agent Architecture** - Specialized agents for different quality dimensions
- 🔄 **Human-in-the-Loop (HITL)** - Interactive validation and feedback
- 📊 **FAIR Principles** - Findable, Accessible, Interoperable, Reusable data
- 🧠 **Adaptive Learning** - System learns from feedback
- 🔍 **Comprehensive Analysis** - Multi-dimensional quality assessment

### Galvenās Kvalitātes Dimensijas

1. **Precision (Precizitāte)** - Format consistency, type validation, outlier detection
2. **Completeness (Pilnīgums)** - Missing value analysis and imputation
3. **Reusability (Atkārtota izmantošana)** - Standardization, metadata, documentation

## ✨ Galvenās Funkcijas

### 1. Multi-Source Data Ingestion
```python
from healthdq.loaders import FileLoader

loader = FileLoader()
data = loader.load("patient_data.csv")
```

Atbalsta:
- CSV, Excel, JSON, Parquet
- SQL databases (PostgreSQL, MySQL, SQLite)
- REST APIs
- FHIR bundles
- Real-time streams

### 2. Intelligent Multi-Agent Analysis
```python
from healthdq.pipeline import DataQualityPipeline

pipeline = DataQualityPipeline()
results = await pipeline.run(
    source="data.csv",
    quality_dimensions=["precision", "completeness", "reusability"],
    apply_improvements=True,
    require_hitl=True
)
```

### 3. Automated Quality Metrics
```python
from healthdq.metrics import MetricsCalculator

calculator = MetricsCalculator()
metrics = calculator.calculate_all(data)

print(f"Overall Score: {metrics['overall_score']}")
print(f"Completeness: {metrics['completeness']['overall_score']}")
```

### 4. Human-in-the-Loop Validation
- Interactive web interface (Streamlit)
- Approve/reject AI suggestions
- Provide feedback for learning
- Track validation history

### 5. FAIR Reproducibility
- Complete metadata tracking
- Version control
- Hash-based data provenance
- Audit trail

## 🏗️ Arhitektūra

```
┌─────────────────────────────────────────────────────┐
│                  User Interface                      │
│          (Streamlit UI / REST API)                   │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│            Pipeline Orchestrator                     │
│         (DataQualityPipeline)                        │
└─┬──────────────┬──────────────┬────────────────────┘
  │              │              │
  │              │              │
┌─▼────────┐ ┌──▼─────────┐ ┌─▼───────────┐
│ Precision│ │Completeness│ │Reusability  │
│  Agent   │ │   Agent    │ │   Agent     │
└─┬────────┘ └──┬─────────┘ └─┬───────────┘
  │              │              │
  └──────────────┴──────────────┘
                 │
        ┌────────▼─────────┐
        │  Coordinator     │
        │     Agent        │
        └────────┬─────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────────┐      ┌─────────▼──────┐
│ Feedback   │      │  Rules Engine  │
│  Memory    │      │  & Transform   │
└────────────┘      └────────────────┘
```

### Agent Communication Protocol (ACP)

Aģenti komunikē caur standartizētu protokolu:
- Request/Response pattern
- Broadcast messaging
- Asynchronous task coordination
- Priority-based routing

## 🚀 Uzstādīšana

### Prerequisites

- Python 3.10+
- pip or conda
- Git

### Quick Install

```bash
# Clone repository
git clone https://github.com/AgateJarmakovica/AI_Agent_Based_Data_Quality_Framework.git
cd AI_Agent_Based_Data_Quality_Framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

Required configuration:
```env
OPENAI_API_KEY=your_key_here
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4
LOG_LEVEL=INFO
```

## 📖 Lietošana

### Command Line Interface

```bash
# Basic analysis
healthdq analyze data.csv --dimensions precision,completeness

# With custom config
healthdq analyze data.csv --config configs/custom.yml

# Export results
healthdq analyze data.csv --output results.csv --format excel
```

### Python API

```python
import asyncio
from healthdq.pipeline import DataQualityPipeline

async def main():
    # Initialize pipeline
    pipeline = DataQualityPipeline()

    # Run analysis
    results = await pipeline.run(
        source="patient_data.csv",
        quality_dimensions=["precision", "completeness", "reusability"],
        apply_improvements=True,
        require_hitl=False,  # Set to True for HITL validation
        output_path="improved_data.csv"
    )

    # Access results
    print(f"Overall Score: {results['quality_results']['overall_score']}")
    print(f"Issues Found: {len(results['quality_results']['improvement_plan']['actions'])}")

    # Get improved data
    improved_data = results['data']

    # Access detailed metrics
    metrics = results['metrics']
    print(f"Completeness: {metrics['completeness']['overall_score']}")
    print(f"FAIR Score: {metrics['fair_metrics']['overall_fair_score']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Web UI (Streamlit)

```bash
# Start Streamlit app
streamlit run src/healthdq/ui/streamlit_app.py

# Or use shortcut
healthdq-ui
```

Access at: `http://localhost:8501`

### REST API

```bash
# Start FastAPI server
uvicorn healthdq.api.server:app --reload

# Or use shortcut
healthdq-api
```

API docs at: `http://localhost:8000/docs`

Example API request:
```python
import requests

# Upload data
files = {'file': open('data.csv', 'rb')}
response = requests.post('http://localhost:8000/api/upload', files=files)
file_id = response.json()['file_id']

# Analyze
response = requests.post(
    'http://localhost:8000/api/analyze',
    json={
        'file_id': file_id,
        'dimensions': ['precision', 'completeness', 'reusability'],
        'apply_improvements': True
    }
)

results = response.json()
print(f"Analysis complete: {results['status']}")
```

## 📚 Dokumentācija

### Project Structure

```
healthdq-ai/
├── configs/          # Configuration files (YAML)
├── data/            # Sample data and outputs
├── docs/            # Documentation
├── src/healthdq/    # Main package
│   ├── agents/      # Agent implementations
│   ├── communication/  # Agent communication
│   ├── loaders/     # Data loaders
│   ├── metrics/     # Quality metrics
│   ├── rules/       # Rules engine
│   ├── schema/      # Schema learning
│   ├── ui/          # User interface
│   └── api/         # REST API
├── tests/           # Test suite
└── scripts/         # Utility scripts
```

### Configuration Files

- `configs/agents.yml` - Agent configuration
- `configs/rules.yml` - Quality rules
- `configs/prompts.yml` - LLM prompts
- `.env` - Environment variables

### Examples

See `notebooks/` for Jupyter notebook examples:
- `demo.ipynb` - Basic usage tutorial
- `experiments.ipynb` - Advanced features

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/healthdq tests/

# Run specific test
pytest tests/test_agents.py

# Run integration tests
pytest tests/integration/
```

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
mypy src/
```

## 📄 Licence

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Agate Jarmakoviča**
- GitHub: [@AgateJarmakovica](https://github.com/AgateJarmakovica)
- Project: [AI_Agent_Based_Data_Quality_Framework](https://github.com/AgateJarmakovica/AI_Agent_Based_Data_Quality_Framework)

## 🙏 Acknowledgments

This framework is based on research in:
- Data-Centric AI (Andrew Ng, 2021)
- FAIR Principles (Wilkinson et al., 2016)
- Agent Communication (FIPA Standards)
- Human-in-the-Loop ML (Amershi et al., 2014)

## 📊 Project Status

- ✅ Core framework implemented
- ✅ Multi-agent system operational
- ✅ Basic HITL interface
- 🚧 Advanced ML imputation (in progress)
- 🚧 Real-time streaming (planned)
- 📅 Full FHIR support (planned)

---

**Version:** 2.0.0
**Last Updated:** 2025-11-06
**Status:** Beta
