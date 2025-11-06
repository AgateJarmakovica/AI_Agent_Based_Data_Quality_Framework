# healthdq-ai v2.0 — AI Agent-Based Data Quality Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Adaptive AI-Powered Data Quality Framework for Healthcare Data**

Prototype developed as part of PhD research — Agate Jarmakoviča

---

## Table of Contents

- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Healthcare Data Model Support](#healthcare-data-model-support)
- [Documentation](#documentation)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About the Project

**healthdq-ai** is an AI agent-based data quality assessment and improvement framework designed specifically for healthcare datasets. The system implements a multi-agent architecture with human-in-the-loop validation, adaptive learning mechanisms, and automated healthcare data model detection to ensure high-quality, FAIR-compliant clinical data.

### Core Capabilities

- **Multi-Agent System** — Specialized agents for each data quality dimension
- **Adaptive Learning** — ML-powered schema discovery and continuous improvement from feedback
- **Healthcare Model Detection** — Automatic recognition of FHIR, HL7 v2.x, OMOP CDM, and generic EHR formats
- **Human-in-the-Loop (HITL)** — Interactive validation and approval workflows
- **FAIR Principles** — Findable, Accessible, Interoperable, Reusable data governance
- **LLM Integration** — Semantic analysis with GPT-4, Claude, or local models
- **Comprehensive Analysis** — Multi-dimensional, explainable quality scoring with publication-accurate formulas

### Quality Dimensions

| Dimension | Description | Formula |
|-----------|-------------|---------|
| **Accuracy** | Format consistency, outlier detection using Isolation Forest + LOF | `Accuracy = 1 - (anomalies / total_records)` |
| **Completeness** | Missing value detection and imputation recommendations | `Completeness = 1 - (missing / total_values)` |
| **Reusability** | Metadata, documentation, version control (FAIR principles) | `Reusability = (documentation + metadata + version) / 3` |

**Weighted DQ Score:** `DQ_total = 0.4×Accuracy + 0.4×Completeness + 0.2×Reusability` (ISO 25024)

---

## Key Features

### 1. Adaptive Schema Learning

Automatically learn data schemas using unsupervised machine learning:

```python
from healthdq.learners import SchemaLearner

learner = SchemaLearner()
schema = learner.learn(dataframe, schema_name="patient_data")

# Schema includes:
# - Inferred column types (patient_id, diagnosis, medication, etc.)
# - Statistical constraints (ranges, enumerations, patterns)
# - Healthcare field detection (ICD codes, LOINC, vital signs)
# - ML-based column grouping (related fields clustered together)

# Validate new data against learned schema
validation = learner.validate_against_schema(new_data, schema)
print(f"Compliance: {validation['compliance_score']}")
```

**Features:**
- Column type inference and semantic classification
- Healthcare-specific field detection
- ML-based column clustering (KMeans)
- Pattern recognition (medical codes, dates, identifiers)
- Constraint discovery (ranges, enums, formats)
- Confidence scoring

### 2. Healthcare Data Model Detection

Automatically detect and classify healthcare data formats:

```python
from healthdq.learners import HealthcareDataDetector

detector = HealthcareDataDetector()
model_info = detector.detect(dataframe)

print(f"Model: {model_info['model_type']}")        # e.g., "fhir"
print(f"Sub-type: {model_info['sub_type']}")       # e.g., "Patient"
print(f"Confidence: {model_info['confidence']}")   # e.g., 0.87
```

**Supported Models:**
- **FHIR** (Fast Healthcare Interoperability Resources) — Resource types: Patient, Observation, Condition, MedicationRequest, Procedure, Encounter
- **HL7 v2.x** — Message segments: MSH, PID, PV1, OBR, OBX, DG1, PR1
- **OMOP CDM** — Tables: person, observation_period, visit_occurrence, condition_occurrence, drug_exposure, measurement
- **Generic EHR** — Common electronic health record formats

**Medical Coding Systems Detected:**
- SNOMED CT (`^\d{6,18}$`)
- LOINC (`^\d{4,5}-\d{1}$`)
- ICD-10 (`^[A-Z]\d{2}\.?\d{0,2}$`)
- CPT (`^\d{5}$`)
- RxNorm (`^\d{6,8}$`)

### 3. LLM-Powered Semantic Analysis

Use advanced LLM prompts for clinical validation:

```python
from healthdq.prompts import load_prompt

# Load specialized prompts
fhir_prompt = load_prompt("fhir_analysis")
semantic_prompt = load_prompt("semantic_analysis")
schema_prompt = load_prompt("schema_learning")

# Use with any LLM provider
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": fhir_prompt},
        {"role": "user", "content": json.dumps(fhir_resource)}
    ]
)

findings = json.loads(response.choices[0].message.content)
# Returns structured findings with quality score, issues, and recommendations
```

**Available Prompts:**
- `fhir_analysis.md` — FHIR resource validation and compliance checking
- `schema_learning.md` — Automated schema inference and model detection
- `semantic_analysis.md` — Clinical plausibility and terminology validation

**Compatible with:**
- OpenAI GPT-4
- Anthropic Claude
- Local models (Llama, Mistral)
- BioClinicalBERT (for medical text embeddings)

### 4. Multi-Source Data Ingestion

```python
from healthdq.loaders import FileLoader

loader = FileLoader()
data = loader.load("patient_data.csv")
```

**Supported Sources:**
- **Files:** CSV, Excel, JSON, Parquet
- **Databases:** PostgreSQL, MySQL, SQLite
- **APIs:** REST endpoints
- **Healthcare Standards:** FHIR bundles
- **Streaming:** Real-time data streams

### 5. Intelligent Multi-Agent Analysis

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

### 6. Publication-Accurate Quality Metrics

```python
from healthdq.metrics import MetricsCalculator

calculator = MetricsCalculator()

# Calculate comprehensive metrics
metrics = calculator.calculate_all(data)
print(f"Overall Score: {metrics['overall_score']}")

# Calculate publication DQ score with exact formulas
dq_score = calculator.calculate_publication_dq_score(data, original_data)
print(f"DQ Total: {dq_score['dq_total']}")
print(f"Formula: {dq_score['formula']}")

# Breakdown by dimension
for dimension, info in dq_score['dimensions'].items():
    print(f"{dimension}: {info['score']} (weight: {info['weight']})")
```

### 7. Human-in-the-Loop Validation

**Interactive Streamlit Interface:**
- Review AI-proposed improvements
- Approve or reject changes individually or in bulk
- Automatic approval for high-confidence changes (configurable threshold)
- Real-time toast notifications
- Interactive data editing with `st.data_editor()`
- Feedback tracking and audit trails

**Features:**
- Auto-approve threshold (default 95%)
- Bulk approval/rejection actions
- Transformation history for reproducibility
- Version tracking for data lineage

### 8. FAIR Compliance

- **Findable:** Complete metadata and cataloging
- **Accessible:** Standard formats and APIs
- **Interoperable:** FHIR, HL7, OMOP support
- **Reusable:** Versioning, documentation, provenance tracking

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│                   User Interface                        │
│           (Streamlit UI / REST API / CLI)               │
└─────────────────────┬──────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────┐
│              Pipeline Orchestrator                      │
│            (DataQualityPipeline)                        │
└──┬──────────────┬──────────────┬──────────────────────┘
   │              │              │
   │              │              │
┌──▼─────────┐ ┌─▼────────────┐ ┌▼───────────┐
│ Precision  │ │ Completeness │ │Reusability │
│   Agent    │ │    Agent     │ │   Agent    │
└──┬─────────┘ └─┬────────────┘ └┬───────────┘
   │              │               │
   └──────────────┴───────────────┘
                  │
    ┌─────────────▼──────────────┐
    │     Coordinator Agent      │
    └─────────────┬──────────────┘
                  │
    ┌─────────────┴──────────────┐
    │                            │
┌───▼──────────┐     ┌───────────▼────────┐
│  Learners    │     │  Rules Engine      │
│  (NEW!)      │     │  & Transformers    │
├──────────────┤     ├────────────────────┤
│ SchemaLearner│     │ DataTransformer    │
│ HealthcareData│     │ ValidationRules   │
│ Detector     │     │ MetricsCalculator  │
└───┬──────────┘     └───────────┬────────┘
    │                            │
    │        ┌───────────────────┘
    │        │
┌───▼────────▼────┐
│  LLM Prompts    │
│  (Healthcare)   │
├─────────────────┤
│ FHIR Analysis   │
│ Schema Learning │
│ Semantic Check  │
└─────────────────┘
```

### Agent Communication Protocol (ACP)

Agents communicate through a standardized message-passing protocol:

- **Request/Response:** Synchronous task delegation
- **Broadcast:** Multi-agent collaboration
- **Asynchronous:** Non-blocking task execution
- **Priority Routing:** Critical tasks processed first

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- Git version control
- 4GB+ RAM recommended
- Optional: GPU for faster ML inference

### Quick Install

```bash
# Clone repository
git clone https://github.com/AgateJarmakovica/AI_Agent_Based_Data_Quality_Framework.git
cd AI_Agent_Based_Data_Quality_Framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Installation Options

#### Option 1: Streamlit UI Only (Quick Start)

Minimal installation for web interface testing:

```bash
pip install -r requirements-streamlit.txt
```

**Use case:** Quick demos, limited resources, basic analysis
**Limitations:** No AI/ML features, rule-based analysis only

#### Option 2: Full Installation (Recommended)

Complete installation with all capabilities:

```bash
# Install full requirements
pip install -r requirements.txt

# Or install as package
pip install -e .

# For development
pip install -e ".[dev]"
```

**Includes:**
- All AI/ML libraries
- Healthcare data model support (FHIR, HL7, OMOP)
- LLM integration capabilities
- Advanced schema learning
- Complete metric calculations

**Requirements:** ~3GB disk space, 4GB+ RAM

> **Note:** The application auto-detects available dependencies. Missing ML libraries trigger demo mode with basic rule-based analysis.

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Configure with your settings
nano .env
```

**Environment Variables:**

```env
# LLM Configuration (optional for advanced features)
OPENAI_API_KEY=your_key_here
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4

# Logging
LOG_LEVEL=INFO

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/healthdq

# FHIR Server (optional)
FHIR_SERVER_URL=https://hapi.fhir.org/baseR4
```

---

## Usage

### Command Line Interface

```bash
# Basic analysis
healthdq analyze data.csv --dimensions precision,completeness,reusability

# With healthcare model detection
healthdq analyze data.csv --detect-model --learn-schema

# Custom configuration
healthdq analyze data.csv --config configs/custom.yml

# Export results
healthdq analyze data.csv --output results/ --format json
```

### Python API

#### Complete Adaptive Pipeline

```python
import asyncio
from healthdq.pipeline import DataQualityPipeline
from healthdq.learners import SchemaLearner, HealthcareDataDetector

async def main():
    # Load data
    pipeline = DataQualityPipeline()
    data = pipeline.load_data("patient_data.csv")

    # Step 1: Detect healthcare model
    detector = HealthcareDataDetector()
    model_info = detector.detect(data)
    print(f"Detected model: {model_info['model_type']}")
    print(f"Confidence: {model_info['confidence']}")

    # Step 2: Learn schema
    learner = SchemaLearner()
    schema = learner.learn(data, schema_name="patient_encounters")
    print(f"Schema confidence: {schema['confidence']}")

    # Step 3: Run quality analysis
    results = await pipeline.run(
        source=data,
        quality_dimensions=["precision", "completeness", "reusability"],
        apply_improvements=True,
        require_hitl=False,
        learned_schema=schema,
        model_info=model_info
    )

    # Step 4: Calculate publication DQ score
    from healthdq.metrics import MetricsCalculator
    calculator = MetricsCalculator()
    dq_score = calculator.calculate_publication_dq_score(
        results['data'],
        original_data=data
    )

    print(f"DQ Total: {dq_score['dq_total']}")
    print(f"Improvement: {dq_score.get('improvement', {})}")

    # Step 5: Validate new data against learned schema
    new_data = pipeline.load_data("new_patient_data.csv")
    validation = learner.validate_against_schema(new_data, schema)
    print(f"Validation compliance: {validation['compliance_score']}")

    return results

if __name__ == "__main__":
    results = asyncio.run(main())
```

#### Using LLM for Semantic Analysis

```python
from healthdq.prompts import load_prompt
import openai

# Load appropriate prompt based on detected model
prompt = load_prompt("fhir_analysis")  # or "semantic_analysis"

# Analyze with LLM
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps({
            "data_type": "observation",
            "records": data.to_dict('records'),
            "context": {
                "care_setting": "ambulatory",
                "clinical_domain": "cardiology"
            }
        })}
    ]
)

findings = json.loads(response.choices[0].message.content)
print(f"Semantic quality score: {findings['semantic_quality_score']}")
for finding in findings['findings']:
    print(f"[{finding['severity']}] {finding['finding']}")
```

### Web UI (Streamlit)

```bash
# Start Streamlit application
streamlit run src/healthdq/ui/streamlit_app.py

# Or use shortcut
healthdq-ui
```

**Access at:** `http://localhost:8501`

**Features:**
- Interactive data upload (CSV, Excel, JSON, Parquet)
- Automatic healthcare model detection
- Visual quality assessment dashboards
- Human-in-the-loop approval workflow
- Auto-approve with configurable confidence threshold
- Interactive data editing
- Real-time progress notifications
- Export improved data

### REST API

```bash
# Start FastAPI server
uvicorn healthdq.api.server:app --reload

# Or use shortcut
healthdq-api
```

**API Documentation:** `http://localhost:8000/docs`

**Example Request:**

```python
import requests

# Upload data
files = {'file': open('patient_data.csv', 'rb')}
response = requests.post('http://localhost:8000/api/upload', files=files)
file_id = response.json()['file_id']

# Analyze with healthcare model detection
response = requests.post(
    'http://localhost:8000/api/analyze',
    json={
        'file_id': file_id,
        'dimensions': ['precision', 'completeness', 'reusability'],
        'detect_model': True,
        'learn_schema': True,
        'apply_improvements': True
    }
)

results = response.json()
print(f"Model detected: {results['model_info']['model_type']}")
print(f"DQ score: {results['quality_score']}")
```

---

## Healthcare Data Model Support

### FHIR (Fast Healthcare Interoperability Resources)

**Supported Resources:**
- Patient, Observation, Condition, MedicationRequest, Procedure, Encounter

**Validation:**
```python
from healthdq.learners import HealthcareDataDetector

detector = HealthcareDataDetector()

# Validate FHIR resource
fhir_resource = {...}  # FHIR JSON
validation = detector.validate_fhir_resource(fhir_resource, "Patient")

if validation['valid']:
    print("FHIR resource is compliant with R4 specification")
else:
    for error in validation['errors']:
        print(f"Error: {error}")
```

**Mapping Suggestions:**
```python
# Get mapping recommendations
mappings = detector.suggest_mappings(dataframe, target_model="fhir")
# Returns: {'patient_id': 'Patient.identifier', 'gender': 'Patient.gender', ...}
```

### HL7 v2.x

**Supported Segments:**
MSH (Message Header), PID (Patient Identification), PV1 (Patient Visit), OBR (Observation Request), OBX (Observation Result), DG1 (Diagnosis), PR1 (Procedures)

**Detection:**
```python
model_info = detector.detect(hl7_dataframe)
if model_info['model_type'] == 'hl7':
    print(f"HL7 segments detected")
```

### OMOP CDM (Observational Medical Outcomes Partnership)

**Supported Tables:**
- person, observation_period, visit_occurrence
- condition_occurrence, drug_exposure, measurement
- procedure_occurrence, device_exposure

**Mapping:**
```python
mappings = detector.suggest_mappings(dataframe, target_model="omop")
# Returns mappings to OMOP CDM fields
```

### Medical Coding Systems

Automatic detection and validation of:
- **SNOMED CT** — Clinical terminology
- **LOINC** — Laboratory observations
- **ICD-10** — Diagnosis codes
- **CPT** — Procedural terminology
- **RxNorm** — Medication codes

---

## Documentation

### Project Structure

```
healthdq-ai/
├── configs/             # Configuration files
│   ├── agents.yml       # Agent settings
│   ├── rules.yml        # Quality rules
│   └── prompts.yml      # LLM prompt templates
├── data/                # Sample data and outputs
├── docs/                # Documentation
├── examples/            # Usage examples
│   └── adaptive_learning_demo.py
├── src/healthdq/        # Main source package
│   ├── agents/          # Multi-agent system
│   │   ├── base_agent.py
│   │   └── coordinator.py
│   ├── communication/   # Agent messaging protocol
│   ├── learners/        # NEW: Adaptive learning
│   │   ├── schema_learner.py
│   │   └── healthcare_detector.py
│   ├── loaders/         # Data ingestion
│   ├── metrics/         # Quality calculations
│   │   └── calculator.py  # Publication formulas
│   ├── prompts/         # NEW: LLM prompts
│   │   ├── fhir_analysis.md
│   │   ├── schema_learning.md
│   │   └── semantic_analysis.md
│   ├── rules/           # Transformation engine
│   │   └── transform.py   # Enhanced with z-score, MAD
│   ├── schema/          # Schema validation
│   ├── ui/              # Web interface
│   │   └── streamlit_app.py  # Enhanced UI
│   └── api/             # REST API
├── tests/               # Test suite
│   ├── unit/
│   └── integration/
└── scripts/             # Utility scripts
```

### Configuration Files

**agents.yml** — Agent configuration
```yaml
coordinator:
  max_concurrent_agents: 5
  timeout: 300
  enable_learning: true

precision_agent:
  outlier_method: iqr  # or zscore, mad
  threshold: 1.5
```

**rules.yml** — Data quality rules
```yaml
completeness:
  missing_threshold: 0.1
  imputation_methods: [knn, mean, median, mode]

accuracy:
  anomaly_detection: [isolation_forest, lof]
  contamination: 0.1
```

**.env** — Environment configuration
```env
OPENAI_API_KEY=sk-...
LOG_LEVEL=INFO
DATABASE_URL=postgresql://...
```

### Examples and Demos

**Complete Demo Script:**
```bash
python examples/adaptive_learning_demo.py
```

**Includes:**
1. Healthcare Data Model Detection
2. Automatic Schema Learning with ML
3. LLM-Powered Analysis Integration
4. Complete Adaptive Pipeline
5. FHIR Resource Validation

**Jupyter Notebooks:**
- `notebooks/demo.ipynb` — Basic tutorial
- `notebooks/experiments.ipynb` — Advanced features
- `notebooks/healthcare_models.ipynb` — FHIR/OMOP examples

---

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/healthdq --cov-report=html tests/

# Run specific test module
pytest tests/test_learners.py

# Run integration tests
pytest tests/integration/

# Run with verbose output
pytest -v tests/
```

### Test Structure

```
tests/
├── unit/
│   ├── test_agents.py
│   ├── test_learners.py
│   ├── test_metrics.py
│   └── test_transforms.py
├── integration/
│   ├── test_pipeline.py
│   ├── test_healthcare_detection.py
│   └── test_adaptive_learning.py
└── fixtures/
    ├── sample_data/
    └── test_configs/
```

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Code style and standards
- Pull request process
- Issue reporting
- Documentation requirements

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

# Run security checks
bandit -r src/
```

### Code Quality Standards

- **Style:** Black formatter, 88-character line length
- **Import sorting:** isort
- **Type hints:** mypy static type checking
- **Linting:** flake8
- **Testing:** pytest with >80% coverage
- **Documentation:** Google-style docstrings

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.

---

## Contact

**Author:** Agate Jarmakoviča
**Institution:** PhD Research Prototype
**GitHub:** [@AgateJarmakovica](https://github.com/AgateJarmakovica)
**Project:** [AI_Agent_Based_Data_Quality_Framework](https://github.com/AgateJarmakovica/AI_Agent_Based_Data_Quality_Framework)

For questions, issues, or collaboration inquiries, please open an issue on GitHub.

---

## Acknowledgments

This framework is based on research in:

- **Data-Centric AI** — Andrew Ng, Stanford University, 2021
- **FAIR Principles** — Wilkinson et al., 2016
- **Agent Communication Standards** — FIPA (Foundation for Intelligent Physical Agents)
- **Human-in-the-Loop Machine Learning** — Amershi et al., Microsoft Research, 2014
- **ISO 25024** — Data Quality Model Standard
- **FHIR R4** — HL7 Fast Healthcare Interoperability Resources
- **OMOP CDM** — Observational Health Data Sciences and Informatics

---

## Project Status

| Feature | Status |
|---------|--------|
| Core Framework | Implemented |
| Multi-Agent System | Operational |
| Adaptive Schema Learning | Implemented |
| Healthcare Model Detection | Implemented |
| LLM Integration | Implemented |
| Human-in-the-Loop Interface | Operational |
| FHIR Support | Basic validation available |
| HL7 v2.x Support | Detection implemented |
| OMOP CDM Support | Detection implemented |
| Advanced ML Imputation | In Progress |
| Real-time Streaming | Planned |
| Full FHIR R4 Compliance | Planned |

---

**Version:** 2.0.0
**Last Updated:** 2025-11-06
**Status:** Beta
**Maturity:** Research Prototype

---

For more information, visit the [project documentation](docs/) or explore [usage examples](examples/).
