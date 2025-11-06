# Healthcare Data Quality Prompts

This directory contains LLM prompts for AI-driven healthcare data analysis and quality assessment.

## Available Prompts

### 1. FHIR Analysis (`fhir_analysis.md`)
**Purpose**: Validate and analyze FHIR resources (Patient, Observation, Condition, etc.)

**Use Cases**:
- FHIR resource validation
- Interoperability compliance
- Code system verification
- Semantic anomaly detection

**Integration Example**:
```python
from healthdq.prompts import load_prompt

prompt = load_prompt("fhir_analysis")
response = llm.generate(prompt, data=fhir_resource)
```

### 2. Schema Learning (`schema_learning.md`)
**Purpose**: Automatically infer data schemas and healthcare data models

**Use Cases**:
- Unknown data structure analysis
- Healthcare model detection (FHIR, OMOP, HL7)
- Schema constraint discovery
- Mapping suggestions

**Integration Example**:
```python
from healthdq.learners import SchemaLearner

learner = SchemaLearner()
schema = learner.learn(dataframe)

# Use LLM for advanced semantic analysis
prompt = load_prompt("schema_learning")
enhanced_schema = llm.generate(prompt, schema=schema)
```

### 3. Semantic Analysis (`semantic_analysis.md`)
**Purpose**: Deep clinical and semantic validation of healthcare data

**Use Cases**:
- Clinical plausibility checks
- Medical terminology validation
- Diagnosis-procedure-medication relationships
- Age/gender-specific validation
- Physiological range checking

**Integration Example**:
```python
from healthdq.agents import SemanticAgent

agent = SemanticAgent(prompt_file="semantic_analysis")
results = await agent.analyze(clinical_data)
```

## Prompt Structure

All prompts follow a consistent structure:

1. **Role Definition**: Who the AI agent is
2. **Mission**: What it needs to accomplish
3. **Input Format**: Expected data structure
4. **Output Format**: Required response format (usually JSON)
5. **Validation Rules**: Domain-specific rules and constraints
6. **Examples**: Concrete input/output examples
7. **Context Awareness**: How to adapt analysis to context
8. **Remember**: Key principles and priorities

## Using Prompts with LLMs

### With OpenAI GPT Models
```python
import openai
from healthdq.prompts import load_prompt

prompt_template = load_prompt("semantic_analysis")
prompt = prompt_template.format(data=clinical_records)

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(clinical_records)}
    ]
)

findings = json.loads(response.choices[0].message.content)
```

### With Anthropic Claude
```python
import anthropic
from healthdq.prompts import load_prompt

client = anthropic.Anthropic()
prompt = load_prompt("fhir_analysis")

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    messages=[
        {"role": "user", "content": f"{prompt}\n\nData: {json.dumps(fhir_data)}"}
    ]
)

analysis = json.loads(message.content[0].text)
```

### With Local Models (Llama, etc.)
```python
from transformers import pipeline
from healthdq.prompts import load_prompt

generator = pipeline("text-generation", model="meta-llama/Llama-2-70b-chat-hf")
prompt = load_prompt("schema_learning")

result = generator(
    f"{prompt}\n\nSchema:\n{json.dumps(schema_data)}",
    max_new_tokens=2048,
    temperature=0.7
)

schema_analysis = json.loads(result[0]["generated_text"])
```

### With BioClinicalBERT (Domain-Specific)
```python
from transformers import AutoTokenizer, AutoModel
from healthdq.prompts import load_prompt

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

prompt = load_prompt("semantic_analysis")
# Use for embeddings and semantic similarity
```

## Customizing Prompts

You can extend or customize prompts for your specific use case:

```python
from healthdq.prompts import PromptTemplate

# Load base prompt
base_prompt = PromptTemplate.load("semantic_analysis")

# Add custom validation rules
custom_rules = """
## Custom Institutional Rules
- All blood pressures >160 mmHg require physician review
- Medication orders require pharmacist verification
- Lab critical values trigger automatic notifications
"""

enhanced_prompt = base_prompt + custom_rules
```

## Best Practices

### 1. Prompt Selection
- **FHIR Analysis**: When working with FHIR-formatted data or interoperability
- **Schema Learning**: When encountering new/unknown healthcare data structures
- **Semantic Analysis**: For deep clinical validation and plausibility checks

### 2. Data Preparation
- Provide sufficient context (patient age, gender, care setting)
- Include relevant metadata
- Sample representative records
- Specify the clinical domain

### 3. Response Handling
- Always parse JSON responses safely
- Validate response structure
- Handle confidence scores
- Log findings for audit trails

### 4. Iterative Refinement
- Start with broad analysis
- Drill down into specific issues
- Re-run with refined context
- Validate findings with domain experts

### 5. Performance Optimization
- Batch similar records
- Cache prompt templates
- Use streaming for large datasets
- Implement rate limiting for API calls

## Integration with healthdq Framework

```python
from healthdq.pipeline import DataQualityPipeline
from healthdq.learners import SchemaLearner, HealthcareDataDetector
from healthdq.agents import CoordinatorAgent

# Initialize pipeline
pipeline = DataQualityPipeline()

# Load data
data = pipeline.load_data("patient_records.csv")

# Auto-detect healthcare model
detector = HealthcareDataDetector()
model_info = detector.detect(data)
print(f"Detected: {model_info['model_type']}")

# Learn schema
learner = SchemaLearner()
schema = learner.learn(data)

# Run AI-powered analysis with appropriate prompt
if model_info['model_type'] == 'fhir':
    prompt_file = "fhir_analysis"
elif model_info['model_type'] == 'omop':
    prompt_file = "semantic_analysis"
else:
    prompt_file = "schema_learning"

# Run quality analysis with learned schema
coordinator = CoordinatorAgent(prompt_file=prompt_file)
results = await coordinator.analyze(
    data,
    dimensions=["precision", "completeness", "semantic_integrity"],
    learned_schema=schema
)
```

## Prompt Versioning

Prompts are versioned to track improvements:

```
prompts/
  fhir_analysis.md          # Latest version
  fhir_analysis.v1.md       # Version 1 (deprecated)
  schema_learning.md        # Latest version
  semantic_analysis.md      # Latest version
```

Always use the latest version unless you need reproducibility with a specific version.

## Contributing

To contribute new prompts or improve existing ones:

1. Follow the standard prompt structure
2. Include comprehensive examples
3. Add validation rules and reference ranges
4. Test with real healthcare data
5. Document use cases and integration examples

## References

- [FHIR Specification](http://hl7.org/fhir/)
- [OMOP CDM](https://ohdsi.github.io/CommonDataModel/)
- [SNOMED CT](https://www.snomed.org/)
- [LOINC](https://loinc.org/)
- [HL7 Standards](http://www.hl7.org/)

## Support

For questions or issues with prompts:
- Open an issue on GitHub
- Check documentation at docs/prompts/
- Join community discussions
