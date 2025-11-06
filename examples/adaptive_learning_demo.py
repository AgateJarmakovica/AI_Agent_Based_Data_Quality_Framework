"""
Adaptive Learning Demo for healthdq-ai
Author: Agate Jarmakoviča

Demonstrates the AI-powered adaptive data quality layer for healthcare data:
1. Automatic healthcare data model detection (FHIR, HL7, OMOP, EHR)
2. Schema learning from data
3. LLM-powered semantic analysis
4. Dynamic validation rule adaptation
"""

import pandas as pd
import asyncio
from pathlib import Path


# Example 1: Automatic Healthcare Data Model Detection
def demo_healthcare_detection():
    """Detect healthcare data models automatically."""
    print("=" * 70)
    print("DEMO 1: Healthcare Data Model Detection")
    print("=" * 70)

    from healthdq.learners import HealthcareDataDetector

    # Sample patient data (generic EHR format)
    patient_data = pd.DataFrame({
        "patient_id": [1001, 1002, 1003],
        "name": ["John Doe", "Jane Smith", "Bob Wilson"],
        "age": [45, 32, 67],
        "gender": ["male", "female", "male"],
        "diagnosis_code": ["I10", "E11.9", "J45.909"],
        "diagnosis_name": ["Hypertension", "Type 2 Diabetes", "Asthma"],
    })

    # Detect healthcare model
    detector = HealthcareDataDetector()
    detection_result = detector.detect(patient_data)

    print(f"\n✓ Detected Model: {detection_result['model_type']}")
    print(f"✓ Confidence: {detection_result['confidence'] * 100:.1f}%")
    print(f"✓ Standards Detected: {', '.join(detection_result['detected_standards'])}")

    if detection_result['detected_codes']:
        print("\n✓ Medical Coding Systems Found:")
        for code_info in detection_result['detected_codes']:
            print(f"  - {code_info['system']} in column '{code_info['column']}' "
                  f"(match rate: {code_info['match_rate'] * 100:.0f}%)")

    print("\n✓ Recommendations:")
    for rec in detection_result['recommendations']:
        print(f"  - {rec}")

    # Suggest mappings to FHIR
    print("\n✓ Suggested Mappings to FHIR:")
    mappings = detector.suggest_mappings(patient_data, target_model="fhir")
    for source, target in mappings.items():
        print(f"  {source} → {target}")

    print()


# Example 2: Automatic Schema Learning
def demo_schema_learning():
    """Learn data schema automatically using ML."""
    print("=" * 70)
    print("DEMO 2: Automatic Schema Learning")
    print("=" * 70)

    from healthdq.learners import SchemaLearner

    # Sample clinical observations
    observations = pd.DataFrame({
        "observation_id": range(1, 101),
        "patient_id": [f"P{i:04d}" for i in range(1, 101)],
        "systolic_bp": [120 + (i % 40) for i in range(100)],
        "diastolic_bp": [80 + (i % 20) for i in range(100)],
        "heart_rate": [70 + (i % 30) for i in range(100)],
        "temperature": [36.5 + (i % 2) * 0.5 for i in range(100)],
        "observation_date": pd.date_range("2024-01-01", periods=100),
    })

    # Learn schema
    learner = SchemaLearner()
    learned_schema = learner.learn(
        observations,
        schema_name="clinical_observations",
        include_statistics=True
    )

    print(f"\n✓ Schema Name: {learned_schema['schema_name']}")
    print(f"✓ Confidence Score: {learned_schema['confidence']}")
    print(f"✓ Columns Analyzed: {learned_schema['num_columns']}")

    print("\n✓ Learned Column Information:")
    for col_name, col_schema in learned_schema['columns'].items():
        print(f"\n  Column: {col_name}")
        print(f"    - Inferred Type: {col_schema['inferred_type']}")
        print(f"    - Nullable: {col_schema['nullable']}")
        print(f"    - Missing: {col_schema['missing_percentage']:.1f}%")

        if col_schema.get('healthcare_field'):
            hc_field = col_schema['healthcare_field']
            print(f"    - Healthcare Field: {hc_field['field_type']} "
                  f"(confidence: {hc_field['confidence'] * 100:.0f}%)")

        if 'constraints' in col_schema:
            constraints = col_schema['constraints']
            if 'min' in constraints and 'max' in constraints:
                print(f"    - Range: [{constraints['min']:.1f}, {constraints['max']:.1f}]")

    if learned_schema['column_groups']:
        print("\n✓ Discovered Column Groups (similar behavior):")
        for group_name, columns in learned_schema['column_groups'].items():
            print(f"  {group_name}: {', '.join(columns)}")

    print("\n✓ Detected Patterns:")
    patterns = learned_schema['patterns']
    print(f"  - Temporal columns: {len(patterns['temporal_columns'])}")
    print(f"  - Identifier columns: {len(patterns['identifier_columns'])}")
    print(f"  - Categorical columns: {len(patterns['categorical_columns'])}")

    # Validate new data against learned schema
    print("\n✓ Validating New Data Against Learned Schema:")
    new_data = pd.DataFrame({
        "observation_id": [101, 102],
        "patient_id": ["P0101", "P0102"],
        "systolic_bp": [135, 200],  # 200 is outlier
        "diastolic_bp": [85, 90],
        "heart_rate": [75, 80],
        "temperature": [36.8, 37.2],
        "observation_date": ["2024-04-11", "2024-04-12"],
    })

    validation = learner.validate_against_schema(new_data, learned_schema)
    print(f"  Valid: {validation['valid']}")
    print(f"  Compliance Score: {validation['compliance_score'] * 100:.0f}%")

    if validation['issues']:
        print("  Issues Found:")
        for issue in validation['issues']:
            print(f"    - [{issue['severity']}] {issue['description']}")

    print()


# Example 3: LLM-Powered FHIR Analysis
def demo_llm_prompts():
    """Use LLM prompts for advanced semantic analysis."""
    print("=" * 70)
    print("DEMO 3: LLM-Powered Healthcare Data Analysis")
    print("=" * 70)

    from healthdq.prompts import load_prompt, list_available_prompts

    # List available prompts
    print("\n✓ Available LLM Prompts:")
    prompts = list_available_prompts()
    for prompt_name in prompts:
        print(f"  - {prompt_name}")

    # Load FHIR analysis prompt
    print("\n✓ Loading FHIR Analysis Prompt...")
    fhir_prompt = load_prompt("fhir_analysis")
    print(f"  Prompt length: {len(fhir_prompt)} characters")
    print(f"  First 200 chars: {fhir_prompt[:200]}...")

    # Load schema learning prompt
    print("\n✓ Loading Schema Learning Prompt...")
    schema_prompt = load_prompt("schema_learning")
    print(f"  Prompt length: {len(schema_prompt)} characters")

    # Load semantic analysis prompt
    print("\n✓ Loading Semantic Analysis Prompt...")
    semantic_prompt = load_prompt("semantic_analysis")
    print(f"  Prompt length: {len(semantic_prompt)} characters")

    print("\n✓ These prompts can be used with:")
    print("  - OpenAI GPT-4")
    print("  - Anthropic Claude")
    print("  - Local LLMs (Llama, Mistral)")
    print("  - BioClinicalBERT for medical text")

    print("\n✓ Example Integration (pseudo-code):")
    print("""
    import openai
    from healthdq.prompts import load_prompt

    # Load prompt for FHIR analysis
    prompt = load_prompt("fhir_analysis")

    # Analyze FHIR resource
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(fhir_resource)}
        ]
    )

    # Get structured findings
    findings = json.loads(response.choices[0].message.content)
    print(findings['quality_score'])
    """)

    print()


# Example 4: Complete Adaptive Pipeline
async def demo_adaptive_pipeline():
    """Complete adaptive data quality pipeline."""
    print("=" * 70)
    print("DEMO 4: Complete Adaptive Data Quality Pipeline")
    print("=" * 70)

    from healthdq.learners import SchemaLearner, HealthcareDataDetector
    from healthdq.metrics.calculator import MetricsCalculator

    # Sample healthcare data
    healthcare_data = pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(1, 51)],
        "encounter_id": [f"E{i:04d}" for i in range(1, 51)],
        "age": [30 + i for i in range(50)],
        "gender": ["male" if i % 2 == 0 else "female" for i in range(50)],
        "diagnosis_icd10": ["I10", "E11.9", "J45.909"] * 16 + ["I10", "E11.9"],
        "systolic_bp": [120 + (i % 40) for i in range(50)],
        "diastolic_bp": [80 + (i % 20) for i in range(50)],
        "medication": ["Lisinopril", "Metformin", "Albuterol"] * 16 + ["Lisinopril", "Metformin"],
    })

    # Add some missing values for completeness analysis
    healthcare_data.loc[2:4, "gender"] = None
    healthcare_data.loc[10:12, "medication"] = None

    print("\n✓ Step 1: Detect Healthcare Data Model")
    detector = HealthcareDataDetector()
    model_info = detector.detect(healthcare_data)
    print(f"  Detected: {model_info['model_type']} (confidence: {model_info['confidence'] * 100:.0f}%)")

    print("\n✓ Step 2: Learn Data Schema")
    learner = SchemaLearner()
    schema = learner.learn(healthcare_data, schema_name="patient_encounters")
    print(f"  Schema confidence: {schema['confidence'] * 100:.0f}%")
    print(f"  Column groups discovered: {len(schema['column_groups'])}")

    print("\n✓ Step 3: Calculate Data Quality Metrics (Publication Formulas)")
    calculator = MetricsCalculator()

    # Calculate publication DQ score
    dq_result = calculator.calculate_publication_dq_score(healthcare_data)

    print(f"  DQ Total Score: {dq_result['dq_total'] * 100:.1f}%")
    print(f"  Formula: {dq_result['formula']}")
    print("\n  Dimension Breakdown:")
    for dimension, info in dq_result['dimensions'].items():
        print(f"    {dimension.title()}: {info['score'] * 100:.1f}% "
              f"(weight: {info['weight']}, contribution: {info['contribution']:.3f})")

    print("\n✓ Step 4: Export Learned Schema")
    output_path = "/tmp/learned_schema.json"
    learner.export_schema("patient_encounters", output_path, format="json")
    print(f"  Schema exported to: {output_path}")

    print("\n✓ Step 5: Generate Recommendations")
    print("  Based on learned schema and detected model:")
    print("  - Standardize column names to FHIR Patient resource")
    print("  - Validate ICD-10 codes against terminology server")
    print("  - Check medication-condition relationships")
    print("  - Implement age-gender-specific validation rules")
    print("  - Add metadata for FAIR compliance")

    print("\n✓ Pipeline Complete!")
    print("  The system has:")
    print("  ✓ Automatically detected the healthcare data model")
    print("  ✓ Learned the schema and constraints")
    print("  ✓ Calculated publication-accurate quality metrics")
    print("  ✓ Generated actionable recommendations")

    print()


# Example 5: Real-World FHIR Validation
def demo_fhir_validation():
    """Validate FHIR resources."""
    print("=" * 70)
    print("DEMO 5: FHIR Resource Validation")
    print("=" * 70)

    from healthdq.learners import HealthcareDataDetector

    # Sample FHIR Patient resource
    fhir_patient = {
        "resourceType": "Patient",
        "id": "example-001",
        "identifier": [{
            "system": "urn:oid:1.2.3.4.5",
            "value": "12345"
        }],
        "name": [{
            "family": "Doe",
            "given": ["John"]
        }],
        "gender": "male",
        "birthDate": "1980-01-15",
    }

    print("\n✓ FHIR Patient Resource:")
    import json
    print(json.dumps(fhir_patient, indent=2))

    detector = HealthcareDataDetector()

    # Validate FHIR resource
    print("\n✓ Validating FHIR Resource...")
    validation = detector.validate_fhir_resource(fhir_patient, "Patient")

    print(f"  Valid: {validation['valid']}")

    if validation['errors']:
        print("  Errors:")
        for error in validation['errors']:
            print(f"    - {error}")

    if validation['warnings']:
        print("  Warnings:")
        for warning in validation['warnings']:
            print(f"    - {warning}")

    if validation['valid']:
        print("  ✓ FHIR resource is valid and compliant!")

    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "healthdq-ai: Adaptive Learning Demo" + " " * 22 + "║")
    print("║" + " " * 15 + "AI-Powered Healthcare Data Quality" + " " * 19 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        # Run demos
        demo_healthcare_detection()
        input("Press Enter to continue to Schema Learning demo...")

        demo_schema_learning()
        input("Press Enter to continue to LLM Prompts demo...")

        demo_llm_prompts()
        input("Press Enter to continue to Adaptive Pipeline demo...")

        asyncio.run(demo_adaptive_pipeline())
        input("Press Enter to continue to FHIR Validation demo...")

        demo_fhir_validation()

        print("=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)
        print()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
