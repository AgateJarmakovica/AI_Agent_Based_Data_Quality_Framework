"""
Healthcare Data Model Detector
Author: Agate Jarmakoviča

Automatically detects and classifies healthcare data models:
- FHIR (Fast Healthcare Interoperability Resources)
- HL7 v2.x messages
- OMOP CDM (Observational Medical Outcomes Partnership)
- Generic EHR (Electronic Health Records)
- Clinical codes (SNOMED, LOINC, ICD, CPT)
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import json
from datetime import datetime

from healthdq.utils.logger import get_logger

logger = get_logger(__name__)


class HealthcareDataDetector:
    """
    Detects healthcare data models and standards in datasets.

    Uses pattern matching, column name analysis, and value inspection
    to identify the underlying healthcare data model.

    Example:
        detector = HealthcareDataDetector()
        model_info = detector.detect(dataframe)
        print(f"Detected: {model_info['model_type']}")
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the healthcare data detector."""
        from healthdq.config import get_config

        self.config = config or get_config()
        self.detection_history: List[Dict[str, Any]] = []

        # Define healthcare model signatures
        self._load_model_signatures()

    def _load_model_signatures(self) -> None:
        """Load signature patterns for different healthcare data models."""

        # FHIR resource signatures
        self.fhir_signatures = {
            "Patient": ["id", "identifier", "name", "gender", "birthDate", "address"],
            "Observation": ["resourceType", "status", "code", "subject", "value"],
            "Condition": ["clinicalStatus", "verificationStatus", "category", "code", "subject"],
            "MedicationRequest": ["status", "intent", "medication", "subject", "authoredOn"],
            "Procedure": ["status", "code", "subject", "performed"],
            "Encounter": ["status", "class", "type", "subject", "period"],
        }

        # OMOP CDM table signatures
        self.omop_signatures = {
            "person": ["person_id", "gender_concept_id", "year_of_birth", "race_concept_id"],
            "observation_period": ["person_id", "observation_period_start_date", "observation_period_end_date"],
            "visit_occurrence": ["visit_occurrence_id", "person_id", "visit_concept_id", "visit_start_date"],
            "condition_occurrence": ["condition_occurrence_id", "person_id", "condition_concept_id", "condition_start_date"],
            "drug_exposure": ["drug_exposure_id", "person_id", "drug_concept_id", "drug_exposure_start_date"],
            "measurement": ["measurement_id", "person_id", "measurement_concept_id", "measurement_date"],
        }

        # HL7 v2.x segment patterns
        self.hl7_segments = ["MSH", "PID", "PV1", "OBR", "OBX", "DG1", "PR1"]

        # Medical coding system patterns
        self.coding_patterns = {
            "ICD-10": r"^[A-Z]\d{2}\.?\d{0,2}$",
            "ICD-9": r"^\d{3}\.?\d{0,2}$",
            "SNOMED": r"^\d{6,18}$",
            "LOINC": r"^\d{4,5}-\d{1}$",
            "CPT": r"^\d{5}$",
            "RxNorm": r"^\d{6,8}$",
        }

    def detect(
        self,
        data: pd.DataFrame,
        include_confidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Detect healthcare data model from DataFrame.

        Args:
            data: DataFrame to analyze
            include_confidence: Include confidence scores

        Returns:
            Detection results with model type, confidence, and metadata
        """
        logger.info(f"Detecting healthcare data model from shape {data.shape}")

        results = {
            "model_type": "unknown",
            "sub_type": None,
            "confidence": 0.0,
            "detected_standards": [],
            "detected_codes": [],
            "recommendations": [],
            "metadata": {
                "detected_at": datetime.now().isoformat(),
                "num_rows": len(data),
                "num_columns": len(data.columns),
            }
        }

        # Try each detection method
        fhir_result = self._detect_fhir(data)
        omop_result = self._detect_omop(data)
        hl7_result = self._detect_hl7(data)
        ehr_result = self._detect_ehr(data)

        # Select best match
        detection_results = [
            ("fhir", fhir_result),
            ("omop", omop_result),
            ("hl7", hl7_result),
            ("ehr", ehr_result),
        ]

        best_match = max(detection_results, key=lambda x: x[1]["confidence"])

        if best_match[1]["confidence"] > 0.3:
            results["model_type"] = best_match[0]
            results["sub_type"] = best_match[1].get("sub_type")
            results["confidence"] = best_match[1]["confidence"]
            results["recommendations"] = best_match[1].get("recommendations", [])

        # Detect medical coding systems
        results["detected_codes"] = self._detect_coding_systems(data)

        # Detect healthcare-specific fields
        results["detected_standards"] = self._detect_healthcare_fields(data)

        # Store detection history
        self.detection_history.append({
            "timestamp": datetime.now().isoformat(),
            "model_type": results["model_type"],
            "confidence": results["confidence"],
            "num_rows": len(data),
        })

        logger.info(f"Detected model: {results['model_type']} (confidence: {results['confidence']:.2f})")

        return results

    def _detect_fhir(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect FHIR resources."""
        result = {
            "confidence": 0.0,
            "sub_type": None,
            "recommendations": [],
        }

        columns = set(col.lower() for col in data.columns)

        # Check for FHIR resource type indicators
        best_match = None
        best_score = 0.0

        for resource_type, signature in self.fhir_signatures.items():
            signature_lower = [s.lower() for s in signature]
            matches = sum(1 for field in signature_lower if field in columns)
            score = matches / len(signature)

            if score > best_score:
                best_score = score
                best_match = resource_type

        # Check for FHIR-specific fields
        fhir_indicators = ["resourcetype", "identifier", "meta", "extension"]
        fhir_score = sum(1 for indicator in fhir_indicators if indicator in columns) / len(fhir_indicators)

        # Combine scores
        result["confidence"] = (best_score * 0.7) + (fhir_score * 0.3)

        if best_match and result["confidence"] > 0.3:
            result["sub_type"] = best_match
            result["recommendations"] = [
                f"Consider using fhir.resources.{best_match.lower()} for validation",
                "Ensure compliance with FHIR R4 specification",
                "Validate identifier formats and coding systems",
            ]

        return result

    def _detect_omop(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect OMOP CDM tables."""
        result = {
            "confidence": 0.0,
            "sub_type": None,
            "recommendations": [],
        }

        columns = set(col.lower() for col in data.columns)

        # Check for OMOP table signatures
        best_match = None
        best_score = 0.0

        for table_name, signature in self.omop_signatures.items():
            signature_lower = [s.lower() for s in signature]
            matches = sum(1 for field in signature_lower if field in columns)
            score = matches / len(signature)

            if score > best_score:
                best_score = score
                best_match = table_name

        # Check for OMOP-specific patterns
        omop_patterns = ["_concept_id", "_date", "_datetime", "_id"]
        pattern_matches = sum(1 for col in columns if any(p in col for p in omop_patterns))
        pattern_score = min(1.0, pattern_matches / len(data.columns))

        # Combine scores
        result["confidence"] = (best_score * 0.7) + (pattern_score * 0.3)

        if best_match and result["confidence"] > 0.3:
            result["sub_type"] = best_match
            result["recommendations"] = [
                f"Detected OMOP CDM table: {best_match}",
                "Validate against OMOP CDM v5.4 specification",
                "Ensure concept_id fields reference valid concepts",
                "Check date formats conform to OMOP standards",
            ]

        return result

    def _detect_hl7(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect HL7 v2.x message structure."""
        result = {
            "confidence": 0.0,
            "sub_type": None,
            "recommendations": [],
        }

        # HL7 is typically not in DataFrame format, but check for parsed segments
        columns = set(col.upper() for col in data.columns)

        # Check for HL7 segment names
        segment_matches = sum(1 for seg in self.hl7_segments if seg in columns)
        result["confidence"] = segment_matches / len(self.hl7_segments)

        # Check for HL7 field patterns (PID-3, OBX-5, etc.)
        hl7_field_pattern = any("-" in col and col.split("-")[0] in self.hl7_segments for col in columns)

        if hl7_field_pattern:
            result["confidence"] = max(result["confidence"], 0.6)
            result["recommendations"] = [
                "Consider using hl7apy for HL7 v2.x message parsing",
                "Validate message structure against HL7 specifications",
                "Check segment order and cardinality",
            ]

        return result

    def _detect_ehr(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect generic EHR structure."""
        result = {
            "confidence": 0.0,
            "sub_type": "generic_ehr",
            "recommendations": [],
        }

        columns = set(col.lower() for col in data.columns)

        # Common EHR field indicators
        ehr_indicators = [
            ["patient", "mrn", "patient_id"],
            ["encounter", "visit", "admission"],
            ["diagnosis", "condition", "icd"],
            ["date", "timestamp", "datetime"],
            ["provider", "physician", "clinician"],
        ]

        matches = 0
        for group in ehr_indicators:
            if any(indicator in col for col in columns for indicator in group):
                matches += 1

        result["confidence"] = matches / len(ehr_indicators)

        if result["confidence"] > 0.3:
            result["recommendations"] = [
                "Generic EHR data detected",
                "Consider mapping to standard model (FHIR or OMOP)",
                "Implement data quality checks for clinical accuracy",
                "Validate patient identifiers and temporal consistency",
            ]

        return result

    def _detect_coding_systems(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect medical coding systems used in the data."""
        detected_codes = []

        for col in data.columns:
            if data[col].dtype != 'object':
                continue

            sample = data[col].dropna().head(100).astype(str)

            if sample.empty:
                continue

            for system, pattern in self.coding_patterns.items():
                matches = sample.str.match(pattern).sum()
                match_rate = matches / len(sample)

                if match_rate > 0.5:
                    detected_codes.append({
                        "system": system,
                        "column": col,
                        "match_rate": round(float(match_rate), 2),
                        "sample_codes": sample.head(5).tolist(),
                    })

        return detected_codes

    def _detect_healthcare_fields(self, data: pd.DataFrame) -> List[str]:
        """Detect healthcare-specific fields."""
        standards = set()

        columns = [col.lower() for col in data.columns]

        # Check for common healthcare standards
        if any("fhir" in col or "resource" in col for col in columns):
            standards.add("FHIR")

        if any("concept_id" in col for col in columns):
            standards.add("OMOP")

        if any("icd" in col or "diagnosis" in col for col in columns):
            standards.add("ICD")

        if any("loinc" in col or "lab" in col for col in columns):
            standards.add("LOINC")

        if any("snomed" in col for col in columns):
            standards.add("SNOMED")

        if any("cpt" in col or "procedure" in col for col in columns):
            standards.add("CPT")

        return sorted(list(standards))

    def validate_fhir_resource(
        self,
        data: Dict[str, Any],
        resource_type: str,
    ) -> Dict[str, Any]:
        """
        Validate FHIR resource using fhir.resources library (if available).

        Args:
            data: FHIR resource as dictionary
            resource_type: Expected resource type (Patient, Observation, etc.)

        Returns:
            Validation results
        """
        results = {
            "valid": False,
            "errors": [],
            "warnings": [],
        }

        try:
            # Try to import fhir.resources
            import importlib

            module_name = f"fhir.resources.{resource_type.lower()}"
            module = importlib.import_module(module_name)
            resource_class = getattr(module, resource_type)

            # Validate resource
            try:
                resource = resource_class.parse_obj(data)
                resource.dict()  # Trigger validation
                results["valid"] = True
            except Exception as e:
                results["errors"].append(str(e))

        except ImportError:
            results["warnings"].append("fhir.resources library not available")
        except Exception as e:
            results["errors"].append(f"Validation error: {str(e)}")

        return results

    def suggest_mappings(
        self,
        data: pd.DataFrame,
        target_model: str = "fhir",
    ) -> Dict[str, str]:
        """
        Suggest column mappings to a target healthcare data model.

        Args:
            data: Source DataFrame
            target_model: Target model (fhir, omop, etc.)

        Returns:
            Dictionary mapping source columns to target fields
        """
        mappings = {}

        if target_model.lower() == "fhir":
            # Common mappings to FHIR Patient resource
            mapping_rules = {
                ("patient_id", "mrn", "patient", "id"): "identifier",
                ("name", "patient_name", "full_name"): "name",
                ("gender", "sex"): "gender",
                ("dob", "birth_date", "birthdate"): "birthDate",
                ("address", "street", "city"): "address",
            }

            for col in data.columns:
                col_lower = col.lower()
                for patterns, fhir_field in mapping_rules.items():
                    if any(pattern in col_lower for pattern in patterns):
                        mappings[col] = fhir_field
                        break

        elif target_model.lower() == "omop":
            # Common mappings to OMOP person table
            mapping_rules = {
                ("patient_id", "person_id", "id"): "person_id",
                ("gender", "sex"): "gender_concept_id",
                ("birth_year", "year_of_birth", "yob"): "year_of_birth",
                ("race", "ethnicity"): "race_concept_id",
            }

            for col in data.columns:
                col_lower = col.lower()
                for patterns, omop_field in mapping_rules.items():
                    if any(pattern in col_lower for pattern in patterns):
                        mappings[col] = omop_field
                        break

        return mappings


__all__ = ["HealthcareDataDetector"]
