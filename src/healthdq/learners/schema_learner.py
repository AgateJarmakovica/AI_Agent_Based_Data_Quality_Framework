"""
Adaptive Schema Learning for Healthcare Data
Author: Agate Jarmakoviča

Automatically learns data schemas, patterns, and constraints from healthcare datasets.
Uses machine learning to cluster similar columns and infer implicit schemas.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json

from healthdq.utils.logger import get_logger

logger = get_logger(__name__)


class SchemaLearner:
    """
    Learns data schemas and patterns using unsupervised ML techniques.

    Features:
    - Automatic column type inference
    - Column clustering by statistical similarity
    - Pattern recognition in data values
    - Constraint discovery (ranges, enums, formats)
    - Healthcare-specific field detection
    - Schema evolution tracking

    Example:
        learner = SchemaLearner()
        learned_schema = learner.learn(dataframe)
        validation = learner.validate_against_schema(new_data, learned_schema)
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the schema learner."""
        from healthdq.config import get_config

        self.config = config or get_config()
        self.learned_schemas: Dict[str, Dict[str, Any]] = {}
        self.learning_history: List[Dict[str, Any]] = []

        # Try to import optional ML libraries
        self.ml_available = self._check_ml_dependencies()

    def _check_ml_dependencies(self) -> bool:
        """Check if ML libraries are available."""
        try:
            import sklearn
            return True
        except ImportError:
            logger.warning("scikit-learn not available - using basic schema learning")
            return False

    def learn(
        self,
        data: pd.DataFrame,
        schema_name: Optional[str] = None,
        include_statistics: bool = True,
    ) -> Dict[str, Any]:
        """
        Learn schema from data using statistical analysis and ML clustering.

        Args:
            data: DataFrame to learn from
            schema_name: Name for this schema (auto-generated if None)
            include_statistics: Whether to include detailed statistics

        Returns:
            Learned schema dictionary with structure, constraints, and patterns
        """
        if schema_name is None:
            schema_name = f"schema_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Learning schema '{schema_name}' from data with shape {data.shape}")

        schema = {
            "schema_name": schema_name,
            "learned_at": datetime.now().isoformat(),
            "num_rows": len(data),
            "num_columns": len(data.columns),
            "columns": {},
            "column_groups": {},
            "constraints": {},
            "patterns": {},
            "confidence": 0.0,
        }

        # Learn column-level information
        for col in data.columns:
            schema["columns"][col] = self._learn_column_schema(data[col], include_statistics)

        # Learn column relationships and groupings
        if self.ml_available and len(data.columns) > 1:
            schema["column_groups"] = self._learn_column_groups(data)

        # Discover global constraints
        schema["constraints"] = self._discover_constraints(data)

        # Detect patterns
        schema["patterns"] = self._detect_patterns(data)

        # Calculate confidence score
        schema["confidence"] = self._calculate_schema_confidence(data, schema)

        # Store learned schema
        self.learned_schemas[schema_name] = schema

        # Record learning history
        self.learning_history.append({
            "schema_name": schema_name,
            "timestamp": datetime.now().isoformat(),
            "num_rows": len(data),
            "num_columns": len(data.columns),
            "confidence": schema["confidence"],
        })

        logger.info(f"Schema learned with confidence {schema['confidence']:.2f}")

        return schema

    def _learn_column_schema(
        self,
        series: pd.Series,
        include_statistics: bool = True,
    ) -> Dict[str, Any]:
        """Learn schema for a single column."""
        col_schema = {
            "name": series.name,
            "dtype": str(series.dtype),
            "nullable": bool(series.isna().any()),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": float(series.isna().sum() / len(series) * 100),
            "unique_count": int(series.nunique()),
            "inferred_type": self._infer_semantic_type(series),
        }

        # Value constraints
        if series.dtype in ['int64', 'float64']:
            col_schema["constraints"] = {
                "min": float(series.min()) if not series.empty else None,
                "max": float(series.max()) if not series.empty else None,
                "mean": float(series.mean()) if not series.empty else None,
                "median": float(series.median()) if not series.empty else None,
                "std": float(series.std()) if not series.empty else None,
            }
        elif series.dtype == 'object':
            # String/categorical constraints
            value_counts = series.value_counts()
            if len(value_counts) <= 50:  # Likely categorical
                col_schema["constraints"] = {
                    "enum": value_counts.index.tolist(),
                    "frequencies": value_counts.to_dict(),
                }
            else:
                # String patterns
                col_schema["constraints"] = {
                    "min_length": int(series.astype(str).str.len().min()) if not series.empty else 0,
                    "max_length": int(series.astype(str).str.len().max()) if not series.empty else 0,
                    "pattern": self._detect_string_pattern(series),
                }

        # Healthcare-specific detection
        col_schema["healthcare_field"] = self._detect_healthcare_field(series)

        return col_schema

    def _infer_semantic_type(self, series: pd.Series) -> str:
        """Infer semantic type of a column."""
        col_name = str(series.name).lower()

        # Check for common healthcare fields
        if any(term in col_name for term in ['patient', 'subject', 'person']):
            return "patient_identifier"
        elif any(term in col_name for term in ['date', 'time', 'timestamp']):
            return "temporal"
        elif any(term in col_name for term in ['age', 'year']):
            return "age"
        elif any(term in col_name for term in ['gender', 'sex']):
            return "gender"
        elif any(term in col_name for term in ['diagnosis', 'condition', 'disease']):
            return "diagnosis"
        elif any(term in col_name for term in ['medication', 'drug', 'prescription']):
            return "medication"
        elif any(term in col_name for term in ['lab', 'test', 'result', 'value']):
            return "laboratory"
        elif any(term in col_name for term in ['code', 'icd', 'snomed', 'loinc']):
            return "medical_code"

        # Infer from data type
        if series.dtype in ['int64', 'float64']:
            return "numeric"
        elif series.dtype == 'object':
            if series.nunique() < 50:
                return "categorical"
            return "text"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "temporal"

        return "unknown"

    def _detect_healthcare_field(self, series: pd.Series) -> Optional[Dict[str, Any]]:
        """Detect if column represents a healthcare-specific field."""
        col_name = str(series.name).lower()

        healthcare_patterns = {
            "patient_id": ["patient", "subject", "mrn", "patient_id"],
            "encounter_id": ["encounter", "visit", "admission"],
            "diagnosis": ["diagnosis", "condition", "icd", "disease"],
            "procedure": ["procedure", "operation", "surgery", "cpt"],
            "medication": ["medication", "drug", "prescription", "rx"],
            "laboratory": ["lab", "test", "result", "loinc"],
            "vital_signs": ["blood_pressure", "heart_rate", "temperature", "weight", "height", "bmi"],
            "demographics": ["age", "gender", "sex", "race", "ethnicity", "dob"],
        }

        for field_type, patterns in healthcare_patterns.items():
            if any(pattern in col_name for pattern in patterns):
                return {
                    "field_type": field_type,
                    "confidence": 0.8,
                    "patterns_matched": [p for p in patterns if p in col_name]
                }

        return None

    def _detect_string_pattern(self, series: pd.Series) -> Optional[str]:
        """Detect common string patterns (e.g., dates, codes, IDs)."""
        sample = series.dropna().head(100).astype(str)

        if sample.empty:
            return None

        # Check for common patterns
        if sample.str.match(r'^\d{4}-\d{2}-\d{2}').sum() > len(sample) * 0.8:
            return "date_iso"
        elif sample.str.match(r'^\d{2}/\d{2}/\d{4}').sum() > len(sample) * 0.8:
            return "date_us"
        elif sample.str.match(r'^[A-Z]\d{2,}').sum() > len(sample) * 0.8:
            return "medical_code"
        elif sample.str.match(r'^\d{5,}$').sum() > len(sample) * 0.8:
            return "numeric_id"
        elif sample.str.match(r'^[A-Z0-9-]{8,}$').sum() > len(sample) * 0.8:
            return "identifier"

        return "text"

    def _learn_column_groups(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Use ML clustering to group similar columns.
        Columns that behave similarly statistically likely represent related concepts.
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            # Select numeric columns for clustering
            numeric_data = data.select_dtypes(include=[np.number])

            if len(numeric_data.columns) < 2:
                return {}

            # Compute column statistics for clustering
            stats = []
            for col in numeric_data.columns:
                col_stats = [
                    numeric_data[col].mean(),
                    numeric_data[col].std(),
                    numeric_data[col].min(),
                    numeric_data[col].max(),
                    numeric_data[col].median(),
                ]
                stats.append(col_stats)

            # Cluster columns
            scaler = StandardScaler()
            X = scaler.fit_transform(np.array(stats))

            n_clusters = min(5, len(numeric_data.columns))
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X)

            # Group columns by cluster
            groups = {}
            for i, col in enumerate(numeric_data.columns):
                cluster_id = f"group_{labels[i]}"
                if cluster_id not in groups:
                    groups[cluster_id] = []
                groups[cluster_id].append(col)

            logger.debug(f"Discovered {len(groups)} column groups")
            return groups

        except ImportError:
            logger.warning("scikit-learn not available for column grouping")
            return {}
        except Exception as e:
            logger.error(f"Error in column grouping: {e}")
            return {}

    def _discover_constraints(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Discover global constraints across the dataset."""
        constraints = {
            "row_count_range": (len(data), len(data)),  # For schema validation
            "required_columns": list(data.columns),
            "unique_together": [],  # Composite keys
            "correlations": {},
        }

        # Find potential composite keys (columns that together form unique identifiers)
        for col in data.columns:
            if data[col].nunique() == len(data):
                constraints["unique_together"].append([col])

        # Compute correlations for numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            high_corr = []

            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": float(corr_matrix.iloc[i, j])
                        })

            constraints["correlations"] = high_corr

        return constraints

    def _detect_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect common patterns in the dataset."""
        patterns = {
            "temporal_columns": [],
            "identifier_columns": [],
            "categorical_columns": [],
            "numeric_ranges": {},
        }

        for col in data.columns:
            # Temporal patterns
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                patterns["temporal_columns"].append(col)

            # Identifier patterns (high cardinality, unique)
            elif data[col].nunique() > len(data) * 0.9:
                patterns["identifier_columns"].append(col)

            # Categorical patterns (low cardinality)
            elif data[col].nunique() < 50:
                patterns["categorical_columns"].append(col)

            # Numeric ranges
            elif data[col].dtype in ['int64', 'float64']:
                patterns["numeric_ranges"][col] = {
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "range": float(data[col].max() - data[col].min()),
                }

        return patterns

    def _calculate_schema_confidence(
        self,
        data: pd.DataFrame,
        schema: Dict[str, Any],
    ) -> float:
        """Calculate confidence score for learned schema."""
        factors = []

        # Factor 1: Data completeness
        completeness = 1.0 - (data.isna().sum().sum() / (len(data) * len(data.columns)))
        factors.append(completeness)

        # Factor 2: Column type consistency
        type_consistency = sum(1 for col_schema in schema["columns"].values()
                              if col_schema["inferred_type"] != "unknown") / len(schema["columns"])
        factors.append(type_consistency)

        # Factor 3: Pattern detection success
        pattern_score = min(1.0, len(schema["patterns"]["temporal_columns"]) +
                           len(schema["patterns"]["identifier_columns"]) +
                           len(schema["patterns"]["categorical_columns"])) / len(data.columns)
        factors.append(pattern_score)

        # Factor 4: Sample size adequacy
        sample_score = min(1.0, len(data) / 1000)  # Optimal at 1000+ rows
        factors.append(sample_score)

        return round(float(np.mean(factors)), 2)

    def validate_against_schema(
        self,
        data: pd.DataFrame,
        schema: Dict[str, Any],
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate data against a learned schema.

        Args:
            data: Data to validate
            schema: Learned schema to validate against
            strict: If True, enforces all constraints strictly

        Returns:
            Validation results with issues and compliance score
        """
        results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "compliance_score": 0.0,
        }

        # Check column presence
        missing_cols = set(schema["columns"].keys()) - set(data.columns)
        extra_cols = set(data.columns) - set(schema["columns"].keys())

        if missing_cols:
            results["issues"].append({
                "type": "missing_columns",
                "severity": "high",
                "columns": list(missing_cols),
            })
            results["valid"] = False

        if extra_cols and strict:
            results["warnings"].append({
                "type": "extra_columns",
                "severity": "low",
                "columns": list(extra_cols),
            })

        # Validate each column
        for col in data.columns:
            if col not in schema["columns"]:
                continue

            col_schema = schema["columns"][col]
            col_issues = self._validate_column(data[col], col_schema, strict)

            if col_issues:
                results["issues"].extend(col_issues)
                if any(issue["severity"] == "high" for issue in col_issues):
                    results["valid"] = False

        # Calculate compliance score
        total_checks = len(schema["columns"]) + 1  # +1 for column presence check
        failed_checks = len([i for i in results["issues"] if i["severity"] == "high"])
        results["compliance_score"] = round((total_checks - failed_checks) / total_checks, 2)

        return results

    def _validate_column(
        self,
        series: pd.Series,
        col_schema: Dict[str, Any],
        strict: bool,
    ) -> List[Dict[str, Any]]:
        """Validate a single column against its schema."""
        issues = []

        # Check nullability
        if not col_schema["nullable"] and series.isna().any():
            issues.append({
                "type": "null_constraint_violation",
                "column": series.name,
                "severity": "high",
                "description": f"Column '{series.name}' contains null values but schema requires non-null",
            })

        # Check numeric constraints
        if "constraints" in col_schema and series.dtype in ['int64', 'float64']:
            constraints = col_schema["constraints"]

            if "min" in constraints and series.min() < constraints["min"]:
                issues.append({
                    "type": "range_violation",
                    "column": series.name,
                    "severity": "medium",
                    "description": f"Value below expected minimum: {series.min()} < {constraints['min']}",
                })

            if "max" in constraints and series.max() > constraints["max"]:
                issues.append({
                    "type": "range_violation",
                    "column": series.name,
                    "severity": "medium",
                    "description": f"Value above expected maximum: {series.max()} > {constraints['max']}",
                })

        # Check enum constraints
        if "constraints" in col_schema and "enum" in col_schema["constraints"]:
            valid_values = set(col_schema["constraints"]["enum"])
            invalid = set(series.dropna().unique()) - valid_values

            if invalid:
                issues.append({
                    "type": "enum_violation",
                    "column": series.name,
                    "severity": "medium",
                    "description": f"Invalid values found: {list(invalid)[:5]}",
                })

        return issues

    def export_schema(
        self,
        schema_name: str,
        output_path: str,
        format: str = "json",
    ) -> None:
        """Export learned schema to file."""
        if schema_name not in self.learned_schemas:
            raise ValueError(f"Schema '{schema_name}' not found")

        schema = self.learned_schemas[schema_name]

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(schema, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Schema exported to {output_path}")


__all__ = ["SchemaLearner"]
