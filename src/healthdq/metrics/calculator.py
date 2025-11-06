"""
Data quality metrics calculator
Author: Agate Jarmakoviča
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from healthdq.utils.logger import get_logger
from healthdq.utils.helpers import safe_divide

logger = get_logger(__name__)


class MetricsCalculator:
    """
    Calculate comprehensive data quality metrics.

    Metrics organized by FAIR principles:
    - Findability: metadata completeness, documentation
    - Accessibility: data completeness, availability
    - Interoperability: format standardization, schema compliance
    - Reusability: consistency, accuracy, timeliness
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize metrics calculator."""
        from healthdq.config import get_config

        self.config = config or get_config()

    def calculate_all(
        self,
        data: pd.DataFrame,
        original_data: Optional[pd.DataFrame] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate all quality metrics.

        Args:
            data: DataFrame to analyze
            original_data: Original DataFrame for comparison
            schema: Expected schema for validation

        Returns:
            Dictionary with all metrics
        """
        logger.info("Calculating comprehensive quality metrics")

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "data_shape": {"rows": data.shape[0], "columns": data.shape[1]},
            "completeness": self.calculate_completeness(data),
            "accuracy": self.calculate_accuracy(data, original_data),
            "consistency": self.calculate_consistency(data),
            "timeliness": self.calculate_timeliness(data),
            "uniqueness": self.calculate_uniqueness(data),
            "validity": self.calculate_validity(data, schema),
            "fair_metrics": self.calculate_fair_metrics(data),
        }

        # Calculate overall score
        metrics["overall_score"] = self._calculate_overall_score(metrics)

        logger.info(f"Metrics calculated. Overall score: {metrics['overall_score']:.2f}")
        return metrics

    def calculate_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate completeness metrics.

        Measures: missing values, empty strings, null records
        """
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isna().sum().sum()

        # Per-column completeness
        column_completeness = {}
        for col in data.columns:
            missing = data[col].isna().sum()
            completeness = 1.0 - (missing / len(data))
            column_completeness[col] = {
                "completeness": round(completeness, 4),
                "missing_count": int(missing),
                "missing_percentage": round(missing / len(data) * 100, 2),
            }

        # Overall completeness score
        overall_completeness = 1.0 - (missing_cells / total_cells)

        return {
            "overall_score": round(overall_completeness, 4),
            "total_missing": int(missing_cells),
            "missing_percentage": round(missing_cells / total_cells * 100, 2),
            "column_completeness": column_completeness,
            "columns_with_issues": [
                col for col, stats in column_completeness.items() if stats["completeness"] < 0.95
            ],
        }

    def calculate_accuracy(
        self, data: pd.DataFrame, original_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Calculate accuracy metrics.

        Measures: data type accuracy, format compliance, outliers
        """
        accuracy_metrics = {
            "type_accuracy": {},
            "format_compliance": {},
            "outlier_ratio": {},
        }

        # Type accuracy (check if values match expected types)
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Check for non-numeric values that were coerced
                non_numeric = pd.to_numeric(data[col].astype(str), errors="coerce").isna().sum()
                type_accuracy = 1.0 - safe_divide(non_numeric, len(data))
                accuracy_metrics["type_accuracy"][col] = round(type_accuracy, 4)

            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                # Check for invalid dates
                invalid_dates = data[col].isna().sum()
                type_accuracy = 1.0 - safe_divide(invalid_dates, len(data))
                accuracy_metrics["type_accuracy"][col] = round(type_accuracy, 4)

        # Outlier detection for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((data[col] < (q1 - 1.5 * iqr)) | (data[col] > (q3 + 1.5 * iqr))).sum()
            outlier_ratio = safe_divide(outliers, len(data))
            accuracy_metrics["outlier_ratio"][col] = round(outlier_ratio, 4)

        # Calculate overall accuracy score
        type_scores = list(accuracy_metrics["type_accuracy"].values())
        outlier_scores = [1.0 - r for r in accuracy_metrics["outlier_ratio"].values()]
        all_scores = type_scores + outlier_scores

        overall_accuracy = np.mean(all_scores) if all_scores else 1.0

        accuracy_metrics["overall_score"] = round(overall_accuracy, 4)

        # If original data provided, calculate improvement
        if original_data is not None:
            original_completeness = self.calculate_completeness(original_data)["overall_score"]
            current_completeness = self.calculate_completeness(data)["overall_score"]
            accuracy_metrics["improvement"] = round(current_completeness - original_completeness, 4)

        return accuracy_metrics

    def calculate_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate consistency metrics.

        Measures: format consistency, referential integrity, business rule compliance
        """
        consistency_metrics = {
            "format_consistency": {},
            "value_consistency": {},
        }

        # Format consistency for text columns
        text_cols = data.select_dtypes(include=["object"]).columns
        for col in text_cols:
            if data[col].notna().sum() == 0:
                continue

            # Check for consistent formatting (e.g., all uppercase, all lowercase)
            non_null_values = data[col].dropna()
            if len(non_null_values) > 0:
                # Check case consistency
                all_upper = non_null_values.str.isupper().sum()
                all_lower = non_null_values.str.islower().sum()
                mixed_case = len(non_null_values) - all_upper - all_lower

                case_consistency = max(all_upper, all_lower) / len(non_null_values)
                consistency_metrics["format_consistency"][col] = {
                    "case_consistency": round(case_consistency, 4),
                    "all_upper": int(all_upper),
                    "all_lower": int(all_lower),
                    "mixed_case": int(mixed_case),
                }

        # Value consistency (check for standardized values)
        for col in data.columns:
            unique_ratio = data[col].nunique() / len(data)
            consistency_metrics["value_consistency"][col] = {
                "unique_ratio": round(unique_ratio, 4),
                "unique_count": int(data[col].nunique()),
                "is_consistent": unique_ratio < 0.1,  # Low cardinality suggests consistency
            }

        # Overall consistency score
        format_scores = [
            v["case_consistency"]
            for v in consistency_metrics["format_consistency"].values()
            if "case_consistency" in v
        ]
        overall_consistency = np.mean(format_scores) if format_scores else 1.0

        consistency_metrics["overall_score"] = round(overall_consistency, 4)

        return consistency_metrics

    def calculate_timeliness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate timeliness metrics.

        Measures: data freshness, update frequency
        """
        timeliness_metrics = {
            "has_timestamp": False,
            "timestamp_columns": [],
        }

        # Detect timestamp columns
        datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()

        # Also check for potential timestamp columns in object type
        for col in data.select_dtypes(include=["object"]).columns:
            if any(
                keyword in col.lower() for keyword in ["date", "time", "timestamp", "created", "updated"]
            ):
                try:
                    pd.to_datetime(data[col], errors="coerce")
                    datetime_cols.append(col)
                except Exception:
                    pass

        if datetime_cols:
            timeliness_metrics["has_timestamp"] = True
            timeliness_metrics["timestamp_columns"] = datetime_cols

            # Calculate data freshness (days since most recent timestamp)
            for col in datetime_cols:
                timestamps = pd.to_datetime(data[col], errors="coerce").dropna()
                if len(timestamps) > 0:
                    most_recent = timestamps.max()
                    days_old = (pd.Timestamp.now() - most_recent).days
                    timeliness_metrics[f"{col}_freshness_days"] = int(days_old)

        # Calculate score (1.0 if has timestamps, 0.5 if not)
        timeliness_metrics["overall_score"] = 1.0 if timeliness_metrics["has_timestamp"] else 0.5

        return timeliness_metrics

    def calculate_uniqueness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate uniqueness metrics.

        Measures: duplicate records, duplicate values in key columns
        """
        # Check for duplicate rows
        duplicate_rows = data.duplicated().sum()
        duplicate_ratio = safe_divide(duplicate_rows, len(data))

        # Check for potential ID columns
        id_columns = [col for col in data.columns if "id" in col.lower()]

        column_uniqueness = {}
        for col in data.columns:
            unique_count = data[col].nunique()
            uniqueness_ratio = safe_divide(unique_count, len(data))
            column_uniqueness[col] = {
                "unique_count": int(unique_count),
                "uniqueness_ratio": round(uniqueness_ratio, 4),
                "is_unique": uniqueness_ratio == 1.0,
            }

        # Overall uniqueness score
        overall_uniqueness = 1.0 - duplicate_ratio

        return {
            "overall_score": round(overall_uniqueness, 4),
            "duplicate_rows": int(duplicate_rows),
            "duplicate_ratio": round(duplicate_ratio, 4),
            "column_uniqueness": column_uniqueness,
            "potential_id_columns": id_columns,
        }

    def calculate_validity(
        self, data: pd.DataFrame, schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate validity metrics.

        Measures: schema compliance, constraint violations, data type validity
        """
        validity_metrics = {
            "schema_compliant": schema is not None,
            "constraint_violations": {},
        }

        # If schema provided, validate against it
        if schema:
            from healthdq.utils.validators import SchemaValidator

            validator = SchemaValidator()

            # Check required columns
            if "required_columns" in schema:
                required_check = validator.validate_required_columns(data, schema["required_columns"])
                validity_metrics["missing_required_columns"] = required_check.get("missing_columns", [])

            # Check column constraints
            if "constraints" in schema:
                constraint_results = validator.validate_column_constraints(data, schema["constraints"])
                validity_metrics["constraint_violations"] = {
                    col: result["violations"]
                    for col, result in constraint_results.items()
                    if not result["valid"]
                }

        # Calculate overall validity score
        if schema:
            violations = len(validity_metrics.get("constraint_violations", {}))
            missing_cols = len(validity_metrics.get("missing_required_columns", []))
            total_issues = violations + missing_cols
            validity_metrics["overall_score"] = max(0.0, 1.0 - (total_issues * 0.1))
        else:
            validity_metrics["overall_score"] = 1.0  # No schema to validate against

        return validity_metrics

    def calculate_fair_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate FAIR principles metrics.

        F - Findable: Has metadata, documentation
        A - Accessible: Data completeness
        I - Interoperable: Standard formats
        R - Reusable: Clear licensing, good quality
        """
        fair_metrics = {}

        # Findability
        has_metadata = bool(data.attrs)
        fair_metrics["findable"] = {
            "has_metadata": has_metadata,
            "has_column_names": all(bool(str(col).strip()) for col in data.columns),
            "score": 1.0 if has_metadata else 0.5,
        }

        # Accessibility
        completeness = self.calculate_completeness(data)
        fair_metrics["accessible"] = {"completeness": completeness["overall_score"], "score": completeness["overall_score"]}

        # Interoperability
        has_standard_types = all(
            pd.api.types.is_numeric_dtype(data[col])
            or pd.api.types.is_datetime64_any_dtype(data[col])
            or pd.api.types.is_string_dtype(data[col])
            for col in data.columns
        )
        fair_metrics["interoperable"] = {"standard_types": has_standard_types, "score": 1.0 if has_standard_types else 0.7}

        # Reusability
        consistency = self.calculate_consistency(data)
        fair_metrics["reusable"] = {"consistency": consistency["overall_score"], "score": consistency["overall_score"]}

        # Overall FAIR score
        fair_scores = [v["score"] for v in fair_metrics.values()]
        fair_metrics["overall_fair_score"] = round(np.mean(fair_scores), 4)

        return fair_metrics

    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            "completeness": 0.25,
            "accuracy": 0.25,
            "consistency": 0.20,
            "uniqueness": 0.15,
            "validity": 0.15,
        }

        score = 0.0
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                metric_score = metrics[metric_name].get("overall_score", 1.0)
                score += metric_score * weight

        return round(score, 4)


__all__ = ["MetricsCalculator"]
