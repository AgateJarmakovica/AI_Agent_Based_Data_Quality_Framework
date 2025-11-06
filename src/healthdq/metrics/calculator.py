"""
Data quality metrics calculator based on research publication formulas
Author: Agate Jarmakoviča

Implements three primary dimensions from publication:
1. Accuracy - Anomaly detection with Isolation Forest & LOF
2. Completeness - Missing value analysis with KNN imputation reference
3. Reusability - FAIR principles: documentation, metadata, version control

Weighted DQ Score: w1×Accuracy + w2×Completeness + w3×Reusability
Where: w1=0.4, w2=0.4, w3=0.2 (from ISO 25024 guidelines)
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
        Calculate accuracy metrics using anomaly detection.

        Based on publication formula:
        Accuracy = 1 - (Number of detected anomalies / Total records)

        Uses Isolation Forest and Local Outlier Factor (LOF) for anomaly detection.

        Args:
            data: DataFrame to analyze
            original_data: Optional original data for comparison

        Returns:
            Dictionary with accuracy metrics including:
            - anomaly_ratio: Ratio of anomalies detected
            - overall_score: Publication formula accuracy score
            - method: Detection method used (IsolationForest or LOF)
        """
        accuracy_metrics = {
            "type_accuracy": {},
            "anomaly_detection": {},
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

        # Anomaly detection using Isolation Forest and LOF
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Prepare data for anomaly detection
            numeric_data = data[numeric_cols].dropna()

            if len(numeric_data) >= 10:  # Need minimum data for anomaly detection
                try:
                    from sklearn.ensemble import IsolationForest
                    from sklearn.neighbors import LocalOutlierFactor

                    # Isolation Forest (as per publication)
                    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
                    iso_predictions = iso_forest.fit_predict(numeric_data)
                    iso_anomalies = (iso_predictions == -1).sum()

                    # Local Outlier Factor (as per publication)
                    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
                    lof_predictions = lof.fit_predict(numeric_data)
                    lof_anomalies = (lof_predictions == -1).sum()

                    # Use average of both methods
                    total_anomalies = (iso_anomalies + lof_anomalies) / 2

                    accuracy_metrics["anomaly_detection"] = {
                        "isolation_forest_anomalies": int(iso_anomalies),
                        "lof_anomalies": int(lof_anomalies),
                        "average_anomalies": round(float(total_anomalies), 2),
                        "anomaly_ratio": round(total_anomalies / len(numeric_data), 4),
                        "method": "Isolation Forest + LOF"
                    }

                    # Publication formula: Accuracy = 1 - (Number of anomalies / Total records)
                    accuracy_from_anomalies = 1.0 - (total_anomalies / len(numeric_data))

                except ImportError:
                    logger.warning("sklearn not available, using IQR method for anomaly detection")
                    accuracy_from_anomalies = self._calculate_accuracy_iqr(data, numeric_cols, accuracy_metrics)
            else:
                # Too few rows for ML-based anomaly detection
                accuracy_from_anomalies = self._calculate_accuracy_iqr(data, numeric_cols, accuracy_metrics)
        else:
            # No numeric columns
            accuracy_from_anomalies = 1.0

        # Calculate overall accuracy score
        type_scores = list(accuracy_metrics["type_accuracy"].values())

        # Combine type accuracy and anomaly-based accuracy
        all_scores = type_scores + [accuracy_from_anomalies]
        overall_accuracy = np.mean(all_scores) if all_scores else 1.0

        accuracy_metrics["overall_score"] = round(overall_accuracy, 4)
        accuracy_metrics["anomaly_based_accuracy"] = round(accuracy_from_anomalies, 4)

        # If original data provided, calculate improvement
        if original_data is not None:
            original_accuracy = self.calculate_accuracy(original_data)["overall_score"]
            accuracy_metrics["improvement"] = round(overall_accuracy - original_accuracy, 4)

        return accuracy_metrics

    def _calculate_accuracy_iqr(
        self, data: pd.DataFrame, numeric_cols: pd.Index, accuracy_metrics: Dict[str, Any]
    ) -> float:
        """
        Fallback accuracy calculation using IQR method.

        Returns:
            Accuracy score based on IQR outlier detection
        """
        total_outliers = 0
        total_values = 0

        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((data[col] < (q1 - 1.5 * iqr)) | (data[col] > (q3 + 1.5 * iqr))).sum()
            outlier_ratio = safe_divide(outliers, len(data))
            accuracy_metrics["outlier_ratio"][col] = round(outlier_ratio, 4)

            total_outliers += outliers
            total_values += len(data[col].dropna())

        # IQR-based accuracy
        return 1.0 - safe_divide(total_outliers, total_values) if total_values > 0 else 1.0

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
        Calculate FAIR principles metrics with emphasis on Reusability.

        Based on publication formula for Reusability:
        Reusability = (Documented processes + Metadata + Version control) / 3

        F - Findable: Has metadata, documentation
        A - Accessible: Data completeness
        I - Interoperable: Standard formats
        R - Reusable: Documentation + Metadata + Reproducibility
        """
        fair_metrics = {}

        # Findability
        has_metadata = bool(data.attrs)
        has_description = "description" in data.attrs if data.attrs else False
        fair_metrics["findable"] = {
            "has_metadata": has_metadata,
            "has_description": has_description,
            "has_column_names": all(bool(str(col).strip()) for col in data.columns),
            "score": 1.0 if (has_metadata and has_description) else 0.5 if has_metadata else 0.0,
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

        # Reusability (Publication formula)
        # Check three components: Documentation, Metadata, Version Control/Reproducibility
        doc_exists = has_description or bool(data.attrs)  # Has any documentation
        metadata_exists = has_metadata and len(data.attrs) > 0  # Has meaningful metadata
        version_control = "version" in data.attrs or "created_at" in data.attrs or "source" in data.attrs

        reusability_components = {
            "documentation": 1.0 if doc_exists else 0.0,
            "metadata": 1.0 if metadata_exists else 0.0,
            "version_control": 1.0 if version_control else 0.0,
        }

        # Publication formula: Reusability = (sum of components) / 3
        reusability_score = sum(reusability_components.values()) / 3

        fair_metrics["reusable"] = {
            "components": reusability_components,
            "score": round(reusability_score, 4),
            "formula": "Reusability = (documentation + metadata + version_control) / 3"
        }

        # Overall FAIR score
        fair_scores = [v["score"] for v in fair_metrics.values()]
        fair_metrics["overall_fair_score"] = round(np.mean(fair_scores), 4)

        return fair_metrics

    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate weighted overall quality score.

        Based on publication formula and ISO 25024 guidelines:
        DQ_total = w1 × Accuracy + w2 × Completeness + w3 × Reusability

        Where:
        w1 = 0.4 (Accuracy - critical for clinical precision)
        w2 = 0.4 (Completeness - essential for model integrity)
        w3 = 0.2 (Reusability - ensures reproducibility)
        """
        # Primary dimensions from publication
        weights = {
            "accuracy": 0.4,      # w1 - Accuracy (anomaly detection)
            "completeness": 0.4,  # w2 - Completeness (missing values)
            "reusability": 0.2,   # w3 - Reusability (FAIR principles)
        }

        score = 0.0
        available_weight = 0.0

        # Calculate primary score from three main dimensions
        for metric_name, weight in weights.items():
            if metric_name == "reusability":
                # Use FAIR reusability score
                if "fair_metrics" in metrics:
                    metric_score = metrics["fair_metrics"]["reusable"]["score"]
                    score += metric_score * weight
                    available_weight += weight
            elif metric_name in metrics:
                metric_score = metrics[metric_name].get("overall_score", 1.0)
                score += metric_score * weight
                available_weight += weight

        # Normalize by available weights
        if available_weight > 0:
            score = score / available_weight
        else:
            score = 1.0

        return round(score, 4)

    def calculate_publication_dq_score(
        self,
        data: pd.DataFrame,
        original_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Calculate DQ score exactly as specified in publication.

        Returns three-dimensional analysis:
        1. Accuracy (Isolation Forest + LOF anomaly detection)
        2. Completeness (Missing value ratio)
        3. Reusability (Documentation + Metadata + Version Control)

        Final score: DQ_total = 0.4×Accuracy + 0.4×Completeness + 0.2×Reusability
        """
        logger.info("Calculating publication-based DQ score")

        # Calculate three primary dimensions
        accuracy = self.calculate_accuracy(data, original_data)
        completeness = self.calculate_completeness(data)
        fair_metrics = self.calculate_fair_metrics(data)
        reusability = fair_metrics["reusable"]["score"]

        # Extract scores
        accuracy_score = accuracy["overall_score"]
        completeness_score = completeness["overall_score"]
        reusability_score = reusability

        # Publication formula with specified weights
        w1, w2, w3 = 0.4, 0.4, 0.2
        dq_total = (w1 * accuracy_score) + (w2 * completeness_score) + (w3 * reusability_score)

        result = {
            "dq_total": round(dq_total, 4),
            "formula": "DQ = 0.4×Accuracy + 0.4×Completeness + 0.2×Reusability",
            "weights": {"w1_accuracy": w1, "w2_completeness": w2, "w3_reusability": w3},
            "dimensions": {
                "accuracy": {
                    "score": accuracy_score,
                    "weight": w1,
                    "contribution": round(w1 * accuracy_score, 4),
                    "method": accuracy.get("anomaly_detection", {}).get("method", "IQR"),
                },
                "completeness": {
                    "score": completeness_score,
                    "weight": w2,
                    "contribution": round(w2 * completeness_score, 4),
                    "missing_percentage": completeness["missing_percentage"],
                },
                "reusability": {
                    "score": reusability_score,
                    "weight": w3,
                    "contribution": round(w3 * reusability_score, 4),
                    "components": fair_metrics["reusable"]["components"],
                },
            },
            "data_shape": {"rows": data.shape[0], "columns": data.shape[1]},
            "timestamp": datetime.now().isoformat(),
        }

        # Add improvement if original data provided
        if original_data is not None:
            original_result = self.calculate_publication_dq_score(original_data, None)
            result["improvement"] = {
                "dq_total": round(dq_total - original_result["dq_total"], 4),
                "accuracy": round(accuracy_score - original_result["dimensions"]["accuracy"]["score"], 4),
                "completeness": round(completeness_score - original_result["dimensions"]["completeness"]["score"], 4),
                "reusability": round(reusability_score - original_result["dimensions"]["reusability"]["score"], 4),
            }

        logger.info(f"Publication DQ score: {dq_total:.4f}")
        return result


__all__ = ["MetricsCalculator"]
