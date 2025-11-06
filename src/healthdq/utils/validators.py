"""
Data validation utilities for healthdq-ai framework
Author: Agate Jarmakoviča
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path


class DataValidator:
    """Validates data quality and format compliance."""

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_phone(phone: str, country_code: str = "LV") -> bool:
        """Validate phone number format."""
        # Basic validation - can be extended with phonenumbers library
        phone = re.sub(r"[^\d+]", "", phone)
        if country_code == "LV":
            return bool(re.match(r"^\+?371[0-9]{8}$", phone))
        return len(phone) >= 10

    @staticmethod
    def validate_date(date_str: str, format: str = "%Y-%m-%d") -> bool:
        """Validate date format."""
        try:
            datetime.strptime(date_str, format)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_numeric_range(
        value: Union[int, float], min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> bool:
        """Validate if numeric value is within specified range."""
        try:
            value = float(value)
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_missing_percentage(
        data: pd.DataFrame, max_missing_percent: float = 50.0
    ) -> Dict[str, float]:
        """
        Calculate missing value percentage for each column.

        Returns:
            Dict with column names and their missing percentages
        """
        missing_stats = {}
        for col in data.columns:
            missing_count = data[col].isna().sum()
            missing_percent = (missing_count / len(data)) * 100
            missing_stats[col] = missing_percent

        return missing_stats

    @staticmethod
    def validate_duplicates(data: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
        """
        Check for duplicate rows.

        Args:
            data: DataFrame to check
            subset: List of columns to check for duplicates (if None, check all)

        Returns:
            Number of duplicate rows
        """
        return data.duplicated(subset=subset).sum()

    @staticmethod
    def validate_categorical_values(
        series: pd.Series, allowed_values: List[Any]
    ) -> Dict[str, Any]:
        """
        Validate categorical values against allowed set.

        Returns:
            Dict with validation results
        """
        invalid_values = series[~series.isin(allowed_values)].unique()
        return {
            "valid_count": series.isin(allowed_values).sum(),
            "invalid_count": (~series.isin(allowed_values)).sum(),
            "invalid_values": invalid_values.tolist() if len(invalid_values) > 0 else [],
        }

    @staticmethod
    def validate_data_types(data: pd.DataFrame, expected_types: Dict[str, str]) -> Dict[str, bool]:
        """
        Validate that columns have expected data types.

        Args:
            data: DataFrame to validate
            expected_types: Dict mapping column names to expected types

        Returns:
            Dict with validation results per column
        """
        results = {}
        for col, expected_type in expected_types.items():
            if col not in data.columns:
                results[col] = False
                continue

            actual_type = str(data[col].dtype)
            if expected_type == "numeric":
                results[col] = pd.api.types.is_numeric_dtype(data[col])
            elif expected_type == "datetime":
                results[col] = pd.api.types.is_datetime64_any_dtype(data[col])
            elif expected_type == "string":
                results[col] = pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_object_dtype(
                    data[col]
                )
            else:
                results[col] = expected_type in actual_type

        return results

    @staticmethod
    def validate_uniqueness(series: pd.Series, should_be_unique: bool = True) -> bool:
        """Check if series values are unique (e.g., for ID columns)."""
        is_unique = series.nunique() == len(series)
        return is_unique == should_be_unique

    @staticmethod
    def validate_referential_integrity(
        data: pd.DataFrame, foreign_key: str, reference_data: pd.DataFrame, primary_key: str
    ) -> Dict[str, Any]:
        """
        Validate referential integrity between two datasets.

        Returns:
            Dict with validation results
        """
        foreign_values = set(data[foreign_key].dropna().unique())
        primary_values = set(reference_data[primary_key].dropna().unique())

        invalid_references = foreign_values - primary_values

        return {
            "valid": len(invalid_references) == 0,
            "invalid_count": len(invalid_references),
            "invalid_references": list(invalid_references),
        }


class FileValidator:
    """Validates file formats and accessibility."""

    SUPPORTED_FORMATS = {
        "csv": [".csv"],
        "json": [".json"],
        "excel": [".xlsx", ".xls"],
        "parquet": [".parquet"],
        "xml": [".xml"],
        "fhir": [".json"],
    }

    @staticmethod
    def validate_file_exists(file_path: Union[str, Path]) -> bool:
        """Check if file exists."""
        return Path(file_path).exists()

    @staticmethod
    def validate_file_format(file_path: Union[str, Path], expected_format: str) -> bool:
        """Validate file format based on extension."""
        path = Path(file_path)
        allowed_extensions = FileValidator.SUPPORTED_FORMATS.get(expected_format.lower(), [])
        return path.suffix.lower() in allowed_extensions

    @staticmethod
    def validate_file_size(
        file_path: Union[str, Path], max_size_mb: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validate file size.

        Returns:
            Dict with size info and validation result
        """
        path = Path(file_path)
        if not path.exists():
            return {"valid": False, "error": "File does not exist"}

        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        result = {"size_mb": size_mb, "valid": True}

        if max_size_mb is not None:
            result["valid"] = size_mb <= max_size_mb
            result["max_size_mb"] = max_size_mb

        return result

    @staticmethod
    def validate_csv_structure(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate CSV file structure."""
        try:
            # Try reading first few rows
            df = pd.read_csv(file_path, nrows=5)
            return {
                "valid": True,
                "columns": list(df.columns),
                "column_count": len(df.columns),
                "has_header": not all(col.startswith("Unnamed") for col in df.columns),
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}


class SchemaValidator:
    """Validates data against defined schemas."""

    @staticmethod
    def validate_required_columns(data: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
        """Check if all required columns are present."""
        present_columns = set(data.columns)
        required_set = set(required_columns)
        missing_columns = required_set - present_columns

        return {
            "valid": len(missing_columns) == 0,
            "missing_columns": list(missing_columns),
            "present_columns": list(required_set & present_columns),
        }

    @staticmethod
    def validate_column_constraints(
        data: pd.DataFrame, constraints: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate column-level constraints.

        Args:
            constraints: Dict mapping column names to constraint definitions
                Example: {"age": {"type": "numeric", "min": 0, "max": 120}}
        """
        results = {}

        for col, constraint_def in constraints.items():
            if col not in data.columns:
                results[col] = {"valid": False, "error": "Column not found"}
                continue

            col_results = {"valid": True, "violations": []}

            # Check type
            if "type" in constraint_def:
                validator = DataValidator()
                type_valid = validator.validate_data_types(data, {col: constraint_def["type"]})
                if not type_valid.get(col, False):
                    col_results["valid"] = False
                    col_results["violations"].append(f"Invalid type: expected {constraint_def['type']}")

            # Check min/max for numeric
            if "min" in constraint_def or "max" in constraint_def:
                numeric_data = pd.to_numeric(data[col], errors="coerce")
                if "min" in constraint_def:
                    violations = (numeric_data < constraint_def["min"]).sum()
                    if violations > 0:
                        col_results["valid"] = False
                        col_results["violations"].append(
                            f"{violations} values below minimum {constraint_def['min']}"
                        )
                if "max" in constraint_def:
                    violations = (numeric_data > constraint_def["max"]).sum()
                    if violations > 0:
                        col_results["valid"] = False
                        col_results["violations"].append(
                            f"{violations} values above maximum {constraint_def['max']}"
                        )

            # Check allowed values
            if "allowed_values" in constraint_def:
                validator = DataValidator()
                cat_results = validator.validate_categorical_values(
                    data[col], constraint_def["allowed_values"]
                )
                if cat_results["invalid_count"] > 0:
                    col_results["valid"] = False
                    col_results["violations"].append(
                        f"{cat_results['invalid_count']} invalid categorical values"
                    )

            # Check not null
            if constraint_def.get("not_null", False):
                null_count = data[col].isna().sum()
                if null_count > 0:
                    col_results["valid"] = False
                    col_results["violations"].append(f"{null_count} null values found")

            results[col] = col_results

        return results


__all__ = ["DataValidator", "FileValidator", "SchemaValidator"]
