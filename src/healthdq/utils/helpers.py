"""
Helper utilities for healthdq-ai framework
Author: Agate Jarmakoviča
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np


def generate_hash(data: Union[str, dict, pd.DataFrame], algorithm: str = "sha256") -> str:
    """
    Generate hash for reproducibility tracking.

    Args:
        data: Data to hash (string, dict, or DataFrame)
        algorithm: Hash algorithm to use

    Returns:
        Hex digest of the hash
    """
    if isinstance(data, pd.DataFrame):
        data_str = data.to_json(orient="records", date_format="iso")
    elif isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)

    hash_obj = hashlib.new(algorithm)
    hash_obj.update(data_str.encode("utf-8"))
    return hash_obj.hexdigest()


def create_metadata(
    data: pd.DataFrame,
    operation: str,
    parameters: Optional[Dict[str, Any]] = None,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create metadata for FAIR reproducibility.

    Args:
        data: DataFrame being processed
        operation: Name of the operation
        parameters: Operation parameters
        additional_info: Additional metadata

    Returns:
        Metadata dictionary
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
        "data_shape": data.shape,
        "data_hash": generate_hash(data),
        "columns": list(data.columns),
        "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
    }

    if parameters:
        metadata["parameters"] = parameters

    if additional_info:
        metadata.update(additional_info)

    return metadata


def save_metadata(metadata: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save metadata to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(metadata_path: Union[str, Path]) -> Dict[str, Any]:
    """Load metadata from JSON file."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_data_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive data statistics.

    Returns:
        Dictionary with statistics for each column
    """
    stats = {
        "row_count": len(data),
        "column_count": len(data.columns),
        "columns": {},
    }

    for col in data.columns:
        col_stats = {
            "dtype": str(data[col].dtype),
            "missing_count": int(data[col].isna().sum()),
            "missing_percentage": float(data[col].isna().sum() / len(data) * 100),
            "unique_count": int(data[col].nunique()),
        }

        # Numeric statistics
        if pd.api.types.is_numeric_dtype(data[col]):
            col_stats.update(
                {
                    "mean": float(data[col].mean()) if not data[col].isna().all() else None,
                    "median": float(data[col].median()) if not data[col].isna().all() else None,
                    "std": float(data[col].std()) if not data[col].isna().all() else None,
                    "min": float(data[col].min()) if not data[col].isna().all() else None,
                    "max": float(data[col].max()) if not data[col].isna().all() else None,
                    "q25": float(data[col].quantile(0.25)) if not data[col].isna().all() else None,
                    "q75": float(data[col].quantile(0.75)) if not data[col].isna().all() else None,
                }
            )

        # Categorical statistics
        if pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]):
            value_counts = data[col].value_counts()
            col_stats.update(
                {
                    "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                    "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "top_5_values": value_counts.head(5).to_dict(),
                }
            )

        stats["columns"][col] = col_stats

    return stats


def detect_column_types(data: pd.DataFrame) -> Dict[str, str]:
    """
    Automatically detect semantic column types.

    Returns:
        Dictionary mapping column names to detected types
    """
    column_types = {}

    for col in data.columns:
        # Skip if all null
        if data[col].isna().all():
            column_types[col] = "unknown"
            continue

        # Check for ID columns
        if "id" in col.lower() and data[col].nunique() == len(data):
            column_types[col] = "identifier"
        # Check for numeric
        elif pd.api.types.is_numeric_dtype(data[col]):
            if data[col].nunique() < 10:
                column_types[col] = "categorical_numeric"
            else:
                column_types[col] = "numeric"
        # Check for datetime
        elif pd.api.types.is_datetime64_any_dtype(data[col]):
            column_types[col] = "datetime"
        # Check for boolean
        elif data[col].nunique() <= 2:
            column_types[col] = "boolean"
        # Check for categorical
        elif data[col].nunique() < len(data) * 0.5:
            column_types[col] = "categorical"
        else:
            column_types[col] = "text"

    return column_types


def infer_data_relationships(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Infer relationships between columns.

    Returns:
        List of detected relationships
    """
    relationships = []

    numeric_cols = data.select_dtypes(include=[np.number]).columns

    # Check for correlations
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                corr_value = corr_matrix.loc[col1, col2]
                if abs(corr_value) > 0.7:  # Strong correlation
                    relationships.append(
                        {
                            "type": "correlation",
                            "column1": col1,
                            "column2": col2,
                            "strength": float(corr_value),
                        }
                    )

    # Check for potential foreign keys
    for col in data.columns:
        if "id" in col.lower():
            # Check if values in this ID exist as potential references
            for other_col in data.columns:
                if col != other_col and not "id" in other_col.lower():
                    continue

                # Simple heuristic: if many values overlap
                overlap = len(set(data[col]) & set(data[other_col]))
                if overlap > 0.5 * min(data[col].nunique(), data[other_col].nunique()):
                    relationships.append(
                        {
                            "type": "potential_foreign_key",
                            "column1": col,
                            "column2": other_col,
                            "overlap_ratio": overlap / min(data[col].nunique(), data[other_col].nunique()),
                        }
                    )

    return relationships


def chunk_dataframe(data: pd.DataFrame, chunk_size: int = 1000) -> List[pd.DataFrame]:
    """
    Split DataFrame into chunks for batch processing.

    Args:
        data: DataFrame to chunk
        chunk_size: Size of each chunk

    Returns:
        List of DataFrame chunks
    """
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data.iloc[i : i + chunk_size])
    return chunks


def merge_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge results from multiple agents or processing steps.

    Args:
        results: List of result dictionaries

    Returns:
        Merged results
    """
    merged = {"timestamp": datetime.now().isoformat(), "results": results, "summary": {}}

    # Aggregate statistics
    if all("metrics" in r for r in results):
        all_metrics = {}
        for result in results:
            for metric_name, metric_value in result.get("metrics", {}).items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)

        merged["summary"]["aggregated_metrics"] = {
            name: {
                "mean": np.mean(values),
                "min": np.min(values),
                "max": np.max(values),
                "std": np.std(values),
            }
            for name, values in all_metrics.items()
        }

    return merged


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to standard format.

    - Lowercase
    - Replace spaces with underscores
    - Remove special characters
    """
    new_columns = []
    for col in data.columns:
        # Convert to lowercase
        new_col = col.lower()
        # Replace spaces with underscores
        new_col = new_col.replace(" ", "_")
        # Remove special characters except underscores
        new_col = "".join(c for c in new_col if c.isalnum() or c == "_")
        new_columns.append(new_col)

    data.columns = new_columns
    return data


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def parse_date_flexible(date_str: str) -> Optional[datetime]:
    """
    Try to parse date string with multiple formats.

    Returns:
        Parsed datetime or None if parsing fails
    """
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue

    return None


__all__ = [
    "generate_hash",
    "create_metadata",
    "save_metadata",
    "load_metadata",
    "calculate_data_statistics",
    "detect_column_types",
    "infer_data_relationships",
    "chunk_dataframe",
    "merge_results",
    "format_size",
    "normalize_column_names",
    "safe_divide",
    "parse_date_flexible",
]
