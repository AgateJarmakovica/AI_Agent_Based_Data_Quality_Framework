"""
Data transformation engine
Author: Agate Jarmakoviča
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from healthdq.utils.logger import get_logger
from healthdq.utils.helpers import parse_date_flexible, normalize_column_names

logger = get_logger(__name__)


class DataTransformer:
    """
    Apply data quality transformations.

    Transformations include:
    - Missing value imputation (mean, median, mode, forward/backward fill)
    - Outlier handling (IQR, Z-score, MAD methods)
    - Format standardization
    - Data type conversion (numeric, datetime, boolean, categorical)
    - Column normalization
    - Metadata and transformation history tracking

    Features:
    - Input validation for improvement plans
    - Multiple outlier detection methods (IQR, Z-score, MAD)
    - Automatic boolean and categorical type detection
    - Transformation history for reproducibility (Reusability dimension)
    - Version tracking for data lineage
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the transformer."""
        from healthdq.config import get_config

        self.config = config or get_config()

    async def apply_plan(
        self,
        data: pd.DataFrame,
        improvement_plan: Dict[str, Any],
        require_hitl: bool = True,
    ) -> pd.DataFrame:
        """
        Apply improvement plan to data.

        Args:
            data: DataFrame to transform
            improvement_plan: Plan with actions to apply
            require_hitl: Whether to require human approval

        Returns:
            Transformed DataFrame
        """
        logger.info("Applying improvement plan")

        # Validate improvement_plan structure
        if not isinstance(improvement_plan, dict) or "actions" not in improvement_plan:
            logger.warning("Invalid improvement plan structure. Expected dict with 'actions' key.")
            return data

        transformed_data = data.copy()

        # Get actions from plan
        actions = improvement_plan.get("actions", [])

        for action in actions:
            action_type = action.get("recommended_action")
            column = action.get("column")

            logger.info(f"Applying action: {action_type} on column: {column}")

            try:
                if action_type == "impute_missing_values":
                    transformed_data = self.impute_missing(transformed_data, column)

                elif action_type == "handle_outliers":
                    transformed_data = self.handle_outliers(transformed_data, column)

                elif action_type == "standardize_data_types":
                    transformed_data = self.standardize_types(transformed_data, column)

                elif action_type == "normalize_column_names":
                    transformed_data = normalize_column_names(transformed_data)

                elif action_type == "add_metadata":
                    transformed_data = self.add_metadata(transformed_data, transformations=actions)

            except Exception as e:
                logger.error(f"Failed to apply {action_type}: {str(e)}")
                continue

        # Always add metadata with transformation history for reproducibility
        transformed_data = self.add_metadata(transformed_data, transformations=actions)

        logger.info("Improvement plan applied successfully")
        return transformed_data

    def impute_missing(
        self,
        data: pd.DataFrame,
        column: Optional[str] = None,
        method: str = "auto",
    ) -> pd.DataFrame:
        """
        Impute missing values.

        Args:
            data: DataFrame
            column: Column to impute (None for all)
            method: Imputation method (auto, mean, median, mode, forward_fill)

        Returns:
            DataFrame with imputed values
        """
        result = data.copy()
        columns = [column] if column else result.columns

        for col in columns:
            if col not in result.columns:
                continue

            missing_count = result[col].isna().sum()
            if missing_count == 0:
                continue

            logger.debug(f"Imputing {missing_count} missing values in {col}")

            # Determine method if auto
            if method == "auto":
                if pd.api.types.is_numeric_dtype(result[col]):
                    impute_method = "median"
                else:
                    impute_method = "mode"
            else:
                impute_method = method

            # Apply imputation
            if impute_method == "mean":
                result[col].fillna(result[col].mean(), inplace=True)

            elif impute_method == "median":
                result[col].fillna(result[col].median(), inplace=True)

            elif impute_method == "mode":
                mode_value = result[col].mode()
                if len(mode_value) > 0:
                    result[col].fillna(mode_value[0], inplace=True)

            elif impute_method == "forward_fill":
                result[col].fillna(method="ffill", inplace=True)

            elif impute_method == "backward_fill":
                result[col].fillna(method="bfill", inplace=True)

        return result

    def handle_outliers(
        self,
        data: pd.DataFrame,
        column: Optional[str] = None,
        method: str = "clip",
        threshold: float = 1.5,
        detection_method: str = "iqr",
    ) -> pd.DataFrame:
        """
        Handle outliers in numeric columns.

        Args:
            data: DataFrame
            column: Column to process (None for all numeric)
            method: Method (clip, remove, cap)
            threshold: Threshold multiplier (IQR: 1.5, zscore: 3.0, MAD: 3.0)
            detection_method: Detection method (iqr, zscore, mad)

        Returns:
            DataFrame with outliers handled
        """
        result = data.copy()

        if column:
            columns = [column]
        else:
            columns = result.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col not in result.columns:
                continue

            # Detect outliers based on selected method
            if detection_method == "iqr":
                # IQR method (Interquartile Range)
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

            elif detection_method == "zscore":
                # Z-score method (standard deviation based)
                mean = result[col].mean()
                std = result[col].std()

                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std

            elif detection_method == "mad":
                # MAD method (Median Absolute Deviation)
                median = result[col].median()
                mad = np.median(np.abs(result[col] - median))

                # MAD-based bounds (1.4826 is a constant for normal distribution)
                lower_bound = median - threshold * 1.4826 * mad
                upper_bound = median + threshold * 1.4826 * mad

            else:
                logger.warning(f"Unknown detection method '{detection_method}', using IQR")
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

            outliers = ((result[col] < lower_bound) | (result[col] > upper_bound)).sum()

            if outliers == 0:
                continue

            logger.debug(f"Handling {outliers} outliers in {col} using {detection_method} method")

            if method == "clip":
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)

            elif method == "remove":
                result = result[
                    (result[col] >= lower_bound) & (result[col] <= upper_bound)
                ].reset_index(drop=True)

            elif method == "cap":
                result.loc[result[col] < lower_bound, col] = lower_bound
                result.loc[result[col] > upper_bound, col] = upper_bound

        return result

    def standardize_types(
        self,
        data: pd.DataFrame,
        column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Standardize data types.

        Args:
            data: DataFrame
            column: Column to standardize (None for all)

        Returns:
            DataFrame with standardized types
        """
        result = data.copy()
        columns = [column] if column else result.columns

        for col in columns:
            if col not in result.columns:
                continue

            # Try to convert to appropriate type
            if result[col].dtype == object:
                # Try boolean conversion first
                unique_values = result[col].dropna().str.lower().unique()
                boolean_values = {"yes", "no", "true", "false", "y", "n", "1", "0", "t", "f"}

                if len(unique_values) <= 4 and all(val in boolean_values for val in unique_values):
                    # High confidence boolean detection
                    bool_match_count = result[col].dropna().str.lower().isin(boolean_values).sum()
                    if bool_match_count > len(result) * 0.9:  # 90% are boolean-like
                        # Map to boolean
                        bool_map = {
                            "yes": True, "y": True, "true": True, "t": True, "1": True,
                            "no": False, "n": False, "false": False, "f": False, "0": False
                        }
                        result[col] = result[col].str.lower().map(bool_map)
                        logger.debug(f"Converted {col} to boolean")
                        continue

                # Try categorical conversion for low-cardinality columns
                unique_count = result[col].nunique()
                total_count = len(result)
                if unique_count < 50 and (unique_count / total_count) < 0.05:  # Less than 5% unique
                    result[col] = result[col].astype("category")
                    logger.debug(f"Converted {col} to categorical ({unique_count} unique values)")
                    continue

                # Try numeric conversion
                numeric_converted = pd.to_numeric(result[col], errors="coerce")
                if numeric_converted.notna().sum() > len(result) * 0.8:  # 80% convertible
                    result[col] = numeric_converted
                    logger.debug(f"Converted {col} to numeric")
                    continue

                # Try datetime conversion
                datetime_converted = pd.to_datetime(result[col], errors="coerce")
                if datetime_converted.notna().sum() > len(result) * 0.8:
                    result[col] = datetime_converted
                    logger.debug(f"Converted {col} to datetime")
                    continue

        return result

    def add_metadata(
        self,
        data: pd.DataFrame,
        transformations: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Add metadata to DataFrame.

        Args:
            data: DataFrame
            transformations: List of transformations applied (for history tracking)

        Returns:
            DataFrame with metadata
        """
        from datetime import datetime

        result = data.copy()

        # Add basic metadata
        result.attrs["processed_at"] = datetime.now().isoformat()
        result.attrs["shape"] = data.shape
        result.attrs["columns"] = list(data.columns)
        result.attrs["dtypes"] = {col: str(dtype) for col, dtype in data.dtypes.items()}

        # Add transformation history for reproducibility (Reusability dimension)
        if transformations:
            result.attrs["transformations_applied"] = transformations
            logger.debug(f"Added transformation history: {len(transformations)} actions")

        # Add version for reproducibility
        result.attrs["version"] = self.config.get("version", "1.0")
        result.attrs["transformer_version"] = "1.0"

        logger.debug("Metadata added to DataFrame")
        return result

    def remove_duplicates(
        self,
        data: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            data: DataFrame
            subset: Columns to consider for duplicates
            keep: Which duplicates to keep (first, last, False)

        Returns:
            DataFrame without duplicates
        """
        before_count = len(data)
        result = data.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        after_count = len(result)

        removed = before_count - after_count
        logger.info(f"Removed {removed} duplicate rows")

        return result

    def normalize_text(
        self,
        data: pd.DataFrame,
        column: Optional[str] = None,
        lowercase: bool = True,
        strip: bool = True,
    ) -> pd.DataFrame:
        """
        Normalize text columns.

        Args:
            data: DataFrame
            column: Column to normalize (None for all text columns)
            lowercase: Convert to lowercase
            strip: Strip whitespace

        Returns:
            DataFrame with normalized text
        """
        result = data.copy()

        if column:
            columns = [column]
        else:
            columns = result.select_dtypes(include=["object"]).columns

        for col in columns:
            if col not in result.columns:
                continue

            if lowercase:
                result[col] = result[col].astype(str).str.lower()

            if strip:
                result[col] = result[col].astype(str).str.strip()

        return result


__all__ = ["DataTransformer"]
