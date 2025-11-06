"""
File loader for various data formats
Author: Agate Jarmakoviča
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
import chardet

from healthdq.utils.logger import get_logger
from healthdq.utils.validators import FileValidator

logger = get_logger(__name__)


class FileLoader:
    """
    Load data from various file formats.

    Supported formats:
    - CSV
    - Excel (xlsx, xls)
    - JSON
    - Parquet
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the file loader."""
        from healthdq.config import get_config

        self.config = config or get_config()
        self.validator = FileValidator()

    def load(
        self,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load data from file.

        Args:
            file_path: Path to file
            file_type: File type (csv, excel, json, parquet)
            **kwargs: Additional arguments passed to pandas readers

        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)

        # Validate file exists
        if not self.validator.validate_file_exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect file type if not specified
        if file_type is None:
            file_type = self._detect_file_type(file_path)

        # Validate file type
        if not self.validator.validate_file_format(file_path, file_type):
            raise ValueError(f"Invalid file format for type {file_type}: {file_path}")

        # Validate file size
        size_info = self.validator.validate_file_size(
            file_path, max_size_mb=self.config.data_processing.max_file_size_mb
        )
        if not size_info["valid"]:
            raise ValueError(
                f"File too large: {size_info['size_mb']:.2f} MB "
                f"(max: {size_info['max_size_mb']} MB)"
            )

        logger.info(f"Loading {file_type} file: {file_path} ({size_info['size_mb']:.2f} MB)")

        # Load based on file type
        if file_type == "csv":
            data = self._load_csv(file_path, **kwargs)
        elif file_type == "excel":
            data = self._load_excel(file_path, **kwargs)
        elif file_type == "json":
            data = self._load_json(file_path, **kwargs)
        elif file_type == "parquet":
            data = self._load_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        logger.info(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        return data

    def _detect_file_type(self, file_path: Path) -> str:
        """Auto-detect file type from extension."""
        extension = file_path.suffix.lower()

        type_map = {
            ".csv": "csv",
            ".xlsx": "excel",
            ".xls": "excel",
            ".json": "json",
            ".parquet": "parquet",
        }

        file_type = type_map.get(extension)
        if not file_type:
            raise ValueError(f"Cannot detect file type for extension: {extension}")

        return file_type

    def _load_csv(self, file_path: Path, **kwargs: Any) -> pd.DataFrame:
        """Load CSV file with encoding detection."""
        # Detect encoding if not specified
        if "encoding" not in kwargs:
            with open(file_path, "rb") as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result["encoding"]
                logger.debug(f"Detected encoding: {encoding}")
        else:
            encoding = kwargs.pop("encoding")

        # Default CSV parameters
        csv_params = {
            "encoding": encoding,
            "low_memory": False,
        }
        csv_params.update(kwargs)

        try:
            data = pd.read_csv(file_path, **csv_params)
        except Exception as e:
            logger.error(f"Failed to load CSV: {str(e)}")
            # Try with different parameters
            logger.info("Retrying with alternative parameters...")
            data = pd.read_csv(file_path, encoding="utf-8", encoding_errors="ignore", low_memory=False)

        return data

    def _load_excel(self, file_path: Path, **kwargs: Any) -> pd.DataFrame:
        """Load Excel file."""
        excel_params = {
            "sheet_name": 0,  # First sheet by default
        }
        excel_params.update(kwargs)

        data = pd.read_excel(file_path, **excel_params)
        return data

    def _load_json(self, file_path: Path, **kwargs: Any) -> pd.DataFrame:
        """Load JSON file."""
        json_params = {
            "orient": "records",  # Default orientation
        }
        json_params.update(kwargs)

        try:
            data = pd.read_json(file_path, **json_params)
        except ValueError:
            # Try reading as plain JSON and converting
            import json

            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Handle different JSON structures
            if isinstance(json_data, list):
                data = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                # Try to find the data array
                for key in ["data", "records", "items", "results"]:
                    if key in json_data and isinstance(json_data[key], list):
                        data = pd.DataFrame(json_data[key])
                        break
                else:
                    # Assume it's a single record
                    data = pd.DataFrame([json_data])
            else:
                raise ValueError("Unsupported JSON structure")

        return data

    def _load_parquet(self, file_path: Path, **kwargs: Any) -> pd.DataFrame:
        """Load Parquet file."""
        data = pd.read_parquet(file_path, **kwargs)
        return data

    def save(
        self,
        data: pd.DataFrame,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Save DataFrame to file.

        Args:
            data: DataFrame to save
            file_path: Output file path
            file_type: File type (csv, excel, json, parquet)
            **kwargs: Additional arguments
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect file type if not specified
        if file_type is None:
            file_type = self._detect_file_type(file_path)

        logger.info(f"Saving data to {file_type} file: {file_path}")

        if file_type == "csv":
            data.to_csv(file_path, index=False, **kwargs)
        elif file_type == "excel":
            data.to_excel(file_path, index=False, **kwargs)
        elif file_type == "json":
            orient = kwargs.pop("orient", "records")
            data.to_json(file_path, orient=orient, indent=2, **kwargs)
        elif file_type == "parquet":
            data.to_parquet(file_path, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        logger.info(f"Data saved successfully: {file_path}")


__all__ = ["FileLoader"]
