"""
Main pipeline orchestrator for healthdq-ai framework
Author: Agate Jarmakoviča
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
from datetime import datetime

from healthdq.config import get_config
from healthdq.utils.logger import get_logger
from healthdq.utils.helpers import create_metadata, save_metadata, generate_hash
from healthdq.utils.validators import DataValidator, FileValidator, SchemaValidator

logger = get_logger(__name__)


class DataQualityPipeline:
    """
    Main pipeline for data quality assessment and improvement.

    This pipeline orchestrates the entire data quality workflow:
    1. Data ingestion from multiple sources
    2. Schema learning and validation
    3. Multi-agent quality assessment
    4. HITL validation and feedback
    5. Data transformation and improvement
    6. Results export and metadata tracking
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the pipeline.

        Args:
            config: Configuration object (if None, uses default)
        """
        self.config = config or get_config()
        self.metadata_history: List[Dict[str, Any]] = []
        self.pipeline_id = generate_hash(f"pipeline_{datetime.now().isoformat()}")

        logger.info(f"Pipeline initialized with ID: {self.pipeline_id}")

    def load_data(
        self, source: Union[str, Path, pd.DataFrame], source_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data from various sources.

        Args:
            source: Data source (file path, URL, or DataFrame)
            source_type: Type of source (csv, json, excel, api, database, fhir)

        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from: {source}")

        # If already a DataFrame, return it
        if isinstance(source, pd.DataFrame):
            data = source
        else:
            # Import loaders dynamically to avoid circular imports
            from healthdq.loaders.file_loader import FileLoader

            loader = FileLoader(self.config)

            if source_type is None:
                # Auto-detect source type
                source_path = Path(source)
                if source_path.suffix == ".csv":
                    source_type = "csv"
                elif source_path.suffix in [".xlsx", ".xls"]:
                    source_type = "excel"
                elif source_path.suffix == ".json":
                    source_type = "json"
                elif source_path.suffix == ".parquet":
                    source_type = "parquet"
                else:
                    raise ValueError(f"Cannot determine source type for: {source}")

            data = loader.load(source, file_type=source_type)

        # Create metadata
        metadata = create_metadata(
            data,
            operation="load_data",
            parameters={"source": str(source), "source_type": source_type},
        )
        self.metadata_history.append(metadata)

        logger.info(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        return data

    def validate_schema(
        self, data: pd.DataFrame, schema_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate data against schema.

        Args:
            data: DataFrame to validate
            schema_config: Schema configuration with column constraints

        Returns:
            Validation results
        """
        logger.info("Validating data schema")

        validator = SchemaValidator()
        results = {"valid": True, "violations": []}

        # Check required columns
        if schema_config and "required_columns" in schema_config:
            required_check = validator.validate_required_columns(data, schema_config["required_columns"])
            if not required_check["valid"]:
                results["valid"] = False
                results["violations"].append(
                    f"Missing required columns: {required_check['missing_columns']}"
                )

        # Check column constraints
        if schema_config and "constraints" in schema_config:
            constraint_results = validator.validate_column_constraints(data, schema_config["constraints"])
            for col, col_result in constraint_results.items():
                if not col_result["valid"]:
                    results["valid"] = False
                    results["violations"].extend(col_result["violations"])

        # Log results
        metadata = create_metadata(
            data, operation="validate_schema", parameters={"schema_config": schema_config}, additional_info=results
        )
        self.metadata_history.append(metadata)

        if results["valid"]:
            logger.info("Schema validation passed")
        else:
            logger.warning(f"Schema validation failed: {len(results['violations'])} violations")

        return results

    async def analyze_quality(
        self, data: pd.DataFrame, dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze data quality using multi-agent system.

        Args:
            data: DataFrame to analyze
            dimensions: Quality dimensions to assess (precision, completeness, reusability)

        Returns:
            Quality analysis results
        """
        logger.info("Starting multi-agent quality analysis")

        # Import agents
        from healthdq.agents.coordinator import CoordinatorAgent

        # Default dimensions
        if dimensions is None:
            dimensions = ["precision", "completeness", "reusability"]

        # Initialize coordinator
        coordinator = CoordinatorAgent(self.config)

        # Run analysis
        results = await coordinator.analyze(data, dimensions=dimensions)

        # Create metadata
        metadata = create_metadata(
            data,
            operation="analyze_quality",
            parameters={"dimensions": dimensions},
            additional_info={"results": results},
        )
        self.metadata_history.append(metadata)

        logger.info(f"Quality analysis completed. Overall score: {results.get('overall_score', 'N/A')}")
        return results

    async def apply_improvements(
        self, data: pd.DataFrame, improvement_plan: Dict[str, Any], require_hitl: bool = True
    ) -> pd.DataFrame:
        """
        Apply data quality improvements.

        Args:
            data: DataFrame to improve
            improvement_plan: Plan with improvements to apply
            require_hitl: Whether to require human validation

        Returns:
            Improved DataFrame
        """
        logger.info("Applying data quality improvements")

        # Import transform module
        from healthdq.rules.transform import DataTransformer

        transformer = DataTransformer(self.config)

        # Apply transformations
        improved_data = await transformer.apply_plan(data, improvement_plan, require_hitl=require_hitl)

        # Create metadata
        metadata = create_metadata(
            improved_data,
            operation="apply_improvements",
            parameters={"improvement_plan": improvement_plan, "require_hitl": require_hitl},
        )
        self.metadata_history.append(metadata)

        logger.info(f"Improvements applied. New data shape: {improved_data.shape}")
        return improved_data

    def calculate_metrics(self, data: pd.DataFrame, original_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate quality metrics.

        Args:
            data: DataFrame to calculate metrics for
            original_data: Original DataFrame (for before/after comparison)

        Returns:
            Calculated metrics
        """
        logger.info("Calculating quality metrics")

        from healthdq.metrics.calculator import MetricsCalculator

        calculator = MetricsCalculator(self.config)
        metrics = calculator.calculate_all(data, original_data=original_data)

        # Create metadata
        metadata = create_metadata(
            data, operation="calculate_metrics", additional_info={"metrics": metrics}
        )
        self.metadata_history.append(metadata)

        logger.info(f"Metrics calculated: {len(metrics)} metrics")
        return metrics

    def export_results(
        self,
        data: pd.DataFrame,
        output_path: Union[str, Path],
        format: str = "csv",
        include_metadata: bool = True,
    ) -> None:
        """
        Export results to file.

        Args:
            data: DataFrame to export
            output_path: Output file path
            format: Output format (csv, excel, json, parquet)
            include_metadata: Whether to include metadata file
        """
        logger.info(f"Exporting results to: {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export data
        if format == "csv":
            data.to_csv(output_path, index=False)
        elif format == "excel":
            data.to_excel(output_path, index=False)
        elif format == "json":
            data.to_json(output_path, orient="records", indent=2)
        elif format == "parquet":
            data.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        # Export metadata if requested
        if include_metadata:
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
            pipeline_metadata = {
                "pipeline_id": self.pipeline_id,
                "timestamp": datetime.now().isoformat(),
                "output_path": str(output_path),
                "format": format,
                "data_shape": data.shape,
                "data_hash": generate_hash(data),
                "history": self.metadata_history,
            }
            save_metadata(pipeline_metadata, metadata_path)
            logger.info(f"Metadata saved to: {metadata_path}")

        logger.info("Results exported successfully")

    async def run(
        self,
        source: Union[str, Path, pd.DataFrame],
        source_type: Optional[str] = None,
        schema_config: Optional[Dict[str, Any]] = None,
        quality_dimensions: Optional[List[str]] = None,
        apply_improvements: bool = True,
        require_hitl: bool = True,
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = "csv",
    ) -> Dict[str, Any]:
        """
        Run the complete data quality pipeline.

        Args:
            source: Data source
            source_type: Type of source
            schema_config: Schema configuration
            quality_dimensions: Quality dimensions to assess
            apply_improvements: Whether to apply improvements
            require_hitl: Whether to require human validation
            output_path: Output file path (if None, no export)
            output_format: Output format

        Returns:
            Pipeline results including data, metrics, and metadata
        """
        logger.info("=" * 80)
        logger.info("Starting Data Quality Pipeline")
        logger.info("=" * 80)

        try:
            # 1. Load data
            data = self.load_data(source, source_type)
            original_data = data.copy()

            # 2. Validate schema
            if schema_config:
                schema_validation = self.validate_schema(data, schema_config)
                if not schema_validation["valid"] and not apply_improvements:
                    raise ValueError(f"Schema validation failed: {schema_validation['violations']}")

            # 3. Analyze quality
            quality_results = await self.analyze_quality(data, quality_dimensions)

            # 4. Apply improvements if requested
            if apply_improvements and quality_results.get("improvement_plan"):
                data = await self.apply_improvements(
                    data, quality_results["improvement_plan"], require_hitl=require_hitl
                )

            # 5. Calculate final metrics
            metrics = self.calculate_metrics(data, original_data=original_data)

            # 6. Export results
            if output_path:
                self.export_results(data, output_path, format=output_format)

            # Prepare results
            results = {
                "pipeline_id": self.pipeline_id,
                "status": "success",
                "data": data,
                "original_data": original_data,
                "quality_results": quality_results,
                "metrics": metrics,
                "metadata": self.metadata_history,
            }

            logger.info("=" * 80)
            logger.info("Pipeline completed successfully")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


__all__ = ["DataQualityPipeline"]
