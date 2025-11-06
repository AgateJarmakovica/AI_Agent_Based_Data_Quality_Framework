"""
Coordinator agent - orchestrates multi-agent data quality assessment
Author: Agate Jarmakoviča
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import asyncio

from healthdq.agents.base_agent import BaseAgent, DimensionType, ParameterValue
from healthdq.communication.message import AnalysisRequest, AnalysisResponse, create_request_message
from healthdq.communication.router import get_router
from healthdq.utils.logger import get_logger
from healthdq.utils.helpers import generate_hash, merge_results

logger = get_logger(__name__)


class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent orchestrates the multi-agent data quality workflow.

    Responsibilities:
    - Distribute tasks to specialized agents (Precision, Completeness, Reusability)
    - Aggregate results from multiple agents
    - Resolve conflicts between agents
    - Generate comprehensive quality reports
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the coordinator agent."""
        super().__init__(
            agent_id="coordinator",
            role="Multi-agent orchestrator",
            capabilities=["orchestration", "aggregation", "conflict_resolution"],
            config=config,
        )

        self.router = get_router()
        # Map dimension to agent_id with typed keys
        self.specialized_agents: Dict[DimensionType, str] = {}
        # Enable/disable real agent communication (vs simulation)
        self.use_real_agents: bool = False

    async def register_specialized_agent(self, dimension: DimensionType, agent_id: str) -> None:
        """
        Register a specialized agent for a quality dimension.

        Args:
            dimension: Quality dimension (precision, completeness, reusability)
            agent_id: Agent identifier

        Raises:
            ValueError: If dimension is invalid
        """
        valid_dimensions = ["precision", "completeness", "reusability"]
        if dimension not in valid_dimensions:
            raise ValueError(f"Invalid dimension: {dimension}. Must be one of {valid_dimensions}")

        self.specialized_agents[dimension] = agent_id
        logger.info(f"Registered specialized agent: {dimension} -> {agent_id}")

    def enable_real_agent_communication(self, enable: bool = True) -> None:
        """
        Enable or disable real agent communication.

        Args:
            enable: If True, use protocol.send_message() for real agent communication.
                   If False, use simulated analysis (default for testing).
        """
        self.use_real_agents = enable
        logger.info(f"Real agent communication: {'enabled' if enable else 'disabled'}")

    async def analyze(
        self,
        data: pd.DataFrame,
        dimensions: List[DimensionType] = ["precision", "completeness", "reusability"],
        columns: Optional[List[str]] = None,
        parameters: Optional[Dict[str, ParameterValue]] = None,
    ) -> Dict[str, Any]:
        """
        Coordinate comprehensive data quality analysis with input validation.

        Args:
            data: DataFrame to analyze (must be non-empty)
            dimensions: Quality dimensions to assess
            columns: Specific columns to analyze (None for all)
            parameters: Additional parameters for analysis

        Returns:
            Comprehensive analysis results with:
                - status: "success" or "failed"
                - overall_score: Float 0.0-1.0
                - dimension_results: Dict of results per dimension
                - improvement_plan: Dict with actions and priorities
                - metadata: Dict with analysis metadata

        Raises:
            AssertionError: If input validation fails
        """
        # Input validation
        assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame"
        assert not data.empty, "Data cannot be empty"
        assert isinstance(dimensions, list), "Dimensions must be a list"
        assert all(isinstance(d, str) for d in dimensions), "All dimensions must be strings"
        assert all(d in ["precision", "completeness", "reusability"] for d in dimensions), \
            "Invalid dimension(s). Must be: precision, completeness, or reusability"
        if columns is not None:
            assert isinstance(columns, list), "Columns must be a list"
            assert all(isinstance(c, str) for c in columns), "All columns must be strings"

        logger.info(f"Starting coordinated analysis for dimensions: {dimensions}")
        self.task_count += 1

        # Generate data hash for reference
        data_hash = generate_hash(data)

        # Create analysis request
        request = AnalysisRequest(
            data_hash=data_hash,
            dimensions=dimensions,
            columns=columns,
            parameters=parameters or {},
        )

        # Store data temporarily (in production, use shared storage)
        self.memory[f"data_{data_hash}"] = data

        # Dispatch to specialized agents
        analysis_tasks = []
        for dimension in dimensions:
            if dimension in self.specialized_agents:
                agent_id = self.specialized_agents[dimension]
                task = self._analyze_dimension(agent_id, dimension, data, columns, parameters)
                analysis_tasks.append(task)
            else:
                logger.warning(f"No agent registered for dimension: {dimension}")

        # Wait for all analyses to complete
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, dict) and not isinstance(r, Exception)]

        if not valid_results:
            logger.error("All agent analyses failed")
            return {"status": "failed", "error": "All analyses failed"}

        # Aggregate results
        aggregated = self._aggregate_results(valid_results)

        # Generate improvement plan
        improvement_plan = self._generate_improvement_plan(aggregated)

        # Calculate overall score
        overall_score = self._calculate_overall_score(aggregated)

        final_results = {
            "status": "success",
            "overall_score": overall_score,
            "dimension_results": aggregated,
            "improvement_plan": improvement_plan,
            "metadata": {
                "data_hash": data_hash,
                "dimensions_analyzed": dimensions,
                "total_issues": sum(len(r.get("issues", [])) for r in valid_results),
                "confidence": self._calculate_confidence(data, aggregated),
            },
        }

        logger.info(f"Coordinated analysis completed. Overall score: {overall_score:.2f}")
        return final_results

    async def _analyze_dimension(
        self,
        agent_id: str,
        dimension: DimensionType,
        data: pd.DataFrame,
        columns: Optional[List[str]],
        parameters: Optional[Dict[str, ParameterValue]],
    ) -> Dict[str, Any]:
        """
        Analyze a single dimension using specialized agent.

        Supports both real agent communication and simulated analysis.

        Args:
            agent_id: ID of the specialized agent
            dimension: Quality dimension to analyze
            data: DataFrame to analyze
            columns: Optional list of columns
            parameters: Optional parameters

        Returns:
            Analysis results dictionary
        """
        try:
            logger.info(f"Analyzing {dimension} dimension with agent: {agent_id}")

            if self.use_real_agents:
                # Real agent communication via protocol
                result = await self._communicate_with_agent(agent_id, dimension, data, columns, parameters)
            else:
                # Simulated analysis (for testing or when agents not available)
                if dimension == "precision":
                    result = self._simulate_precision_analysis(data, columns)
                elif dimension == "completeness":
                    result = self._simulate_completeness_analysis(data, columns)
                elif dimension == "reusability":
                    result = self._simulate_reusability_analysis(data, columns)
                else:
                    result = {"dimension": dimension, "score": 0.5, "issues": [], "suggestions": []}

            return result

        except Exception as e:
            logger.exception(f"Analysis failed for {dimension}: {e}")
            return {"dimension": dimension, "error": str(e), "score": 0.0}

    async def _communicate_with_agent(
        self,
        agent_id: str,
        dimension: DimensionType,
        data: pd.DataFrame,
        columns: Optional[List[str]],
        parameters: Optional[Dict[str, ParameterValue]],
    ) -> Dict[str, Any]:
        """
        Communicate with a real specialized agent via protocol.

        Args:
            agent_id: ID of the specialized agent
            dimension: Quality dimension
            data: DataFrame to analyze
            columns: Optional columns
            parameters: Optional parameters

        Returns:
            Analysis results from the agent
        """
        try:
            # Generate data reference
            data_hash = generate_hash(data)

            # Create analysis request message
            message = create_request_message(
                sender=self.agent_id,
                receiver=agent_id,
                action="analyze",
                payload={
                    "data_hash": data_hash,
                    "dimension": dimension,
                    "columns": columns,
                    "parameters": parameters or {},
                },
            )

            # Send message and wait for response
            response = await self.protocol.request_response(message, timeout=30.0)

            if response and response.get("status") == "success":
                return response.get("payload", {})
            else:
                error_msg = response.get("error", "Unknown error") if response else "No response"
                logger.error(f"Agent {agent_id} failed: {error_msg}")
                return {"dimension": dimension, "error": error_msg, "score": 0.0}

        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for agent {agent_id}")
            return {"dimension": dimension, "error": "Timeout", "score": 0.0}
        except Exception as e:
            logger.exception(f"Error communicating with agent {agent_id}: {e}")
            return {"dimension": dimension, "error": str(e), "score": 0.0}

    def _simulate_precision_analysis(
        self, data: pd.DataFrame, columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Simulate precision/accuracy analysis using publication formula.

        Publication formula: Accuracy = 1 - (Number of detected anomalies / Total records)

        Uses Isolation Forest + LOF anomaly detection when available,
        falls back to IQR method.
        """
        issues = []
        target_columns = columns or data.columns.tolist()
        numeric_cols = [col for col in target_columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]

        # Anomaly detection (Isolation Forest + LOF)
        total_anomalies = 0
        total_records = len(data)

        if len(numeric_cols) > 0:
            numeric_data = data[numeric_cols].dropna()

            if len(numeric_data) >= 10:
                try:
                    from sklearn.ensemble import IsolationForest
                    from sklearn.neighbors import LocalOutlierFactor

                    # Isolation Forest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
                    iso_predictions = iso_forest.fit_predict(numeric_data)
                    iso_anomalies = (iso_predictions == -1).sum()

                    # Local Outlier Factor
                    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
                    lof_predictions = lof.fit_predict(numeric_data)
                    lof_anomalies = (lof_predictions == -1).sum()

                    # Average of both methods
                    total_anomalies = (iso_anomalies + lof_anomalies) / 2

                    issues.append({
                        "type": "anomalies_detected",
                        "severity": "medium",
                        "description": f"{total_anomalies:.1f} anomalies detected (Isolation Forest + LOF)",
                        "impact_ratio": total_anomalies / len(numeric_data),
                        "method": "Isolation Forest + LOF",
                        "iso_anomalies": int(iso_anomalies),
                        "lof_anomalies": int(lof_anomalies),
                    })

                except ImportError:
                    # Fallback to IQR method
                    total_anomalies = self._detect_anomalies_iqr(data, numeric_cols, issues)
            else:
                # Too few rows, use IQR
                total_anomalies = self._detect_anomalies_iqr(data, numeric_cols, issues)

        # Check for mixed types
        for col in target_columns:
            if col not in data.columns:
                continue

            if data[col].dtype == object:
                unique_types = set(type(x).__name__ for x in data[col].dropna())
                if len(unique_types) > 1:
                    issues.append({
                        "column": col,
                        "type": "mixed_types",
                        "severity": "high",
                        "description": f"Column has mixed types: {unique_types}",
                        "impact_ratio": 1.0,
                    })

        # Publication formula: Accuracy = 1 - (anomalies / total_records)
        accuracy_score = 1.0 - (total_anomalies / total_records) if total_records > 0 else 1.0
        accuracy_score = max(0.0, min(1.0, accuracy_score))

        return {
            "dimension": "precision",
            "score": round(accuracy_score, 4),
            "issues": issues,
            "total_anomalies": round(float(total_anomalies), 2),
            "total_records": total_records,
            "formula": "Accuracy = 1 - (anomalies / total_records)",
            "suggestions": [
                {"action": "handle_anomalies", "columns": numeric_cols, "method": "review_and_correct"},
                {"action": "standardize_formats", "columns": [i["column"] for i in issues if i.get("type") == "mixed_types"]},
            ],
        }

    def _detect_anomalies_iqr(self, data: pd.DataFrame, numeric_cols: List[str], issues: List[Dict[str, Any]]) -> float:
        """Fallback anomaly detection using IQR method."""
        total_outliers = 0

        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((data[col] < (q1 - 1.5 * iqr)) | (data[col] > (q3 + 1.5 * iqr))).sum()

            if outliers > 0:
                total_outliers += outliers
                issues.append({
                    "column": col,
                    "type": "outliers",
                    "severity": "medium",
                    "description": f"{outliers} outliers detected (IQR method)",
                    "impact_ratio": outliers / len(data),
                    "method": "IQR",
                })

        return float(total_outliers)

    def _simulate_completeness_analysis(
        self, data: pd.DataFrame, columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Simulate completeness analysis using publication formula.

        Publication formula: Completeness = 1 - (Number of missing values / Total number of values)

        Per column: Completeness_col = 1 - (Missing in column / Total rows)

        Publication reference: KNN Imputer (n_neighbors=5) increased data
        completeness from 90.57% to nearly 100%.
        """
        issues = []
        target_columns = columns or data.columns.tolist()

        # Calculate total missing values across all columns
        total_cells = len(data) * len(target_columns)
        total_missing = 0

        for col in target_columns:
            if col not in data.columns:
                continue

            missing_count = data[col].isna().sum()
            missing_ratio = missing_count / len(data)
            total_missing += missing_count

            if missing_ratio > 0.05:  # More than 5% missing
                severity = "critical" if missing_ratio > 0.5 else "high" if missing_ratio > 0.2 else "medium"

                # Per-column completeness score
                col_completeness = 1.0 - missing_ratio

                issues.append({
                    "column": col,
                    "type": "missing_values",
                    "severity": severity,
                    "description": f"{missing_count} missing values ({missing_ratio * 100:.1f}%)",
                    "impact_ratio": missing_ratio,
                    "completeness": round(col_completeness, 4),
                })

        # Publication formula: Completeness = 1 - (total_missing / total_cells)
        completeness_score = 1.0 - (total_missing / total_cells) if total_cells > 0 else 1.0
        completeness_score = max(0.0, min(1.0, completeness_score))

        return {
            "dimension": "completeness",
            "score": round(completeness_score, 4),
            "issues": issues,
            "total_missing": int(total_missing),
            "total_cells": int(total_cells),
            "missing_percentage": round((total_missing / total_cells * 100) if total_cells > 0 else 0, 2),
            "formula": "Completeness = 1 - (missing_values / total_values)",
            "suggestions": [
                {
                    "action": "impute_missing",
                    "columns": [i["column"] for i in issues],
                    "methods": ["KNN (n=5)", "mean", "median", "mode"],
                    "reference": "KNN Imputer with n_neighbors=5 (from publication)",
                },
            ],
        }

    def _simulate_reusability_analysis(
        self, data: pd.DataFrame, columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Simulate reusability analysis using publication formula.

        Publication formula:
        Reusability = (Documented processes + Metadata + Version control) / 3

        Simplified: Reusability = (documentation + metadata + version_control) / 3

        Each component is binary (0 or 1), resulting in a score from 0.0 to 1.0.
        """
        issues = []

        # Component 1: Documentation (description, data dictionary)
        has_description = "description" in data.attrs if data.attrs else False
        has_data_dict = "data_dictionary" in data.attrs if data.attrs else False
        documentation = 1.0 if (has_description or has_data_dict) else 0.0

        if not (has_description or has_data_dict):
            issues.append({
                "type": "missing_documentation",
                "severity": "medium",
                "description": "Dataset lacks documentation (description or data dictionary)",
                "impact_ratio": 0.33,
                "component": "documentation",
            })

        # Component 2: Metadata (attrs present, meaningful content)
        has_metadata = bool(data.attrs) and len(data.attrs) > 0
        metadata = 1.0 if has_metadata else 0.0

        if not has_metadata:
            issues.append({
                "type": "missing_metadata",
                "severity": "medium",
                "description": "Dataset lacks metadata attributes",
                "impact_ratio": 0.33,
                "component": "metadata",
            })

        # Component 3: Version Control / Reproducibility (version, source, created_at)
        has_version = "version" in data.attrs if data.attrs else False
        has_source = "source" in data.attrs if data.attrs else False
        has_timestamp = "created_at" in data.attrs if data.attrs else False
        version_control = 1.0 if (has_version or has_source or has_timestamp) else 0.0

        if not (has_version or has_source or has_timestamp):
            issues.append({
                "type": "missing_version_control",
                "severity": "low",
                "description": "Dataset lacks version control info (version, source, or timestamp)",
                "impact_ratio": 0.33,
                "component": "version_control",
            })

        # Check for naming convention issues (bonus check)
        naming_issues_count = 0
        for col in data.columns:
            if not col.islower() or " " in col:
                naming_issues_count += 1

        if naming_issues_count > 0:
            issues.append({
                "type": "naming_convention",
                "severity": "low",
                "description": f"{naming_issues_count} columns have non-standard names",
                "impact_ratio": 0.1,
                "component": "interoperability",
            })

        # Publication formula: Reusability = (sum of 3 components) / 3
        reusability_score = (documentation + metadata + version_control) / 3

        return {
            "dimension": "reusability",
            "score": round(reusability_score, 4),
            "issues": issues,
            "components": {
                "documentation": documentation,
                "metadata": metadata,
                "version_control": version_control,
            },
            "formula": "Reusability = (documentation + metadata + version_control) / 3",
            "suggestions": [
                {"action": "add_metadata", "priority": "high" if metadata == 0 else "low"},
                {"action": "add_documentation", "priority": "high" if documentation == 0 else "low"},
                {"action": "add_version_control", "priority": "medium" if version_control == 0 else "low"},
                {"action": "normalize_column_names", "priority": "low"},
            ],
        }

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple agents with merge support.

        If multiple results exist for the same dimension, they are merged
        using the merge_results helper function.

        Args:
            results: List of analysis results from agents

        Returns:
            Aggregated results dictionary keyed by dimension
        """
        aggregated: Dict[DimensionType, Dict[str, Any]] = {}

        for result in results:
            dimension = result.get("dimension")
            if not dimension:
                logger.warning("Result missing dimension field, skipping")
                continue

            if dimension in aggregated:
                # Merge with existing result for this dimension
                try:
                    existing = aggregated[dimension]
                    merged = merge_results(existing, result)
                    aggregated[dimension] = merged
                    logger.info(f"Merged multiple results for dimension: {dimension}")
                except Exception as e:
                    logger.exception(f"Error merging results for {dimension}: {e}")
                    # Keep the first result if merge fails
            else:
                # First result for this dimension
                aggregated[dimension] = result

        return aggregated

    def _generate_improvement_plan(self, aggregated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate improvement plan based on aggregated results."""
        plan = {
            "actions": [],
            "priority_order": [],
        }

        # Collect all issues
        all_issues = []
        for dimension, results in aggregated_results.items():
            for issue in results.get("issues", []):
                issue["dimension"] = dimension
                all_issues.append(issue)

        # Prioritize issues
        prioritized = self._prioritize_issues(all_issues)

        # Generate actions
        for issue in prioritized:
            action = {
                "dimension": issue["dimension"],
                "column": issue.get("column"),
                "type": issue["type"],
                "severity": issue["severity"],
                "recommended_action": self._recommend_action(issue),
            }
            plan["actions"].append(action)

        # Set priority order
        plan["priority_order"] = [a["type"] for a in plan["actions"]]

        return plan

    def _recommend_action(self, issue: Dict[str, Any]) -> str:
        """Recommend action for an issue."""
        issue_type = issue["type"]

        action_map = {
            "missing_values": "impute_missing_values",
            "outliers": "handle_outliers",
            "mixed_types": "standardize_data_types",
            "naming_convention": "normalize_column_names",
            "missing_metadata": "add_metadata",
        }

        return action_map.get(issue_type, "manual_review")

    def _calculate_overall_score(self, aggregated_results: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        if not aggregated_results:
            return 0.0

        scores = [r.get("score", 0.0) for r in aggregated_results.values()]
        return sum(scores) / len(scores)

    async def suggest_improvements(
        self, data: pd.DataFrame, issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest improvements (coordinator delegates to specialized agents)."""
        # This would be implemented by coordinating with specialized agents
        return []


__all__ = ["CoordinatorAgent"]
