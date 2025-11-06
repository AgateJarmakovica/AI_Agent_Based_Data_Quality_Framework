"""
Coordinator agent - orchestrates multi-agent data quality assessment
Author: Agate Jarmakoviča
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import asyncio

from healthdq.agents.base_agent import BaseAgent
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
        self.specialized_agents: Dict[str, str] = {}

    async def register_specialized_agent(self, dimension: str, agent_id: str) -> None:
        """
        Register a specialized agent.

        Args:
            dimension: Quality dimension (precision, completeness, reusability)
            agent_id: Agent identifier
        """
        self.specialized_agents[dimension] = agent_id
        logger.info(f"Registered specialized agent: {dimension} -> {agent_id}")

    async def analyze(
        self,
        data: pd.DataFrame,
        dimensions: List[str] = ["precision", "completeness", "reusability"],
        columns: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Coordinate comprehensive data quality analysis.

        Args:
            data: DataFrame to analyze
            dimensions: Quality dimensions to assess
            columns: Specific columns to analyze
            parameters: Additional parameters

        Returns:
            Comprehensive analysis results
        """
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
        dimension: str,
        data: pd.DataFrame,
        columns: Optional[List[str]],
        parameters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze a single dimension using specialized agent."""
        try:
            # In a real implementation, we would send message to the agent
            # For now, we'll simulate the analysis
            logger.info(f"Analyzing {dimension} dimension with agent: {agent_id}")

            # Simulate analysis based on dimension
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
            logger.error(f"Analysis failed for {dimension}: {str(e)}")
            return {"dimension": dimension, "error": str(e), "score": 0.0}

    def _simulate_precision_analysis(
        self, data: pd.DataFrame, columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Simulate precision analysis."""
        issues = []
        target_columns = columns or data.columns.tolist()

        # Check for format inconsistencies
        for col in target_columns:
            if col not in data.columns:
                continue

            # Check for mixed types
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

            # Check for outliers in numeric columns
            if pd.api.types.is_numeric_dtype(data[col]):
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                outliers = ((data[col] < (q1 - 1.5 * iqr)) | (data[col] > (q3 + 1.5 * iqr))).sum()
                if outliers > 0:
                    issues.append({
                        "column": col,
                        "type": "outliers",
                        "severity": "medium",
                        "description": f"{outliers} potential outliers detected",
                        "impact_ratio": outliers / len(data),
                    })

        # Calculate score
        score = max(0.0, 1.0 - (len(issues) * 0.1))

        return {
            "dimension": "precision",
            "score": score,
            "issues": issues,
            "suggestions": [
                {"action": "standardize_formats", "columns": [i["column"] for i in issues if i["type"] == "mixed_types"]},
                {"action": "handle_outliers", "columns": [i["column"] for i in issues if i["type"] == "outliers"]},
            ],
        }

    def _simulate_completeness_analysis(
        self, data: pd.DataFrame, columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Simulate completeness analysis."""
        issues = []
        target_columns = columns or data.columns.tolist()

        for col in target_columns:
            if col not in data.columns:
                continue

            missing_count = data[col].isna().sum()
            missing_ratio = missing_count / len(data)

            if missing_ratio > 0.05:  # More than 5% missing
                severity = "critical" if missing_ratio > 0.5 else "high" if missing_ratio > 0.2 else "medium"

                issues.append({
                    "column": col,
                    "type": "missing_values",
                    "severity": severity,
                    "description": f"{missing_count} missing values ({missing_ratio * 100:.1f}%)",
                    "impact_ratio": missing_ratio,
                })

        # Calculate score
        avg_completeness = 1.0 - (sum(i["impact_ratio"] for i in issues) / max(len(target_columns), 1))
        score = max(0.0, min(1.0, avg_completeness))

        return {
            "dimension": "completeness",
            "score": score,
            "issues": issues,
            "suggestions": [
                {
                    "action": "impute_missing",
                    "columns": [i["column"] for i in issues],
                    "methods": ["mean", "median", "mode", "ml_prediction"],
                },
            ],
        }

    def _simulate_reusability_analysis(
        self, data: pd.DataFrame, columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Simulate reusability analysis."""
        issues = []

        # Check for standardized column names
        for col in data.columns:
            if not col.islower() or " " in col:
                issues.append({
                    "column": col,
                    "type": "naming_convention",
                    "severity": "low",
                    "description": "Column name not standardized",
                    "impact_ratio": 0.1,
                })

        # Check for documentation (metadata)
        if "description" not in data.attrs:
            issues.append({
                "type": "missing_metadata",
                "severity": "medium",
                "description": "Dataset lacks metadata/documentation",
                "impact_ratio": 0.2,
            })

        # Calculate score
        score = max(0.0, 1.0 - (len(issues) * 0.05))

        return {
            "dimension": "reusability",
            "score": score,
            "issues": issues,
            "suggestions": [
                {"action": "normalize_column_names"},
                {"action": "add_metadata"},
                {"action": "add_data_dictionary"},
            ],
        }

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple agents."""
        aggregated = {}

        for result in results:
            dimension = result.get("dimension")
            if dimension:
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
