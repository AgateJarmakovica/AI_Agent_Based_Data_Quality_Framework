"""
Base agent class for healthdq-ai framework
Author: Agate Jarmakoviča
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import asyncio
import numpy as np

from healthdq.communication.protocol import get_protocol
from healthdq.communication.message import AgentMessage, AnalysisResponse
from healthdq.config import get_config
from healthdq.utils.logger import get_logger

logger = get_logger(__name__)


# Type aliases for better type safety
DimensionType = Literal["precision", "completeness", "reusability"]
SeverityType = Literal["critical", "high", "medium", "low"]
ParameterValue = Union[str, float, int, bool]


@dataclass
class FeedbackEntry:
    """
    Structured feedback entry for agent memory.

    Attributes:
        feedback_id: Unique identifier for the feedback
        timestamp: When the feedback was received
        action: What action was taken
        approved: Whether the action was approved
        reason: Reason for approval/rejection
        confidence: Confidence level of the feedback
        metadata: Additional metadata
    """
    feedback_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    action: str = ""
    approved: bool = False
    reason: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "approved": self.approved,
            "reason": self.reason,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Base class for all data quality agents.

    All specialized agents (Precision, Completeness, Reusability)
    inherit from this class.
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        capabilities: List[str],
        config: Optional[Any] = None,
    ):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique agent identifier
            role: Agent role description
            capabilities: List of agent capabilities
            config: Configuration object
        """
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.config = config or get_config()

        # Communication
        self.protocol = get_protocol()
        self.protocol.register_agent(agent_id, capabilities, {"role": role})

        # State
        self.is_active = False
        self.task_count = 0
        self._background_tasks: List[asyncio.Task] = []

        # Enhanced memory with FeedbackEntry structure
        self.memory: Dict[str, FeedbackEntry] = {}

        logger.info(f"Agent initialized: {agent_id} [{role}]")

    async def start(self) -> None:
        """Start the agent with proper task management."""
        try:
            self.is_active = True
            logger.info(f"Agent started: {self.agent_id}")

            # Start message processing in background and track the task
            task = asyncio.create_task(self.protocol.process_messages(self.agent_id))
            self._background_tasks.append(task)
        except Exception as e:
            logger.exception(f"Error starting agent {self.agent_id}: {e}")
            raise

    async def stop(self) -> None:
        """Stop the agent and cleanup all background tasks."""
        try:
            self.is_active = False

            # Cancel all background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            self._background_tasks.clear()

            # Unregister from protocol
            self.protocol.unregister_agent(self.agent_id)
            logger.info(f"Agent stopped: {self.agent_id}")
        except Exception as e:
            logger.exception(f"Error stopping agent {self.agent_id}: {e}")
            raise

    @abstractmethod
    async def analyze(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        parameters: Optional[Dict[str, ParameterValue]] = None,
    ) -> AnalysisResponse:
        """
        Analyze data for quality issues.

        Args:
            data: DataFrame to analyze (must be non-empty)
            columns: Specific columns to analyze (None for all)
            parameters: Additional parameters for analysis

        Returns:
            Analysis response with findings

        Raises:
            ValueError: If data is not a DataFrame or is empty
            TypeError: If columns is not a list of strings
        """
        # Input validation (to be called by implementations)
        assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame"
        assert not data.empty, "Data cannot be empty"
        if columns is not None:
            assert isinstance(columns, list), "Columns must be a list"
            assert all(isinstance(c, str) for c in columns), "All columns must be strings"
        if parameters is not None:
            assert isinstance(parameters, dict), "Parameters must be a dictionary"

        pass

    @abstractmethod
    async def suggest_improvements(
        self,
        data: pd.DataFrame,
        issues: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Suggest improvements for identified issues.

        Args:
            data: DataFrame being analyzed
            issues: List of identified issues

        Returns:
            List of improvement suggestions
        """
        pass

    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Learn from human or system feedback using structured FeedbackEntry.

        Args:
            feedback: Feedback data containing:
                - feedback_id: Unique identifier
                - action: What action was evaluated
                - approved: Whether it was approved
                - reason: Reason for the decision
                - confidence: Confidence level (optional)
                - metadata: Additional info (optional)
        """
        try:
            # Create structured feedback entry
            feedback_id = feedback.get("feedback_id", f"fb_{len(self.memory)}")
            entry = FeedbackEntry(
                feedback_id=feedback_id,
                action=feedback.get("action", ""),
                approved=feedback.get("approved", False),
                reason=feedback.get("reason", ""),
                confidence=feedback.get("confidence", 0.0),
                metadata=feedback.get("metadata", {}),
            )

            # Store in structured memory
            self.memory[feedback_id] = entry

            logger.info(f"Feedback received and stored: {self.agent_id} - {feedback_id}")
        except Exception as e:
            logger.exception(f"Error storing feedback for {self.agent_id}: {e}")
            raise

    async def collaborate(
        self,
        other_agents: List[str],
        task: str,
        data_reference: str,
    ) -> Dict[str, Any]:
        """
        Collaborate with other agents with proper error handling.

        Args:
            other_agents: List of agent IDs to collaborate with
            task: Task description
            data_reference: Reference to shared data

        Returns:
            Collaboration results

        Raises:
            RuntimeError: If collaboration fails
        """
        try:
            from healthdq.communication.router import get_router

            router = get_router()

            results = await router.coordinate_task(
                sender=self.agent_id,
                task_description=task,
                required_capabilities=[],  # Will be determined by task
                data_reference=data_reference,
            )

            logger.info(f"Collaboration completed: {self.agent_id} with {len(other_agents)} agents")
            return results
        except Exception as e:
            logger.exception(f"Error during collaboration for {self.agent_id}: {e}")
            raise RuntimeError(f"Collaboration failed: {e}") from e

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "capabilities": self.capabilities,
            "is_active": self.is_active,
            "task_count": self.task_count,
            "memory_size": len(self.memory),
        }

    def _calculate_confidence(
        self,
        data: pd.DataFrame,
        analysis_results: Dict[str, Any],
    ) -> float:
        """
        Calculate confidence score for analysis with enhanced factors.

        Includes data size, completeness, and dimension scores for better accuracy.

        Args:
            data: Analyzed data
            analysis_results: Analysis results with optional dimension_results

        Returns:
            Confidence score (0.0 to 1.0)
        """
        factors = []

        # 1. Data size factor (larger datasets = higher confidence)
        if len(data) > 1000:
            factors.append(0.9)
        elif len(data) > 100:
            factors.append(0.7)
        else:
            factors.append(0.5)

        # 2. Completeness factor (less missing data = higher confidence)
        if not data.empty and len(data.columns) > 0:
            missing_ratio = data.isna().sum().sum() / (len(data) * len(data.columns))
            factors.append(1.0 - missing_ratio)
        else:
            factors.append(0.5)

        # 3. Dimension scores factor (if available)
        dimension_results = analysis_results.get("dimension_results", {})
        if dimension_results and isinstance(dimension_results, dict):
            dimension_scores = [
                result.get("score", 0.5)
                for result in dimension_results.values()
                if isinstance(result, dict)
            ]
            if dimension_scores:
                mean_dimension_score = np.mean(dimension_scores)
                factors.append(mean_dimension_score)

        # 4. Statistical variance factor (lower variance = higher confidence)
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Normalized coefficient of variation
                cvs = [data[col].std() / (data[col].mean() + 1e-10) for col in numeric_cols if data[col].std() > 0]
                if cvs:
                    # Lower CV = more consistent data = higher confidence
                    avg_cv = np.mean(cvs)
                    cv_confidence = 1.0 / (1.0 + avg_cv)  # Convert to 0-1 range
                    factors.append(cv_confidence)
        except Exception as e:
            logger.debug(f"Could not calculate variance factor: {e}")

        # Calculate weighted average
        confidence = sum(factors) / len(factors) if factors else 0.5

        return round(float(confidence), 2)

    def _prioritize_issues(
        self,
        issues: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Prioritize issues by severity and impact with enhanced scoring.

        Args:
            issues: List of issues with 'severity' and 'impact_ratio' fields

        Returns:
            Sorted list of issues (highest priority first)
        """
        # Define severity weights (critical > high > medium > low)
        severity_weights: Dict[str, int] = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }

        def get_priority_score(issue: Dict[str, Any]) -> float:
            """Calculate priority score for an issue."""
            severity = issue.get("severity", "medium")
            impact_ratio = issue.get("impact_ratio", 0.5)

            # Base score from severity
            base_score = severity_weights.get(severity, 2)

            # Multiply by impact ratio (how much data is affected)
            priority = float(base_score) * float(impact_ratio)

            return priority

        # Sort by priority score (descending - highest priority first)
        sorted_issues = sorted(
            issues,
            key=get_priority_score,
            reverse=True,
        )

        return sorted_issues

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of agent's feedback memory.

        Returns:
            Dictionary with memory statistics and recent feedback
        """
        total_feedback = len(self.memory)
        if total_feedback == 0:
            return {
                "total_feedback": 0,
                "approved_count": 0,
                "rejected_count": 0,
                "average_confidence": 0.0,
                "recent_feedback": [],
            }

        approved_count = sum(1 for entry in self.memory.values() if entry.approved)
        rejected_count = total_feedback - approved_count

        confidences = [entry.confidence for entry in self.memory.values() if entry.confidence > 0]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # Get 5 most recent feedback entries
        recent_entries = sorted(
            self.memory.values(),
            key=lambda x: x.timestamp,
            reverse=True,
        )[:5]

        return {
            "total_feedback": total_feedback,
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "average_confidence": round(float(avg_confidence), 2),
            "recent_feedback": [entry.to_dict() for entry in recent_entries],
        }


__all__ = ["BaseAgent", "FeedbackEntry", "DimensionType", "SeverityType", "ParameterValue"]
