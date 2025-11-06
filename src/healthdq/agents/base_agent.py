"""
Base agent class for healthdq-ai framework
Author: Agate Jarmakoviča
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
import asyncio

from healthdq.communication.protocol import get_protocol
from healthdq.communication.message import AgentMessage, AnalysisResponse
from healthdq.config import get_config
from healthdq.utils.logger import get_logger

logger = get_logger(__name__)


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

        # Memory (will be enhanced with FeedbackMemory)
        self.memory: Dict[str, Any] = {}

        logger.info(f"Agent initialized: {agent_id} [{role}]")

    async def start(self) -> None:
        """Start the agent."""
        self.is_active = True
        logger.info(f"Agent started: {self.agent_id}")

        # Start message processing in background
        asyncio.create_task(self.protocol.process_messages(self.agent_id))

    async def stop(self) -> None:
        """Stop the agent."""
        self.is_active = False
        self.protocol.unregister_agent(self.agent_id)
        logger.info(f"Agent stopped: {self.agent_id}")

    @abstractmethod
    async def analyze(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResponse:
        """
        Analyze data for quality issues.

        Args:
            data: DataFrame to analyze
            columns: Specific columns to analyze (None for all)
            parameters: Additional parameters

        Returns:
            Analysis response with findings
        """
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
        Learn from human or system feedback.

        Args:
            feedback: Feedback data
        """
        # Store in memory
        feedback_id = feedback.get("feedback_id", len(self.memory))
        self.memory[f"feedback_{feedback_id}"] = feedback

        logger.info(f"Feedback received and stored: {self.agent_id}")

    async def collaborate(
        self,
        other_agents: List[str],
        task: str,
        data_reference: str,
    ) -> Dict[str, Any]:
        """
        Collaborate with other agents.

        Args:
            other_agents: List of agent IDs to collaborate with
            task: Task description
            data_reference: Reference to shared data

        Returns:
            Collaboration results
        """
        from healthdq.communication.router import get_router

        router = get_router()

        results = await router.coordinate_task(
            sender=self.agent_id,
            task_description=task,
            required_capabilities=[],  # Will be determined by task
            data_reference=data_reference,
        )

        logger.info(f"Collaboration completed: {self.agent_id}")
        return results

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
        Calculate confidence score for analysis.

        Args:
            data: Analyzed data
            analysis_results: Analysis results

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence factors
        factors = []

        # Data size factor
        if len(data) > 1000:
            factors.append(0.9)
        elif len(data) > 100:
            factors.append(0.7)
        else:
            factors.append(0.5)

        # Completeness factor
        missing_ratio = data.isna().sum().sum() / (len(data) * len(data.columns))
        factors.append(1.0 - missing_ratio)

        # Calculate average
        confidence = sum(factors) / len(factors)

        return round(confidence, 2)

    def _prioritize_issues(
        self,
        issues: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Prioritize issues by severity and impact.

        Args:
            issues: List of issues

        Returns:
            Sorted list of issues
        """
        # Define severity weights
        severity_weights = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }

        def get_priority_score(issue: Dict[str, Any]) -> float:
            severity = issue.get("severity", "medium")
            impact_ratio = issue.get("impact_ratio", 0.5)
            return severity_weights.get(severity, 2) * impact_ratio

        # Sort by priority score (descending)
        sorted_issues = sorted(
            issues,
            key=get_priority_score,
            reverse=True,
        )

        return sorted_issues


__all__ = ["BaseAgent"]
