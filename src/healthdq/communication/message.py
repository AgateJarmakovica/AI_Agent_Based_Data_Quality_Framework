"""
Message formats for agent communication
Author: Agate Jarmakoviča
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class MessageType(str, Enum):
    """Types of messages in the system."""

    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    QUERY = "query"
    NOTIFICATION = "notification"
    ERROR = "error"


class MessagePriority(str, Enum):
    """Message priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class AgentMessage(BaseModel):
    """
    Standard message format for agent communication.

    Follows Agent Communication Protocol (ACP) principles.
    """

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    sender: str = Field(description="Agent ID of sender")
    receiver: Optional[str] = Field(default=None, description="Agent ID of receiver (None for broadcast)")
    message_type: MessageType = Field(default=MessageType.REQUEST)
    priority: MessagePriority = Field(default=MessagePriority.MEDIUM)

    # Content
    action: str = Field(description="Action to perform or type of message")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Message payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Conversation tracking
    conversation_id: Optional[str] = Field(default=None)
    reply_to: Optional[str] = Field(default=None, description="ID of message this is replying to")

    # Status
    requires_response: bool = Field(default=False)
    timeout_seconds: Optional[int] = Field(default=None)

    class Config:
        use_enum_values = True


class AnalysisRequest(BaseModel):
    """Request for data quality analysis."""

    data_hash: str
    dimensions: List[str] = Field(default=["precision", "completeness", "reusability"])
    columns: Optional[List[str]] = Field(default=None)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """Response from data quality analysis."""

    status: str  # success, partial, failure
    dimension: str
    score: float = Field(ge=0.0, le=1.0)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CollaborationRequest(BaseModel):
    """Request for agent collaboration."""

    task_id: str
    task_description: str
    required_capabilities: List[str]
    data_reference: str
    deadline: Optional[str] = Field(default=None)


class CollaborationResponse(BaseModel):
    """Response to collaboration request."""

    agent_id: str
    can_participate: bool
    estimated_duration: Optional[int] = Field(default=None, description="Duration in seconds")
    confidence: float = Field(ge=0.0, le=1.0)
    message: Optional[str] = Field(default=None)


class FeedbackMessage(BaseModel):
    """Feedback from human or system."""

    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    source: str  # human, system, agent
    feedback_type: str  # approval, rejection, correction, suggestion
    target_action: str
    original_value: Optional[Any] = Field(default=None)
    corrected_value: Optional[Any] = Field(default=None)
    comment: Optional[str] = Field(default=None)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RuleProposal(BaseModel):
    """Proposal for a new data quality rule."""

    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_type: str  # validation, transformation, imputation
    target_column: Optional[str] = Field(default=None)
    condition: str
    action: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    requires_approval: bool = Field(default=True)


class ImputationProposal(BaseModel):
    """Proposal for missing value imputation."""

    column: str
    row_indices: List[int]
    method: str  # mean, median, mode, ml_prediction, rule_based
    proposed_values: List[Any]
    confidence_scores: List[float]
    rationale: str


class ErrorMessage(BaseModel):
    """Error message."""

    error_type: str
    error_message: str
    stacktrace: Optional[str] = Field(default=None)
    recovery_suggestions: List[str] = Field(default_factory=list)


def create_request_message(
    sender: str,
    receiver: str,
    action: str,
    payload: Dict[str, Any],
    priority: MessagePriority = MessagePriority.MEDIUM,
    requires_response: bool = True,
    conversation_id: Optional[str] = None,
) -> AgentMessage:
    """Helper to create a request message."""
    return AgentMessage(
        sender=sender,
        receiver=receiver,
        message_type=MessageType.REQUEST,
        action=action,
        payload=payload,
        priority=priority,
        requires_response=requires_response,
        conversation_id=conversation_id or str(uuid.uuid4()),
    )


def create_response_message(
    sender: str,
    original_message: AgentMessage,
    payload: Dict[str, Any],
    success: bool = True,
) -> AgentMessage:
    """Helper to create a response message."""
    return AgentMessage(
        sender=sender,
        receiver=original_message.sender,
        message_type=MessageType.RESPONSE,
        action=f"response_to_{original_message.action}",
        payload={**payload, "success": success},
        conversation_id=original_message.conversation_id,
        reply_to=original_message.message_id,
        requires_response=False,
    )


def create_broadcast_message(
    sender: str, action: str, payload: Dict[str, Any], priority: MessagePriority = MessagePriority.LOW
) -> AgentMessage:
    """Helper to create a broadcast message."""
    return AgentMessage(
        sender=sender,
        receiver=None,
        message_type=MessageType.BROADCAST,
        action=action,
        payload=payload,
        priority=priority,
        requires_response=False,
    )


__all__ = [
    "MessageType",
    "MessagePriority",
    "AgentMessage",
    "AnalysisRequest",
    "AnalysisResponse",
    "CollaborationRequest",
    "CollaborationResponse",
    "FeedbackMessage",
    "RuleProposal",
    "ImputationProposal",
    "ErrorMessage",
    "create_request_message",
    "create_response_message",
    "create_broadcast_message",
]
