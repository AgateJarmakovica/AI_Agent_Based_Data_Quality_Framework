"""
Human-in-the-Loop (HITL) Module
Author: Agate Jarmakovia

Complete HITL system implementing Active ML best practices from
"Managing the Human in the Loop" (Active Learning for ML).

Components:
- Approval management (existing)
- Review system (existing)
- Feedback collection (existing)
- Model-label disagreement detection (NEW)
- Annotation quality metrics (NEW)
- Annotator management (NEW)
- Active learning integration (NEW)
- Workflow automation (NEW)
"""

# Existing components
from healthdq.hitl.approval import ApprovalManager, ApprovalStatus, ApprovalDecision
from healthdq.hitl.review import DataReview, ImprovementReview
from healthdq.hitl.feedback import FeedbackCollector, FeedbackLearner, FeedbackType

# New components
from healthdq.hitl.disagreement import DisagreementDetector
from healthdq.hitl.quality_metrics import AnnotationQualityMetrics
from healthdq.hitl.annotator_manager import AnnotatorManager, AnnotatorProfile
from healthdq.hitl.active_learning import ActiveLearningStrategy, ActiveLearningPipeline
from healthdq.hitl.workflow import HITLWorkflow, WorkflowStage, TaskStatus

__all__ = [
    # Existing
    "ApprovalManager",
    "ApprovalStatus",
    "ApprovalDecision",
    "DataReview",
    "ImprovementReview",
    "FeedbackCollector",
    "FeedbackLearner",
    "FeedbackType",
    # New
    "DisagreementDetector",
    "AnnotationQualityMetrics",
    "AnnotatorManager",
    "AnnotatorProfile",
    "ActiveLearningStrategy",
    "ActiveLearningPipeline",
    "HITLWorkflow",
    "WorkflowStage",
    "TaskStatus",
]
