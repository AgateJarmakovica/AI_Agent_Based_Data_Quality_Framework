"""
HITL Workflow Management
Author: Agate Jarmakoviča

Pārvalda visu HITL darba plūsmu - no sample selection līdz final approval.
Implementē automatizētu workflow no grāmatas "Managing the Human in the Loop".
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import uuid

from healthdq.utils.logger import get_logger
from healthdq.hitl.active_learning import ActiveLearningStrategy
from healthdq.hitl.disagreement import DisagreementDetector
from healthdq.hitl.annotator_manager import AnnotatorManager
from healthdq.hitl.approval import ApprovalManager
from healthdq.hitl.feedback import FeedbackCollector

logger = get_logger(__name__)


class WorkflowStage(str, Enum):
    """Workflow stages."""
    SAMPLE_SELECTION = "sample_selection"
    TASK_ASSIGNMENT = "task_assignment"
    ANNOTATION = "annotation"
    REVIEW = "review"
    DISAGREEMENT_RESOLUTION = "disagreement_resolution"
    QUALITY_CHECK = "quality_check"
    APPROVAL = "approval"
    COMPLETED = "completed"


class TaskStatus(str, Enum):
    """Task status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    ANNOTATED = "annotated"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REWORK = "needs_rework"
    COMPLETED = "completed"


class HITLWorkflow:
    """
    Complete Human-in-the-Loop workflow manager.

    Implementē pilnu HITL lifecycle:
    1. Sample selection (Active Learning)
    2. Task creation and assignment
    3. Annotation by labelers
    4. Quality check and review
    5. Disagreement resolution
    6. Final approval
    7. Feedback collection and learning

    From the book: "Optimizing the workflow involves streamlining these steps
    so that annotators can complete labeling efficiently."
    """

    def __init__(
        self,
        annotator_manager: Optional[AnnotatorManager] = None,
        approval_manager: Optional[ApprovalManager] = None,
        feedback_collector: Optional[FeedbackCollector] = None,
    ):
        """
        Initialize HITL workflow.

        Args:
            annotator_manager: Annotator management system
            approval_manager: Approval management system
            feedback_collector: Feedback collection system
        """
        self.annotator_manager = annotator_manager or AnnotatorManager()
        self.approval_manager = approval_manager or ApprovalManager()
        self.feedback_collector = feedback_collector or FeedbackCollector()

        self.active_learning = ActiveLearningStrategy()
        self.disagreement_detector = DisagreementDetector()

        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.workflow_sessions: Dict[str, Dict[str, Any]] = {}

        logger.info("HITLWorkflow initialized")

    def create_workflow_session(
        self,
        session_name: str,
        description: Optional[str] = None,
    ) -> str:
        """
        Create a new workflow session.

        Args:
            session_name: Session name
            description: Session description

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())

        session = {
            "session_id": session_id,
            "session_name": session_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "current_stage": WorkflowStage.SAMPLE_SELECTION,
            "tasks": [],
            "metrics": {
                "total_samples": 0,
                "annotated_samples": 0,
                "approved_samples": 0,
                "rejected_samples": 0,
            },
        }

        self.workflow_sessions[session_id] = session
        logger.info(f"Created workflow session: {session_name} (ID: {session_id})")

        return session_id

    def create_annotation_tasks(
        self,
        session_id: str,
        samples: Any,
        task_type: str = "classification",
        priority: str = "normal",
        requires_multiple_annotators: bool = False,
        n_annotators_per_sample: int = 1,
    ) -> List[str]:
        """
        Create annotation tasks from selected samples.

        Args:
            session_id: Workflow session ID
            samples: Samples to annotate (DataFrame or dict)
            task_type: Task type (classification, labeling, review)
            priority: Task priority (low, normal, high, critical)
            requires_multiple_annotators: Require multiple annotators
            n_annotators_per_sample: Number of annotators per sample

        Returns:
            List of created task IDs
        """
        if session_id not in self.workflow_sessions:
            raise ValueError(f"Session not found: {session_id}")

        task_ids = []

        # Convert samples to list of records
        if hasattr(samples, 'to_dict'):
            sample_records = samples.to_dict('records')
        elif isinstance(samples, list):
            sample_records = samples
        else:
            raise ValueError("Samples must be DataFrame or list")

        # Create tasks
        for sample in sample_records:
            task_id = str(uuid.uuid4())

            task = {
                "task_id": task_id,
                "session_id": session_id,
                "task_type": task_type,
                "priority": priority,
                "created_at": datetime.now().isoformat(),
                "status": TaskStatus.PENDING,

                # Sample data
                "sample_data": sample,

                # Assignment
                "assigned_annotators": [],
                "requires_multiple_annotators": requires_multiple_annotators,
                "n_annotators_required": n_annotators_per_sample,

                # Annotations
                "annotations": [],
                "final_annotation": None,

                # Review
                "reviewer": None,
                "review_status": None,
                "review_comment": None,

                # Timestamps
                "assigned_at": None,
                "started_at": None,
                "completed_at": None,
                "reviewed_at": None,
            }

            self.tasks[task_id] = task
            task_ids.append(task_id)

        # Update session
        session = self.workflow_sessions[session_id]
        session["tasks"].extend(task_ids)
        session["metrics"]["total_samples"] += len(task_ids)
        session["current_stage"] = WorkflowStage.TASK_ASSIGNMENT

        logger.info(f"Created {len(task_ids)} annotation tasks for session {session_id}")
        return task_ids

    def assign_tasks_to_annotators(
        self,
        task_ids: List[str],
        strategy: str = "balanced",
    ) -> Dict[str, List[str]]:
        """
        Assign tasks to annotators using specified strategy.

        Args:
            task_ids: List of task IDs to assign
            strategy: Assignment strategy (balanced, expertise, speed)

        Returns:
            Assignment mapping (annotator_id -> task_ids)
        """
        # Get tasks that need assignment
        pending_tasks = []
        for task_id in task_ids:
            if task_id not in self.tasks:
                continue

            task = self.tasks[task_id]
            if task["status"] == TaskStatus.PENDING:
                pending_tasks.append(task_id)

        if not pending_tasks:
            logger.warning("No pending tasks to assign")
            return {}

        # Distribute tasks using annotator manager
        assignment = self.annotator_manager.distribute_tasks(
            task_ids=pending_tasks,
            strategy=strategy,
        )

        # Update task status
        for annotator_id, assigned_task_ids in assignment.items():
            for task_id in assigned_task_ids:
                task = self.tasks[task_id]
                task["assigned_annotators"].append(annotator_id)
                task["status"] = TaskStatus.ASSIGNED
                task["assigned_at"] = datetime.now().isoformat()

        logger.info(
            f"Assigned {len(pending_tasks)} tasks to {len(assignment)} annotators "
            f"using {strategy} strategy"
        )

        return assignment

    def submit_annotation(
        self,
        task_id: str,
        annotator_id: str,
        annotation: Any,
        time_spent_seconds: Optional[float] = None,
    ) -> bool:
        """
        Submit an annotation for a task.

        Args:
            task_id: Task ID
            annotator_id: Annotator ID
            annotation: Annotation result
            time_spent_seconds: Time spent on annotation

        Returns:
            Success status
        """
        if task_id not in self.tasks:
            logger.error(f"Task not found: {task_id}")
            return False

        task = self.tasks[task_id]

        # Verify annotator is assigned
        if annotator_id not in task["assigned_annotators"]:
            logger.error(f"Annotator {annotator_id} not assigned to task {task_id}")
            return False

        # Add annotation
        annotation_record = {
            "annotator_id": annotator_id,
            "annotation": annotation,
            "timestamp": datetime.now().isoformat(),
            "time_spent_seconds": time_spent_seconds,
        }

        task["annotations"].append(annotation_record)

        # Update task status
        if len(task["annotations"]) >= task["n_annotators_required"]:
            task["status"] = TaskStatus.ANNOTATED
            task["completed_at"] = datetime.now().isoformat()

            # If multiple annotators, determine final annotation
            if task["requires_multiple_annotators"]:
                task["final_annotation"] = self._resolve_multiple_annotations(task)
            else:
                task["final_annotation"] = annotation

        else:
            task["status"] = TaskStatus.IN_PROGRESS

        # Record annotation count for annotator
        if time_spent_seconds:
            self.annotator_manager.record_annotation_batch(
                annotator_id=annotator_id,
                n_annotations=1,
                time_spent_hours=time_spent_seconds / 3600,
            )

        logger.info(f"Annotation submitted for task {task_id} by annotator {annotator_id}")
        return True

    def _resolve_multiple_annotations(self, task: Dict[str, Any]) -> Any:
        """Resolve multiple annotations using majority vote."""
        annotations = [ann["annotation"] for ann in task["annotations"]]

        # Simple majority vote
        from collections import Counter
        vote_counts = Counter(annotations)
        majority_annotation = vote_counts.most_common(1)[0][0]

        logger.info(f"Resolved multiple annotations for task {task['task_id']}: {majority_annotation}")
        return majority_annotation

    def review_annotations(
        self,
        task_ids: List[str],
        reviewer_id: str,
        auto_approve_threshold: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Review completed annotations.

        Args:
            task_ids: Task IDs to review
            reviewer_id: Reviewer ID
            auto_approve_threshold: Auto-approve if confidence >= threshold

        Returns:
            Review summary
        """
        reviewed = 0
        auto_approved = 0
        needs_manual_review = 0

        for task_id in task_ids:
            if task_id not in self.tasks:
                continue

            task = self.tasks[task_id]

            if task["status"] != TaskStatus.ANNOTATED:
                continue

            # Check if auto-approval is possible
            if task["requires_multiple_annotators"]:
                # Calculate agreement
                annotations = [ann["annotation"] for ann in task["annotations"]]
                from collections import Counter
                vote_counts = Counter(annotations)
                majority_count = vote_counts.most_common(1)[0][1]
                agreement = majority_count / len(annotations)

                if agreement >= auto_approve_threshold:
                    # Auto-approve
                    task["status"] = TaskStatus.APPROVED
                    task["reviewer"] = reviewer_id
                    task["review_status"] = "auto_approved"
                    task["reviewed_at"] = datetime.now().isoformat()
                    auto_approved += 1
                else:
                    # Needs manual review
                    task["status"] = TaskStatus.UNDER_REVIEW
                    needs_manual_review += 1
            else:
                # Single annotator - needs manual review if no confidence score
                task["status"] = TaskStatus.UNDER_REVIEW
                needs_manual_review += 1

            reviewed += 1

        summary = {
            "reviewed": reviewed,
            "auto_approved": auto_approved,
            "needs_manual_review": needs_manual_review,
            "reviewer_id": reviewer_id,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Reviewed {reviewed} tasks: {auto_approved} auto-approved, "
            f"{needs_manual_review} need manual review"
        )

        return summary

    def detect_and_resolve_disagreements(
        self,
        session_id: str,
        y_true: List,
        y_pred: List,
        create_relabeling_tasks: bool = True,
    ) -> Dict[str, Any]:
        """
        Detect model-label disagreements and create re-labeling tasks.

        Args:
            session_id: Session ID
            y_true: True labels (from humans)
            y_pred: Predicted labels (from model)
            create_relabeling_tasks: Create tasks for re-labeling

        Returns:
            Disagreement analysis
        """
        # Import here to avoid circular imports
        import pandas as pd

        # Detect disagreements
        dummy_data = pd.DataFrame({"index": range(len(y_true))})
        analysis = self.disagreement_detector.detect_mismatches(
            data=dummy_data,
            y_true=y_true,
            y_pred=y_pred,
        )

        if create_relabeling_tasks and analysis["total_mismatches"] > 0:
            # Create re-labeling queue
            relabel_queue = self.disagreement_detector.create_relabeling_queue(
                priority="high_confidence",
                max_items=100,
            )

            # Create tasks for re-labeling
            relabel_samples = []
            for mismatch in relabel_queue:
                relabel_samples.append({
                    "mismatch_id": mismatch["mismatch_id"],
                    "sample_id": mismatch["sample_id"],
                    "true_label": mismatch["true_label"],
                    "predicted_label": mismatch["predicted_label"],
                    "requires_review": True,
                })

            if relabel_samples:
                task_ids = self.create_annotation_tasks(
                    session_id=session_id,
                    samples=relabel_samples,
                    task_type="relabeling",
                    priority="high",
                    requires_multiple_annotators=True,
                    n_annotators_per_sample=2,
                )

                analysis["relabeling_tasks_created"] = len(task_ids)
                logger.info(f"Created {len(task_ids)} re-labeling tasks for disagreements")

        return analysis

    def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """Get workflow session status."""
        if session_id not in self.workflow_sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.workflow_sessions[session_id]

        # Calculate task statistics
        task_statuses = {}
        for task_id in session["tasks"]:
            if task_id in self.tasks:
                status = self.tasks[task_id]["status"]
                task_statuses[status] = task_statuses.get(status, 0) + 1

        return {
            "session_id": session_id,
            "session_name": session["session_name"],
            "status": session["status"],
            "current_stage": session["current_stage"],
            "created_at": session["created_at"],
            "metrics": session["metrics"],
            "task_statuses": task_statuses,
            "total_tasks": len(session["tasks"]),
        }

    def export_annotations(
        self,
        session_id: str,
        output_path: str,
        include_metadata: bool = True,
    ):
        """
        Export annotations for a session.

        Args:
            session_id: Session ID
            output_path: Output file path
            include_metadata: Include task metadata
        """
        import pandas as pd

        if session_id not in self.workflow_sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.workflow_sessions[session_id]

        # Collect annotations
        export_data = []
        for task_id in session["tasks"]:
            if task_id not in self.tasks:
                continue

            task = self.tasks[task_id]

            if task["final_annotation"] is not None:
                record = {
                    "task_id": task_id,
                    "final_annotation": task["final_annotation"],
                }

                if include_metadata:
                    record.update({
                        "task_type": task["task_type"],
                        "n_annotators": len(task["annotations"]),
                        "status": task["status"],
                        "created_at": task["created_at"],
                        "completed_at": task["completed_at"],
                    })

                export_data.append(record)

        # Export to CSV
        df = pd.DataFrame(export_data)
        df.to_csv(output_path, index=False)

        logger.info(f"Exported {len(export_data)} annotations to {output_path}")


__all__ = ["HITLWorkflow", "WorkflowStage", "TaskStatus"]
