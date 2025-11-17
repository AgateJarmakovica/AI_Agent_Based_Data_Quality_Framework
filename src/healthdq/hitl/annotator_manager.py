"""
Annotator Management System
Author: Agate Jarmakoviča

Pārvalda anotētājus, viņu sniegumu un uzdevumu piešķiršanu.
Implementē Active ML labākās prakses annotator management.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
import json
from pathlib import Path

from healthdq.utils.logger import get_logger
from healthdq.hitl.quality_metrics import AnnotationQualityMetrics

logger = get_logger(__name__)


class AnnotatorProfile:
    """Annotator profile with performance tracking."""

    def __init__(
        self,
        annotator_id: str,
        name: str,
        email: Optional[str] = None,
        expertise_level: str = "junior",
        specializations: Optional[List[str]] = None,
    ):
        """
        Initialize annotator profile.

        Args:
            annotator_id: Unique annotator identifier
            name: Annotator name
            email: Contact email
            expertise_level: Expertise level (junior, intermediate, senior, expert)
            specializations: List of domain specializations
        """
        self.annotator_id = annotator_id
        self.name = name
        self.email = email
        self.expertise_level = expertise_level
        self.specializations = specializations or []

        # Performance tracking
        self.created_at = datetime.now().isoformat()
        self.total_annotations = 0
        self.total_reviews = 0
        self.accuracy_scores: List[float] = []
        self.kappa_scores: List[float] = []
        self.speed_metrics: List[float] = []  # annotations per hour

        # Task assignment
        self.assigned_tasks: List[str] = []
        self.completed_tasks: List[str] = []
        self.active_tasks: List[str] = []

        # Qualification status
        self.is_qualified = False
        self.qualification_date: Optional[str] = None
        self.last_assessment_date: Optional[str] = None

        # Feedback and notes
        self.feedback_history: List[Dict[str, Any]] = []
        self.notes: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "annotator_id": self.annotator_id,
            "name": self.name,
            "email": self.email,
            "expertise_level": self.expertise_level,
            "specializations": self.specializations,
            "created_at": self.created_at,
            "total_annotations": self.total_annotations,
            "total_reviews": self.total_reviews,
            "average_accuracy": sum(self.accuracy_scores) / len(self.accuracy_scores) if self.accuracy_scores else 0.0,
            "average_kappa": sum(self.kappa_scores) / len(self.kappa_scores) if self.kappa_scores else 0.0,
            "average_speed": sum(self.speed_metrics) / len(self.speed_metrics) if self.speed_metrics else 0.0,
            "is_qualified": self.is_qualified,
            "qualification_date": self.qualification_date,
            "last_assessment_date": self.last_assessment_date,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
        }


class AnnotatorManager:
    """
    Manages annotators and their performance.

    Implements best practices from "Managing the Human in the Loop":
    - Annotator qualification and assessment
    - Performance tracking and dashboards
    - Task assignment and workload balancing
    - Continuous quality monitoring
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize annotator manager.

        Args:
            storage_path: Path to store annotator profiles
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.annotators: Dict[str, AnnotatorProfile] = {}

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_annotators()

        logger.info("AnnotatorManager initialized")

    def _load_annotators(self):
        """Load annotator profiles from disk."""
        if self.storage_path:
            profiles_file = self.storage_path / "annotators.json"
            if profiles_file.exists():
                with open(profiles_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for annotator_data in data.get("annotators", []):
                        profile = AnnotatorProfile(
                            annotator_id=annotator_data["annotator_id"],
                            name=annotator_data["name"],
                            email=annotator_data.get("email"),
                            expertise_level=annotator_data.get("expertise_level", "junior"),
                            specializations=annotator_data.get("specializations", []),
                        )
                        # Restore performance data
                        profile.total_annotations = annotator_data.get("total_annotations", 0)
                        profile.total_reviews = annotator_data.get("total_reviews", 0)
                        profile.accuracy_scores = annotator_data.get("accuracy_scores", [])
                        profile.kappa_scores = annotator_data.get("kappa_scores", [])
                        profile.is_qualified = annotator_data.get("is_qualified", False)
                        profile.qualification_date = annotator_data.get("qualification_date")

                        self.annotators[profile.annotator_id] = profile

                logger.info(f"Loaded {len(self.annotators)} annotator profiles")

    def _save_annotators(self):
        """Save annotator profiles to disk."""
        if self.storage_path:
            profiles_file = self.storage_path / "annotators.json"
            data = {
                "annotators": [
                    {
                        "annotator_id": profile.annotator_id,
                        "name": profile.name,
                        "email": profile.email,
                        "expertise_level": profile.expertise_level,
                        "specializations": profile.specializations,
                        "created_at": profile.created_at,
                        "total_annotations": profile.total_annotations,
                        "total_reviews": profile.total_reviews,
                        "accuracy_scores": profile.accuracy_scores,
                        "kappa_scores": profile.kappa_scores,
                        "is_qualified": profile.is_qualified,
                        "qualification_date": profile.qualification_date,
                    }
                    for profile in self.annotators.values()
                ]
            }

            with open(profiles_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def register_annotator(
        self,
        name: str,
        email: Optional[str] = None,
        expertise_level: str = "junior",
        specializations: Optional[List[str]] = None,
    ) -> str:
        """
        Register a new annotator.

        Args:
            name: Annotator name
            email: Contact email
            expertise_level: Expertise level
            specializations: Domain specializations

        Returns:
            Annotator ID
        """
        annotator_id = str(uuid.uuid4())

        profile = AnnotatorProfile(
            annotator_id=annotator_id,
            name=name,
            email=email,
            expertise_level=expertise_level,
            specializations=specializations,
        )

        self.annotators[annotator_id] = profile
        self._save_annotators()

        logger.info(f"Registered new annotator: {name} (ID: {annotator_id})")
        return annotator_id

    def assess_annotator(
        self,
        annotator_id: str,
        annotator_labels: List,
        gold_standard_labels: List,
        min_accuracy: float = 0.90,
        min_kappa: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Assess annotator quality using qualification test.

        This implements the "Assess annotator skills" practice from the book.

        Args:
            annotator_id: Annotator ID
            annotator_labels: Labels from annotator
            gold_standard_labels: Known correct labels
            min_accuracy: Minimum acceptable accuracy
            min_kappa: Minimum acceptable kappa

        Returns:
            Assessment results
        """
        if annotator_id not in self.annotators:
            raise ValueError(f"Annotator not found: {annotator_id}")

        profile = self.annotators[annotator_id]

        # Perform assessment
        assessment = AnnotationQualityMetrics.assess_annotator_quality(
            annotator_labels=annotator_labels,
            gold_standard_labels=gold_standard_labels,
            min_accuracy=min_accuracy,
            min_kappa=min_kappa,
        )

        # Update profile
        profile.accuracy_scores.append(assessment["accuracy"])
        profile.kappa_scores.append(assessment["kappa"])
        profile.last_assessment_date = datetime.now().isoformat()

        if assessment["qualified"]:
            profile.is_qualified = True
            if profile.qualification_date is None:
                profile.qualification_date = datetime.now().isoformat()

        self._save_annotators()

        logger.info(
            f"Assessed annotator {profile.name}: "
            f"Qualified={assessment['qualified']}, "
            f"Accuracy={assessment['accuracy']:.2%}, "
            f"Kappa={assessment['kappa']:.3f}"
        )

        return assessment

    def record_annotation_batch(
        self,
        annotator_id: str,
        n_annotations: int,
        time_spent_hours: Optional[float] = None,
    ):
        """
        Record completion of annotation batch.

        Args:
            annotator_id: Annotator ID
            n_annotations: Number of annotations completed
            time_spent_hours: Time spent in hours
        """
        if annotator_id not in self.annotators:
            raise ValueError(f"Annotator not found: {annotator_id}")

        profile = self.annotators[annotator_id]
        profile.total_annotations += n_annotations

        if time_spent_hours is not None and time_spent_hours > 0:
            speed = n_annotations / time_spent_hours
            profile.speed_metrics.append(speed)

        self._save_annotators()

        logger.info(
            f"Recorded {n_annotations} annotations for {profile.name}"
        )

    def assign_task(
        self,
        annotator_id: str,
        task_id: str,
    ) -> bool:
        """
        Assign a task to an annotator.

        Args:
            annotator_id: Annotator ID
            task_id: Task ID to assign

        Returns:
            Success status
        """
        if annotator_id not in self.annotators:
            logger.error(f"Annotator not found: {annotator_id}")
            return False

        profile = self.annotators[annotator_id]
        profile.assigned_tasks.append(task_id)
        profile.active_tasks.append(task_id)
        self._save_annotators()

        logger.info(f"Assigned task {task_id} to {profile.name}")
        return True

    def complete_task(
        self,
        annotator_id: str,
        task_id: str,
    ) -> bool:
        """
        Mark a task as completed.

        Args:
            annotator_id: Annotator ID
            task_id: Task ID

        Returns:
            Success status
        """
        if annotator_id not in self.annotators:
            logger.error(f"Annotator not found: {annotator_id}")
            return False

        profile = self.annotators[annotator_id]

        if task_id in profile.active_tasks:
            profile.active_tasks.remove(task_id)
            profile.completed_tasks.append(task_id)
            self._save_annotators()

            logger.info(f"Task {task_id} completed by {profile.name}")
            return True

        logger.warning(f"Task {task_id} not found in active tasks for {profile.name}")
        return False

    def get_annotator_statistics(self, annotator_id: str) -> Dict[str, Any]:
        """Get detailed statistics for an annotator."""
        if annotator_id not in self.annotators:
            raise ValueError(f"Annotator not found: {annotator_id}")

        profile = self.annotators[annotator_id]

        return {
            "annotator_id": annotator_id,
            "name": profile.name,
            "expertise_level": profile.expertise_level,
            "is_qualified": profile.is_qualified,
            "total_annotations": profile.total_annotations,
            "total_reviews": profile.total_reviews,
            "completed_tasks": len(profile.completed_tasks),
            "active_tasks": len(profile.active_tasks),
            "average_accuracy": (
                sum(profile.accuracy_scores) / len(profile.accuracy_scores)
                if profile.accuracy_scores else None
            ),
            "average_kappa": (
                sum(profile.kappa_scores) / len(profile.kappa_scores)
                if profile.kappa_scores else None
            ),
            "average_speed": (
                sum(profile.speed_metrics) / len(profile.speed_metrics)
                if profile.speed_metrics else None
            ),
            "recent_accuracy_scores": profile.accuracy_scores[-5:],
            "recent_kappa_scores": profile.kappa_scores[-5:],
        }

    def get_all_annotators_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all annotators."""
        return [profile.to_dict() for profile in self.annotators.values()]

    def get_qualified_annotators(self) -> List[str]:
        """Get list of qualified annotator IDs."""
        return [
            annotator_id
            for annotator_id, profile in self.annotators.items()
            if profile.is_qualified
        ]

    def get_available_annotators(
        self,
        max_active_tasks: int = 5,
        require_qualified: bool = True,
    ) -> List[str]:
        """
        Get available annotators for task assignment.

        Args:
            max_active_tasks: Maximum active tasks per annotator
            require_qualified: Only return qualified annotators

        Returns:
            List of available annotator IDs
        """
        available = []

        for annotator_id, profile in self.annotators.items():
            # Check qualification
            if require_qualified and not profile.is_qualified:
                continue

            # Check workload
            if len(profile.active_tasks) < max_active_tasks:
                available.append(annotator_id)

        return available

    def distribute_tasks(
        self,
        task_ids: List[str],
        strategy: str = "balanced",
        max_tasks_per_annotator: int = 10,
    ) -> Dict[str, List[str]]:
        """
        Distribute tasks among available annotators.

        Args:
            task_ids: List of task IDs to distribute
            strategy: Distribution strategy ('balanced', 'expertise', 'speed')
            max_tasks_per_annotator: Maximum tasks per annotator

        Returns:
            Dictionary mapping annotator IDs to assigned task IDs
        """
        available_annotators = self.get_available_annotators(
            max_active_tasks=max_tasks_per_annotator
        )

        if not available_annotators:
            logger.warning("No available annotators for task distribution")
            return {}

        assignment = {annotator_id: [] for annotator_id in available_annotators}

        if strategy == "balanced":
            # Round-robin distribution
            for i, task_id in enumerate(task_ids):
                annotator_id = available_annotators[i % len(available_annotators)]
                assignment[annotator_id].append(task_id)
                self.assign_task(annotator_id, task_id)

        elif strategy == "expertise":
            # Assign to highest expertise first
            sorted_annotators = sorted(
                available_annotators,
                key=lambda aid: self.annotators[aid].expertise_level,
                reverse=True
            )
            for i, task_id in enumerate(task_ids):
                annotator_id = sorted_annotators[i % len(sorted_annotators)]
                assignment[annotator_id].append(task_id)
                self.assign_task(annotator_id, task_id)

        elif strategy == "speed":
            # Assign to fastest annotators
            sorted_annotators = sorted(
                available_annotators,
                key=lambda aid: sum(self.annotators[aid].speed_metrics) / len(self.annotators[aid].speed_metrics)
                if self.annotators[aid].speed_metrics else 0,
                reverse=True
            )
            for i, task_id in enumerate(task_ids):
                annotator_id = sorted_annotators[i % len(sorted_annotators)]
                assignment[annotator_id].append(task_id)
                self.assign_task(annotator_id, task_id)

        logger.info(f"Distributed {len(task_ids)} tasks among {len(available_annotators)} annotators")
        return assignment


__all__ = ["AnnotatorManager", "AnnotatorProfile"]
