"""
Human-in-the-Loop Feedback Collection and Learning
Author: Agate Jarmakoviča

Savāc un apstrādā cilvēka feedback, lai uzlabotu sistēmu.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
import json
from pathlib import Path

from healthdq.utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackType(str):
    """Feedback tipi."""
    APPROVAL = "approval"
    REJECTION = "rejection"
    CORRECTION = "correction"
    SUGGESTION = "suggestion"
    RATING = "rating"
    COMMENT = "comment"


class FeedbackCollector:
    """
    Savāc un saglabā cilvēka feedback.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize feedback collector.

        Args:
            storage_path: Path to store feedback (None = memory only)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.feedback_items: List[Dict[str, Any]] = []

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_existing_feedback()

    def _load_existing_feedback(self):
        """Ielādēt eksistējošo feedback no diska."""
        if self.storage_path:
            feedback_file = self.storage_path / "feedback.jsonl"
            if feedback_file.exists():
                with open(feedback_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            self.feedback_items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

                logger.info(f"Loaded {len(self.feedback_items)} existing feedback items")

    def collect_feedback(
        self,
        feedback_type: str,
        context: Dict[str, Any],
        user_input: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Savākt feedback no lietotāja.

        Args:
            feedback_type: Feedback tips
            context: Konteksts (kas tika vērtēts)
            user_input: Lietotāja ievade
            metadata: Papildu metadati

        Returns:
            Feedback record
        """
        feedback_id = str(uuid.uuid4())

        feedback_record = {
            "feedback_id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            "type": feedback_type,
            "context": context,
            "user_input": user_input,
            "metadata": metadata or {},
        }

        # Save to memory
        self.feedback_items.append(feedback_record)

        # Save to disk if configured
        if self.storage_path:
            self._save_feedback(feedback_record)

        logger.info(f"Feedback collected: {feedback_id} [{feedback_type}]")

        return feedback_record

    def _save_feedback(self, feedback_record: Dict[str, Any]):
        """Saglabāt feedback uz diska."""
        if self.storage_path:
            feedback_file = self.storage_path / "feedback.jsonl"
            with open(feedback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback_record) + "\n")

    def collect_approval_feedback(
        self,
        action: Dict[str, Any],
        approved: bool,
        reason: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Savākt apstiprināšanas/noraidīšanas feedback.

        Args:
            action: Darbība, kas tika vērtēta
            approved: Vai apstiprināts
            reason: Iemesls
            confidence: Lietotāja pārliecība (0-1)

        Returns:
            Feedback record
        """
        return self.collect_feedback(
            feedback_type=FeedbackType.APPROVAL if approved else FeedbackType.REJECTION,
            context={"action": action},
            user_input={
                "approved": approved,
                "reason": reason,
                "confidence": confidence,
            },
        )

    def collect_correction_feedback(
        self,
        original_value: Any,
        ai_suggested_value: Any,
        correct_value: Any,
        column: str,
        row_id: Optional[Any] = None,
        explanation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Savākt korekcijas feedback (kad cilvēks labo AI ieteikumu).

        Args:
            original_value: Oriģinālā vērtība
            ai_suggested_value: AI ieteiktā vērtība
            correct_value: Pareizā vērtība (pēc cilvēka)
            column: Kolonna
            row_id: Rindas ID
            explanation: Paskaidrojums

        Returns:
            Feedback record
        """
        return self.collect_feedback(
            feedback_type=FeedbackType.CORRECTION,
            context={
                "column": column,
                "row_id": row_id,
                "original_value": original_value,
                "ai_suggested_value": ai_suggested_value,
            },
            user_input={
                "correct_value": correct_value,
                "explanation": explanation,
            },
        )

    def collect_rating_feedback(
        self,
        item_id: str,
        item_type: str,
        rating: float,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Savākt vērtējuma feedback (zvaigznes, skala, utt.).

        Args:
            item_id: Novērtētā elementa ID
            item_type: Elementa tips
            rating: Vērtējums (0-1 vai 1-5, atkarībā no sistēmas)
            comment: Komentārs

        Returns:
            Feedback record
        """
        return self.collect_feedback(
            feedback_type=FeedbackType.RATING,
            context={
                "item_id": item_id,
                "item_type": item_type,
            },
            user_input={
                "rating": rating,
                "comment": comment,
            },
        )

    def collect_suggestion_feedback(
        self,
        suggestion: str,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Savākt ieteikumu feedback.

        Args:
            suggestion: Lietotāja ieteikums
            category: Ieteikuma kategorija

        Returns:
            Feedback record
        """
        return self.collect_feedback(
            feedback_type=FeedbackType.SUGGESTION,
            context={"category": category},
            user_input={"suggestion": suggestion},
        )

    def get_feedback_by_type(self, feedback_type: str) -> List[Dict[str, Any]]:
        """Iegūt feedback pēc tipa."""
        return [f for f in self.feedback_items if f["type"] == feedback_type]

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Iegūt feedback statistiku."""
        stats = {
            "total_feedback": len(self.feedback_items),
            "by_type": {},
            "approval_rate": 0.0,
            "correction_count": 0,
        }

        # Count by type
        for feedback in self.feedback_items:
            ftype = feedback["type"]
            stats["by_type"][ftype] = stats["by_type"].get(ftype, 0) + 1

        # Calculate approval rate
        approvals = stats["by_type"].get(FeedbackType.APPROVAL, 0)
        rejections = stats["by_type"].get(FeedbackType.REJECTION, 0)
        total_approvals = approvals + rejections

        if total_approvals > 0:
            stats["approval_rate"] = approvals / total_approvals

        # Correction count
        stats["correction_count"] = stats["by_type"].get(FeedbackType.CORRECTION, 0)

        return stats


class FeedbackLearner:
    """
    Mācās no feedback un ģenerē uzlabojumus.
    """

    def __init__(self):
        """Initialize feedback learner."""
        self.learned_patterns: List[Dict[str, Any]] = []

    def analyze_corrections(self, feedback_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analizēt korekcijas feedback, lai mācītos.

        Args:
            feedback_items: Feedback ieraksti

        Returns:
            Analysis results
        """
        corrections = [f for f in feedback_items if f["type"] == FeedbackType.CORRECTION]

        if not corrections:
            return {"patterns": [], "recommendations": []}

        # Analizēt patterns
        patterns = self._identify_correction_patterns(corrections)

        # Ģenerēt rekomendācijas
        recommendations = self._generate_recommendations(patterns)

        return {
            "total_corrections": len(corrections),
            "patterns": patterns,
            "recommendations": recommendations,
        }

    def _identify_correction_patterns(self, corrections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identificēt korekciju patterns."""
        patterns = []

        # Group by column
        by_column: Dict[str, List] = {}
        for correction in corrections:
            column = correction["context"].get("column")
            if column:
                if column not in by_column:
                    by_column[column] = []
                by_column[column].append(correction)

        # Analyze each column
        for column, col_corrections in by_column.items():
            if len(col_corrections) >= 3:  # Minimum pattern threshold
                pattern = {
                    "column": column,
                    "correction_count": len(col_corrections),
                    "pattern_type": "frequent_corrections",
                    "confidence": min(1.0, len(col_corrections) / 10),
                }
                patterns.append(pattern)

        return patterns

    def _generate_recommendations(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ģenerēt rekomendācijas, balstoties uz patterns."""
        recommendations = []

        for pattern in patterns:
            if pattern["pattern_type"] == "frequent_corrections":
                recommendations.append({
                    "type": "improve_imputation_method",
                    "column": pattern["column"],
                    "reason": f"Frequent manual corrections detected ({pattern['correction_count']} times)",
                    "confidence": pattern["confidence"],
                })

        return recommendations

    def learn_from_approvals(self, feedback_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mācīties no apstiprināšanas/noraidīšanas patterns.

        Returns:
            Learning insights
        """
        approvals = [
            f for f in feedback_items
            if f["type"] == FeedbackType.APPROVAL and f["user_input"].get("approved") is True
        ]

        rejections = [
            f for f in feedback_items
            if f["type"] == FeedbackType.REJECTION or
            (f["type"] == FeedbackType.APPROVAL and f["user_input"].get("approved") is False)
        ]

        insights = {
            "approval_count": len(approvals),
            "rejection_count": len(rejections),
            "approval_rate": len(approvals) / max(len(approvals) + len(rejections), 1),
            "commonly_approved_actions": self._find_common_actions(approvals),
            "commonly_rejected_actions": self._find_common_actions(rejections),
        }

        return insights

    def _find_common_actions(self, feedback_items: List[Dict[str, Any]]) -> List[str]:
        """Atrast bieži sastopamās darbības."""
        action_counts: Dict[str, int] = {}

        for feedback in feedback_items:
            action = feedback["context"].get("action", {})
            action_type = action.get("action_type") or action.get("recommended_action")

            if action_type:
                action_counts[action_type] = action_counts.get(action_type, 0) + 1

        # Sort by frequency
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)

        return [action for action, count in sorted_actions[:5]]


__all__ = ["FeedbackCollector", "FeedbackLearner", "FeedbackType"]
