"""
Model-Label Disagreement Detection
Author: Agate Jarmakoviča

Detektē un analizē neatbilstības starp modeļa prognozēm un cilvēka anotācijām.
Implementē Active ML labākās prakses no "Managing the Human in the Loop".
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import uuid

from healthdq.utils.logger import get_logger

logger = get_logger(__name__)


class DisagreementDetector:
    """
    Identificē un analizē neatbilstības starp modeļa prognozēm un cilvēka anotācijām.

    Šī klase implementē metodes, kas aprakstītas grāmatā "Managing the Human in the Loop":
    - Programmatic mismatch identification
    - Sampling mismatched cases for review
    - Creating re-labeling queues for confusing cases
    """

    def __init__(self):
        """Initialize disagreement detector."""
        self.mismatches: List[Dict[str, Any]] = []
        self.disagreement_history: List[Dict[str, Any]] = []

    def detect_mismatches(
        self,
        data: pd.DataFrame,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        sample_ids: Optional[Union[np.ndarray, pd.Series, List]] = None,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Identificēt neatbilstības starp patiesajām un prognozētajām etiķetēm.

        Args:
            data: Datu DataFrame
            y_true: Patiesās etiķetes (no cilvēka)
            y_pred: Prognozētās etiķetes (no modeļa)
            sample_ids: Paraugu identifikatori
            threshold: Confidence threshold (optional)

        Returns:
            Dictionary ar mismatch analīzi
        """
        logger.info("Detecting model-label disagreements...")

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if sample_ids is None:
            sample_ids = np.arange(len(y_true))
        else:
            sample_ids = np.array(sample_ids)

        # Find mismatches
        mismatch_mask = y_true != y_pred
        mismatch_indices = np.where(mismatch_mask)[0]

        # Extract mismatched data
        mismatched_data = data.iloc[mismatch_indices].copy() if isinstance(data, pd.DataFrame) else data[mismatch_indices]
        mismatched_true = y_true[mismatch_indices]
        mismatched_pred = y_pred[mismatch_indices]
        mismatched_ids = sample_ids[mismatch_indices]

        # Create mismatch records
        mismatches = []
        for idx, (data_idx, sample_id, true_label, pred_label) in enumerate(
            zip(mismatch_indices, mismatched_ids, mismatched_true, mismatched_pred)
        ):
            mismatch_record = {
                "mismatch_id": str(uuid.uuid4()),
                "sample_id": sample_id,
                "data_index": int(data_idx),
                "true_label": true_label,
                "predicted_label": pred_label,
                "detected_at": datetime.now().isoformat(),
                "reviewed": False,
                "review_decision": None,
                "review_comment": None,
            }

            # Add sample data if available
            if isinstance(data, pd.DataFrame):
                mismatch_record["sample_data"] = data.iloc[data_idx].to_dict()

            mismatches.append(mismatch_record)

        self.mismatches.extend(mismatches)

        # Calculate statistics
        total_samples = len(y_true)
        total_mismatches = len(mismatch_indices)
        mismatch_rate = total_mismatches / total_samples if total_samples > 0 else 0.0

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": total_samples,
            "total_mismatches": total_mismatches,
            "mismatch_rate": mismatch_rate,
            "agreement_rate": 1.0 - mismatch_rate,
            "mismatch_indices": mismatch_indices.tolist(),
            "mismatches": mismatches,
        }

        # Add to history
        self.disagreement_history.append({
            "timestamp": datetime.now().isoformat(),
            "total_mismatches": total_mismatches,
            "mismatch_rate": mismatch_rate,
        })

        logger.info(
            f"Detected {total_mismatches} mismatches out of {total_samples} samples "
            f"({mismatch_rate:.2%} disagreement rate)"
        )

        return analysis

    def detect_mismatches_with_confidence(
        self,
        data: pd.DataFrame,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        y_pred_proba: Optional[Union[np.ndarray, pd.Series]] = None,
        confidence_threshold: float = 0.5,
        sample_ids: Optional[Union[np.ndarray, pd.Series, List]] = None,
    ) -> Dict[str, Any]:
        """
        Detect mismatches with model confidence scores.

        High-confidence mismatches are especially valuable for identifying:
        - Labeling errors
        - Model weaknesses
        - Ambiguous cases

        Args:
            data: Data DataFrame
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (max confidence for each prediction)
            confidence_threshold: Minimum confidence to flag as high-confidence mismatch
            sample_ids: Sample identifiers

        Returns:
            Enhanced mismatch analysis with confidence scores
        """
        # First detect all mismatches
        analysis = self.detect_mismatches(data, y_true, y_pred, sample_ids)

        if y_pred_proba is not None:
            y_pred_proba = np.array(y_pred_proba)

            # Add confidence scores to mismatches
            for mismatch in analysis["mismatches"]:
                data_idx = mismatch["data_index"]
                confidence = float(y_pred_proba[data_idx])
                mismatch["model_confidence"] = confidence
                mismatch["high_confidence_mismatch"] = confidence >= confidence_threshold

            # Calculate high-confidence mismatch statistics
            high_conf_mismatches = [
                m for m in analysis["mismatches"]
                if m.get("high_confidence_mismatch", False)
            ]

            analysis["high_confidence_mismatches"] = len(high_conf_mismatches)
            analysis["high_confidence_mismatch_rate"] = (
                len(high_conf_mismatches) / analysis["total_samples"]
                if analysis["total_samples"] > 0 else 0.0
            )

            logger.info(
                f"Found {len(high_conf_mismatches)} high-confidence mismatches "
                f"(confidence >= {confidence_threshold})"
            )

        return analysis

    def sample_mismatches_for_review(
        self,
        n_samples: int = 10,
        strategy: str = "random",
        priority: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sample mismatched cases for manual review.

        Args:
            n_samples: Number of samples to return
            strategy: Sampling strategy ('random', 'high_confidence', 'diverse')
            priority: Priority filter ('all', 'unreviewed', 'high_confidence')

        Returns:
            List of sampled mismatches for review
        """
        # Filter by priority
        candidates = self.mismatches.copy()

        if priority == "unreviewed":
            candidates = [m for m in candidates if not m["reviewed"]]
        elif priority == "high_confidence":
            candidates = [
                m for m in candidates
                if m.get("high_confidence_mismatch", False)
            ]

        if not candidates:
            logger.warning("No candidates available for sampling")
            return []

        # Sample based on strategy
        if strategy == "random":
            sampled = np.random.choice(
                candidates,
                size=min(n_samples, len(candidates)),
                replace=False
            ).tolist()

        elif strategy == "high_confidence":
            # Sort by confidence (descending)
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.get("model_confidence", 0.0),
                reverse=True
            )
            sampled = sorted_candidates[:n_samples]

        elif strategy == "diverse":
            # Sample diverse cases (different label combinations)
            sampled = self._diverse_sampling(candidates, n_samples)

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        logger.info(f"Sampled {len(sampled)} mismatches for review using '{strategy}' strategy")
        return sampled

    def _diverse_sampling(
        self,
        candidates: List[Dict[str, Any]],
        n_samples: int
    ) -> List[Dict[str, Any]]:
        """Sample diverse cases covering different label combinations."""
        # Group by (true_label, predicted_label) combinations
        from collections import defaultdict
        groups = defaultdict(list)

        for candidate in candidates:
            key = (candidate["true_label"], candidate["predicted_label"])
            groups[key].append(candidate)

        # Sample from each group
        sampled = []
        samples_per_group = max(1, n_samples // len(groups))

        for group_candidates in groups.values():
            group_sample = np.random.choice(
                group_candidates,
                size=min(samples_per_group, len(group_candidates)),
                replace=False
            ).tolist()
            sampled.extend(group_sample)

        # If we need more samples, add randomly
        if len(sampled) < n_samples:
            remaining = [c for c in candidates if c not in sampled]
            additional = np.random.choice(
                remaining,
                size=min(n_samples - len(sampled), len(remaining)),
                replace=False
            ).tolist()
            sampled.extend(additional)

        return sampled[:n_samples]

    def mark_reviewed(
        self,
        mismatch_id: str,
        decision: str,
        comment: Optional[str] = None,
        correct_label: Optional[Any] = None,
    ) -> bool:
        """
        Mark a mismatch as reviewed with decision.

        Args:
            mismatch_id: Mismatch ID
            decision: Review decision ('model_correct', 'human_correct', 'ambiguous', 'relabel')
            comment: Review comment
            correct_label: Corrected label if applicable

        Returns:
            Success status
        """
        for mismatch in self.mismatches:
            if mismatch["mismatch_id"] == mismatch_id:
                mismatch["reviewed"] = True
                mismatch["review_decision"] = decision
                mismatch["review_comment"] = comment
                mismatch["reviewed_at"] = datetime.now().isoformat()

                if correct_label is not None:
                    mismatch["correct_label"] = correct_label

                logger.info(f"Mismatch {mismatch_id} marked as reviewed: {decision}")
                return True

        logger.error(f"Mismatch not found: {mismatch_id}")
        return False

    def create_relabeling_queue(
        self,
        priority: str = "high_confidence",
        max_items: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Create a re-labeling queue for confusing/ambiguous cases.

        This implements the Active ML best practice of adding mismatched
        samples back to the labeling pool for clarification.

        Args:
            priority: Priority for queue ('high_confidence', 'all_unreviewed', 'ambiguous')
            max_items: Maximum items in queue

        Returns:
            Re-labeling queue
        """
        queue = []

        if priority == "high_confidence":
            # High-confidence mismatches (model confident but wrong)
            queue = [
                m for m in self.mismatches
                if m.get("high_confidence_mismatch", False) and not m["reviewed"]
            ]

        elif priority == "all_unreviewed":
            # All unreviewed mismatches
            queue = [m for m in self.mismatches if not m["reviewed"]]

        elif priority == "ambiguous":
            # Previously reviewed as ambiguous
            queue = [
                m for m in self.mismatches
                if m.get("review_decision") == "ambiguous"
            ]

        # Limit queue size
        queue = queue[:max_items]

        logger.info(
            f"Created re-labeling queue with {len(queue)} items (priority: {priority})"
        )

        return queue

    def get_mismatch_statistics(self) -> Dict[str, Any]:
        """Get comprehensive mismatch statistics."""
        if not self.mismatches:
            return {
                "total_mismatches": 0,
                "reviewed": 0,
                "unreviewed": 0,
                "review_decisions": {},
            }

        reviewed = [m for m in self.mismatches if m["reviewed"]]
        unreviewed = [m for m in self.mismatches if not m["reviewed"]]

        # Count review decisions
        decision_counts = {}
        for mismatch in reviewed:
            decision = mismatch.get("review_decision", "unknown")
            decision_counts[decision] = decision_counts.get(decision, 0) + 1

        return {
            "total_mismatches": len(self.mismatches),
            "reviewed": len(reviewed),
            "unreviewed": len(unreviewed),
            "review_rate": len(reviewed) / len(self.mismatches) if self.mismatches else 0.0,
            "review_decisions": decision_counts,
            "high_confidence_mismatches": sum(
                1 for m in self.mismatches if m.get("high_confidence_mismatch", False)
            ),
        }

    def export_for_analysis(
        self,
        output_path: str,
        include_reviewed: bool = True,
    ) -> None:
        """
        Export mismatches to CSV for external analysis.

        Args:
            output_path: Output file path
            include_reviewed: Include reviewed mismatches
        """
        mismatches_to_export = self.mismatches

        if not include_reviewed:
            mismatches_to_export = [m for m in mismatches_to_export if not m["reviewed"]]

        # Convert to DataFrame
        df = pd.DataFrame(mismatches_to_export)
        df.to_csv(output_path, index=False)

        logger.info(f"Exported {len(df)} mismatches to {output_path}")


__all__ = ["DisagreementDetector"]
