"""
Annotation Quality Metrics
Author: Agate Jarmakoviča

Novērtē anotāciju kvalitāti, izmantojot dažādus metrikus.
Implementē Active ML labākās prakses - annotator skill assessment.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

from healthdq.utils.logger import get_logger

logger = get_logger(__name__)


class AnnotationQualityMetrics:
    """
    Annotation quality assessment metrics.

    Implementē metrikus, kas aprakstīti grāmatā "Managing the Human in the Loop":
    - Annotator accuracy against gold standard
    - Cohen's Kappa for inter-annotator agreement
    - Fleiss' Kappa for multiple annotators
    - Krippendorff's Alpha for advanced agreement measurement
    """

    @staticmethod
    def calculate_accuracy(
        annotator_labels: Union[List, np.ndarray],
        gold_standard_labels: Union[List, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Calculate annotator accuracy against known labels.

        Args:
            annotator_labels: Labels from annotator
            gold_standard_labels: Known correct labels (gold standard)

        Returns:
            Accuracy metrics
        """
        annotator_labels = np.array(annotator_labels)
        gold_standard_labels = np.array(gold_standard_labels)

        if len(annotator_labels) != len(gold_standard_labels):
            raise ValueError("Label arrays must have the same length")

        # Calculate accuracy
        correct = np.sum(annotator_labels == gold_standard_labels)
        total = len(annotator_labels)
        accuracy = correct / total if total > 0 else 0.0

        # Per-class accuracy
        unique_labels = np.unique(np.concatenate([annotator_labels, gold_standard_labels]))
        per_class_accuracy = {}

        for label in unique_labels:
            mask = gold_standard_labels == label
            if np.sum(mask) > 0:
                class_correct = np.sum(annotator_labels[mask] == label)
                class_total = np.sum(mask)
                per_class_accuracy[str(label)] = class_correct / class_total

        result = {
            "accuracy": accuracy,
            "correct": int(correct),
            "total": int(total),
            "incorrect": int(total - correct),
            "per_class_accuracy": per_class_accuracy,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Annotator accuracy: {accuracy:.2%}")
        return result

    @staticmethod
    def cohen_kappa(
        annotator1_labels: Union[List, np.ndarray],
        annotator2_labels: Union[List, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Calculate Cohen's Kappa coefficient for two annotators.

        Cohen's Kappa measures inter-annotator agreement, accounting for
        agreement by chance.

        Interpretation:
        - < 0: No agreement (worse than chance)
        - 0.0-0.20: Slight agreement
        - 0.21-0.40: Fair agreement
        - 0.41-0.60: Moderate agreement
        - 0.61-0.80: Substantial agreement
        - 0.81-1.00: Almost perfect agreement

        Args:
            annotator1_labels: Labels from first annotator
            annotator2_labels: Labels from second annotator

        Returns:
            Kappa score and interpretation
        """
        annotator1_labels = np.array(annotator1_labels)
        annotator2_labels = np.array(annotator2_labels)

        if len(annotator1_labels) != len(annotator2_labels):
            raise ValueError("Label arrays must have the same length")

        # Calculate observed agreement
        observed_agreement = np.mean(annotator1_labels == annotator2_labels)

        # Calculate expected agreement (by chance)
        unique_labels = np.unique(np.concatenate([annotator1_labels, annotator2_labels]))
        expected_agreement = 0.0

        for label in unique_labels:
            p1 = np.mean(annotator1_labels == label)
            p2 = np.mean(annotator2_labels == label)
            expected_agreement += p1 * p2

        # Calculate Cohen's Kappa
        if expected_agreement == 1.0:
            kappa = 1.0
        else:
            kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

        # Interpret kappa
        if kappa < 0:
            interpretation = "No agreement (worse than chance)"
        elif kappa < 0.20:
            interpretation = "Slight agreement"
        elif kappa < 0.40:
            interpretation = "Fair agreement"
        elif kappa < 0.60:
            interpretation = "Moderate agreement"
        elif kappa < 0.80:
            interpretation = "Substantial agreement"
        else:
            interpretation = "Almost perfect agreement"

        result = {
            "kappa": kappa,
            "observed_agreement": observed_agreement,
            "expected_agreement": expected_agreement,
            "interpretation": interpretation,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Cohen's Kappa: {kappa:.3f} ({interpretation})")
        return result

    @staticmethod
    def fleiss_kappa(
        ratings: Union[np.ndarray, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Calculate Fleiss' Kappa for multiple annotators.

        Fleiss' Kappa extends Cohen's Kappa to multiple raters.

        Args:
            ratings: Matrix where rows are items and columns are categories.
                     Each cell contains the number of raters who assigned
                     that category to that item.
                     Shape: (n_items, n_categories)

        Returns:
            Fleiss' Kappa score and interpretation
        """
        if isinstance(ratings, pd.DataFrame):
            ratings = ratings.values

        ratings = np.array(ratings)
        n_items, n_categories = ratings.shape

        # Number of raters per item
        n_raters = ratings.sum(axis=1)[0]  # Assuming same number of raters per item

        # Calculate p_i (proportion of agreement for each item)
        p_i = (np.sum(ratings ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

        # Calculate P_bar (mean proportion of agreement)
        P_bar = np.mean(p_i)

        # Calculate p_j (proportion of items assigned to each category)
        p_j = np.sum(ratings, axis=0) / (n_items * n_raters)

        # Calculate P_bar_e (expected agreement by chance)
        P_bar_e = np.sum(p_j ** 2)

        # Calculate Fleiss' Kappa
        if P_bar_e == 1.0:
            kappa = 1.0
        else:
            kappa = (P_bar - P_bar_e) / (1.0 - P_bar_e)

        # Interpret kappa (same as Cohen's Kappa)
        if kappa < 0:
            interpretation = "No agreement (worse than chance)"
        elif kappa < 0.20:
            interpretation = "Slight agreement"
        elif kappa < 0.40:
            interpretation = "Fair agreement"
        elif kappa < 0.60:
            interpretation = "Moderate agreement"
        elif kappa < 0.80:
            interpretation = "Substantial agreement"
        else:
            interpretation = "Almost perfect agreement"

        result = {
            "fleiss_kappa": kappa,
            "observed_agreement": P_bar,
            "expected_agreement": P_bar_e,
            "interpretation": interpretation,
            "n_items": n_items,
            "n_categories": n_categories,
            "n_raters": int(n_raters),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Fleiss' Kappa: {kappa:.3f} ({interpretation})")
        return result

    @staticmethod
    def majority_vote(
        annotations: List[List[Any]],
        return_confidence: bool = True,
    ) -> Union[List[Any], List[Dict[str, Any]]]:
        """
        Calculate majority vote across multiple annotators.

        Args:
            annotations: List of annotation lists, one per annotator
                        Example: [[1, 0, 1], [1, 0, 0], [1, 1, 1]]
            return_confidence: Return confidence scores with votes

        Returns:
            Majority vote labels (and confidence if requested)
        """
        # Transpose to get annotations per item
        annotations = np.array(annotations).T

        results = []
        for item_annotations in annotations:
            # Count votes
            vote_counts = Counter(item_annotations)
            majority_label = vote_counts.most_common(1)[0][0]
            majority_count = vote_counts[majority_label]

            if return_confidence:
                confidence = majority_count / len(item_annotations)
                results.append({
                    "label": majority_label,
                    "confidence": confidence,
                    "votes": dict(vote_counts),
                })
            else:
                results.append(majority_label)

        return results

    @staticmethod
    def inter_annotator_agreement_matrix(
        annotations_dict: Dict[str, List],
    ) -> pd.DataFrame:
        """
        Calculate pairwise inter-annotator agreement matrix.

        Args:
            annotations_dict: Dictionary mapping annotator IDs to their labels
                             Example: {"annotator1": [1, 0, 1], "annotator2": [1, 0, 0]}

        Returns:
            Agreement matrix (DataFrame)
        """
        annotator_ids = list(annotations_dict.keys())
        n_annotators = len(annotator_ids)

        # Create agreement matrix
        agreement_matrix = np.zeros((n_annotators, n_annotators))

        for i, annotator1 in enumerate(annotator_ids):
            for j, annotator2 in enumerate(annotator_ids):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    labels1 = np.array(annotations_dict[annotator1])
                    labels2 = np.array(annotations_dict[annotator2])
                    agreement = np.mean(labels1 == labels2)
                    agreement_matrix[i, j] = agreement

        # Create DataFrame
        df = pd.DataFrame(
            agreement_matrix,
            index=annotator_ids,
            columns=annotator_ids
        )

        logger.info(f"Created {n_annotators}x{n_annotators} inter-annotator agreement matrix")
        return df

    @staticmethod
    def assess_annotator_quality(
        annotator_labels: Union[List, np.ndarray],
        gold_standard_labels: Union[List, np.ndarray],
        min_accuracy: float = 0.90,
        min_kappa: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Comprehensive annotator quality assessment.

        This implements the "Assess annotator skills" practice from the book.

        Args:
            annotator_labels: Labels from annotator
            gold_standard_labels: Known correct labels
            min_accuracy: Minimum acceptable accuracy
            min_kappa: Minimum acceptable kappa

        Returns:
            Quality assessment with pass/fail status
        """
        # Calculate accuracy
        accuracy_result = AnnotationQualityMetrics.calculate_accuracy(
            annotator_labels, gold_standard_labels
        )

        # Calculate Cohen's Kappa
        kappa_result = AnnotationQualityMetrics.cohen_kappa(
            annotator_labels, gold_standard_labels
        )

        # Determine if annotator passes qualification
        passes_accuracy = accuracy_result["accuracy"] >= min_accuracy
        passes_kappa = kappa_result["kappa"] >= min_kappa
        qualified = passes_accuracy and passes_kappa

        assessment = {
            "qualified": qualified,
            "accuracy": accuracy_result["accuracy"],
            "accuracy_threshold": min_accuracy,
            "passes_accuracy": passes_accuracy,
            "kappa": kappa_result["kappa"],
            "kappa_threshold": min_kappa,
            "passes_kappa": passes_kappa,
            "total_samples": accuracy_result["total"],
            "correct_samples": accuracy_result["correct"],
            "interpretation": kappa_result["interpretation"],
            "recommendation": (
                "Qualified for independent labeling"
                if qualified
                else "Requires additional training or supervision"
            ),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Annotator quality: Accuracy={accuracy_result['accuracy']:.2%}, "
            f"Kappa={kappa_result['kappa']:.3f}, Qualified={qualified}"
        )

        return assessment

    @staticmethod
    def calculate_confusion_matrix(
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Calculate confusion matrix for annotation quality analysis.

        Args:
            y_true: True labels
            y_pred: Predicted/annotated labels

        Returns:
            Confusion matrix and derived metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Get unique labels
        labels = np.unique(np.concatenate([y_true, y_pred]))

        # Create confusion matrix
        n_labels = len(labels)
        confusion_matrix = np.zeros((n_labels, n_labels), dtype=int)

        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                confusion_matrix[i, j] = np.sum(
                    (y_true == true_label) & (y_pred == pred_label)
                )

        # Convert to DataFrame for readability
        cm_df = pd.DataFrame(
            confusion_matrix,
            index=[f"True_{label}" for label in labels],
            columns=[f"Pred_{label}" for label in labels]
        )

        # Calculate per-class metrics
        per_class_metrics = {}
        for i, label in enumerate(labels):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            tn = confusion_matrix.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_metrics[str(label)] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": int(confusion_matrix[i, :].sum()),
            }

        return {
            "confusion_matrix": confusion_matrix.tolist(),
            "confusion_matrix_df": cm_df,
            "per_class_metrics": per_class_metrics,
            "labels": labels.tolist(),
            "timestamp": datetime.now().isoformat(),
        }


__all__ = ["AnnotationQualityMetrics"]
