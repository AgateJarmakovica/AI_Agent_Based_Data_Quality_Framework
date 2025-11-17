"""
Active Learning Integration for Human-in-the-Loop
Author: Agate Jarmakoviča

Integrē active learning metodes ar HITL pipeline.
Implementē uncertainty sampling un balanced sampling no grāmatas.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime

from healthdq.utils.logger import get_logger

logger = get_logger(__name__)


class ActiveLearningStrategy:
    """
    Active Learning strategies for intelligent sample selection.

    Implementē metodes no "Managing the Human in the Loop":
    - Uncertainty sampling (select most uncertain predictions)
    - Balanced sampling (prevent dataset imbalance)
    - Diversity sampling (cover different data regions)
    - Model-disagreement sampling (where models disagree)
    """

    def __init__(self):
        """Initialize active learning strategy."""
        self.selection_history: List[Dict[str, Any]] = []

    def uncertainty_sampling(
        self,
        unlabeled_data: pd.DataFrame,
        prediction_probabilities: np.ndarray,
        n_samples: int = 100,
        strategy: str = "least_confident",
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Select most uncertain samples for labeling using uncertainty sampling.

        This is the core Active ML technique - intelligently selecting the most
        informative samples that will improve the model the most.

        Args:
            unlabeled_data: Unlabeled data pool
            prediction_probabilities: Model prediction probabilities (shape: [n_samples, n_classes])
            n_samples: Number of samples to select
            strategy: Uncertainty strategy ('least_confident', 'margin', 'entropy')

        Returns:
            Selected samples and their indices
        """
        logger.info(f"Performing uncertainty sampling with strategy: {strategy}")

        if strategy == "least_confident":
            # Select samples where model is least confident
            # Confidence = max probability
            confidences = np.max(prediction_probabilities, axis=1)
            uncertainties = 1.0 - confidences  # Higher uncertainty = lower confidence

        elif strategy == "margin":
            # Margin sampling: difference between top 2 predictions
            # Small margin = high uncertainty
            sorted_probs = np.sort(prediction_probabilities, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]
            uncertainties = 1.0 - margins  # Lower margin = higher uncertainty

        elif strategy == "entropy":
            # Entropy-based uncertainty
            # Higher entropy = more uncertain
            epsilon = 1e-10  # Prevent log(0)
            entropies = -np.sum(
                prediction_probabilities * np.log(prediction_probabilities + epsilon),
                axis=1
            )
            # Normalize to [0, 1]
            max_entropy = np.log(prediction_probabilities.shape[1])
            uncertainties = entropies / max_entropy

        else:
            raise ValueError(f"Unknown uncertainty strategy: {strategy}")

        # Select top uncertain samples
        uncertain_indices = np.argsort(uncertainties)[::-1][:n_samples]
        selected_samples = unlabeled_data.iloc[uncertain_indices].copy()

        # Add uncertainty scores to samples
        selected_samples["uncertainty_score"] = uncertainties[uncertain_indices]

        # Record selection
        self.selection_history.append({
            "timestamp": datetime.now().isoformat(),
            "strategy": f"uncertainty_{strategy}",
            "n_samples": n_samples,
            "mean_uncertainty": float(np.mean(uncertainties[uncertain_indices])),
            "min_uncertainty": float(np.min(uncertainties[uncertain_indices])),
            "max_uncertainty": float(np.max(uncertainties[uncertain_indices])),
        })

        logger.info(
            f"Selected {len(selected_samples)} samples with mean uncertainty: "
            f"{np.mean(uncertainties[uncertain_indices]):.3f}"
        )

        return selected_samples, uncertain_indices

    def balanced_sampling(
        self,
        unlabeled_data: pd.DataFrame,
        predictions: np.ndarray,
        n_samples: int = 100,
        class_ratios: Optional[Dict[Any, float]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Sample with class balance to prevent imbalanced datasets.

        From the book: "To prevent imbalanced datasets, we can actively sample
        minority classes at higher rates during data collection."

        Args:
            unlabeled_data: Unlabeled data pool
            predictions: Model predictions (for stratification)
            n_samples: Total number of samples to select
            class_ratios: Desired class ratios (if None, uses equal distribution)

        Returns:
            Balanced samples and their indices
        """
        logger.info("Performing balanced sampling to prevent dataset imbalance")

        unique_classes = np.unique(predictions)
        n_classes = len(unique_classes)

        # Determine target counts per class
        if class_ratios is None:
            # Equal distribution
            samples_per_class = n_samples // n_classes
            class_ratios = {cls: 1.0 / n_classes for cls in unique_classes}
        else:
            # User-specified ratios
            samples_per_class = {}
            for cls, ratio in class_ratios.items():
                samples_per_class[cls] = int(n_samples * ratio)

        # Sample from each class
        selected_indices = []

        for cls in unique_classes:
            class_mask = predictions == cls
            class_indices = np.where(class_mask)[0]

            if len(class_indices) == 0:
                continue

            # Determine sample count for this class
            if isinstance(samples_per_class, dict):
                n_class_samples = samples_per_class.get(cls, 0)
            else:
                n_class_samples = samples_per_class

            # Sample from this class
            n_to_sample = min(n_class_samples, len(class_indices))
            sampled_indices = np.random.choice(
                class_indices,
                size=n_to_sample,
                replace=False
            )
            selected_indices.extend(sampled_indices)

        selected_indices = np.array(selected_indices)
        selected_samples = unlabeled_data.iloc[selected_indices].copy()

        # Add class information
        selected_samples["predicted_class"] = predictions[selected_indices]

        # Record selection
        class_distribution = pd.Series(predictions[selected_indices]).value_counts().to_dict()
        self.selection_history.append({
            "timestamp": datetime.now().isoformat(),
            "strategy": "balanced_sampling",
            "n_samples": len(selected_samples),
            "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
            "target_ratios": {str(k): float(v) for k, v in class_ratios.items()},
        })

        logger.info(
            f"Selected {len(selected_samples)} balanced samples. "
            f"Class distribution: {class_distribution}"
        )

        return selected_samples, selected_indices

    def diversity_sampling(
        self,
        unlabeled_data: pd.DataFrame,
        feature_vectors: np.ndarray,
        n_samples: int = 100,
        method: str = "kmeans",
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Sample diverse examples to cover different regions of feature space.

        Args:
            unlabeled_data: Unlabeled data pool
            feature_vectors: Feature representations (embeddings)
            n_samples: Number of samples to select
            method: Diversity method ('kmeans', 'random', 'max_distance')

        Returns:
            Diverse samples and their indices
        """
        logger.info(f"Performing diversity sampling with method: {method}")

        if method == "kmeans":
            # Use k-means clustering to find diverse samples
            from sklearn.cluster import KMeans

            # Cluster into n_samples clusters
            kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_vectors)

            # Select one sample from each cluster (closest to centroid)
            selected_indices = []
            for cluster_id in range(n_samples):
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]

                if len(cluster_indices) == 0:
                    continue

                # Find sample closest to cluster center
                cluster_features = feature_vectors[cluster_indices]
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)

            selected_indices = np.array(selected_indices)

        elif method == "random":
            # Simple random sampling (baseline)
            selected_indices = np.random.choice(
                len(unlabeled_data),
                size=n_samples,
                replace=False
            )

        elif method == "max_distance":
            # Greedy max-distance sampling
            selected_indices = [np.random.randint(len(unlabeled_data))]

            for _ in range(n_samples - 1):
                # Find sample farthest from already selected samples
                selected_features = feature_vectors[selected_indices]
                remaining_indices = list(set(range(len(unlabeled_data))) - set(selected_indices))

                if not remaining_indices:
                    break

                max_min_distance = -1
                farthest_idx = None

                for idx in remaining_indices:
                    distances = np.linalg.norm(
                        feature_vectors[idx] - selected_features,
                        axis=1
                    )
                    min_distance = np.min(distances)

                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        farthest_idx = idx

                if farthest_idx is not None:
                    selected_indices.append(farthest_idx)

            selected_indices = np.array(selected_indices)

        else:
            raise ValueError(f"Unknown diversity method: {method}")

        selected_samples = unlabeled_data.iloc[selected_indices].copy()

        # Record selection
        self.selection_history.append({
            "timestamp": datetime.now().isoformat(),
            "strategy": f"diversity_{method}",
            "n_samples": len(selected_samples),
        })

        logger.info(f"Selected {len(selected_samples)} diverse samples")
        return selected_samples, selected_indices

    def combined_sampling(
        self,
        unlabeled_data: pd.DataFrame,
        prediction_probabilities: np.ndarray,
        predictions: np.ndarray,
        n_samples: int = 100,
        uncertainty_weight: float = 0.5,
        balance_weight: float = 0.5,
        minority_class_boost: float = 2.0,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Combine uncertainty and balance sampling.

        This implements the best practice of using both uncertainty sampling
        (to find informative examples) and balanced sampling (to prevent
        dataset imbalance).

        Args:
            unlabeled_data: Unlabeled data pool
            prediction_probabilities: Model prediction probabilities
            predictions: Model predictions
            n_samples: Number of samples to select
            uncertainty_weight: Weight for uncertainty (0-1)
            balance_weight: Weight for balance (0-1)
            minority_class_boost: Boost factor for minority classes

        Returns:
            Selected samples and their indices
        """
        logger.info("Performing combined uncertainty + balance sampling")

        # Calculate uncertainty scores
        confidences = np.max(prediction_probabilities, axis=1)
        uncertainty_scores = 1.0 - confidences

        # Calculate class frequencies
        unique_classes, class_counts = np.unique(predictions, return_counts=True)
        class_frequencies = {
            cls: count / len(predictions)
            for cls, count in zip(unique_classes, class_counts)
        }

        # Calculate minority boost scores
        minority_scores = np.array([
            1.0 / (class_frequencies[pred] ** minority_class_boost)
            for pred in predictions
        ])

        # Normalize scores to [0, 1]
        minority_scores = (minority_scores - minority_scores.min()) / (
            minority_scores.max() - minority_scores.min() + 1e-10
        )

        # Combine scores
        combined_scores = (
            uncertainty_weight * uncertainty_scores +
            balance_weight * minority_scores
        )

        # Select top samples
        selected_indices = np.argsort(combined_scores)[::-1][:n_samples]
        selected_samples = unlabeled_data.iloc[selected_indices].copy()

        # Add scores
        selected_samples["uncertainty_score"] = uncertainty_scores[selected_indices]
        selected_samples["minority_score"] = minority_scores[selected_indices]
        selected_samples["combined_score"] = combined_scores[selected_indices]
        selected_samples["predicted_class"] = predictions[selected_indices]

        # Record selection
        class_distribution = pd.Series(predictions[selected_indices]).value_counts().to_dict()
        self.selection_history.append({
            "timestamp": datetime.now().isoformat(),
            "strategy": "combined_uncertainty_balance",
            "n_samples": len(selected_samples),
            "uncertainty_weight": uncertainty_weight,
            "balance_weight": balance_weight,
            "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
            "mean_uncertainty": float(np.mean(uncertainty_scores[selected_indices])),
            "mean_minority_score": float(np.mean(minority_scores[selected_indices])),
        })

        logger.info(
            f"Selected {len(selected_samples)} samples with combined strategy. "
            f"Class distribution: {class_distribution}"
        )

        return selected_samples, selected_indices

    def get_selection_history(self) -> List[Dict[str, Any]]:
        """Get history of sample selections."""
        return self.selection_history


class ActiveLearningPipeline:
    """
    Complete Active Learning pipeline for HITL.

    Integrates:
    - Sample selection strategies
    - Labeling queue management
    - Model retraining triggers
    - Performance tracking
    """

    def __init__(self):
        """Initialize Active Learning pipeline."""
        self.strategy = ActiveLearningStrategy()
        self.labeling_queue: List[Dict[str, Any]] = []
        self.labeled_pool: List[Dict[str, Any]] = []
        self.iteration_history: List[Dict[str, Any]] = []

    def create_labeling_batch(
        self,
        unlabeled_data: pd.DataFrame,
        model_predictions: Optional[np.ndarray] = None,
        prediction_probabilities: Optional[np.ndarray] = None,
        n_samples: int = 100,
        strategy: str = "uncertainty",
        **strategy_kwargs,
    ) -> pd.DataFrame:
        """
        Create a batch of samples for labeling.

        Args:
            unlabeled_data: Unlabeled data pool
            model_predictions: Model predictions
            prediction_probabilities: Model prediction probabilities
            n_samples: Number of samples to select
            strategy: Selection strategy
            **strategy_kwargs: Additional strategy parameters

        Returns:
            Batch of samples for labeling
        """
        if strategy == "uncertainty":
            if prediction_probabilities is None:
                raise ValueError("prediction_probabilities required for uncertainty sampling")

            batch, indices = self.strategy.uncertainty_sampling(
                unlabeled_data=unlabeled_data,
                prediction_probabilities=prediction_probabilities,
                n_samples=n_samples,
                **strategy_kwargs,
            )

        elif strategy == "balanced":
            if model_predictions is None:
                raise ValueError("model_predictions required for balanced sampling")

            batch, indices = self.strategy.balanced_sampling(
                unlabeled_data=unlabeled_data,
                predictions=model_predictions,
                n_samples=n_samples,
                **strategy_kwargs,
            )

        elif strategy == "combined":
            if prediction_probabilities is None or model_predictions is None:
                raise ValueError("Both predictions and probabilities required for combined sampling")

            batch, indices = self.strategy.combined_sampling(
                unlabeled_data=unlabeled_data,
                prediction_probabilities=prediction_probabilities,
                predictions=model_predictions,
                n_samples=n_samples,
                **strategy_kwargs,
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Add to labeling queue
        batch["selection_strategy"] = strategy
        batch["selection_timestamp"] = datetime.now().isoformat()

        logger.info(f"Created labeling batch with {len(batch)} samples using {strategy} strategy")
        return batch

    def record_iteration(
        self,
        iteration: int,
        n_samples_labeled: int,
        model_performance: Optional[Dict[str, float]] = None,
    ):
        """Record an Active Learning iteration."""
        record = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "n_samples_labeled": n_samples_labeled,
            "total_labeled": sum(r["n_samples_labeled"] for r in self.iteration_history) + n_samples_labeled,
            "model_performance": model_performance or {},
        }

        self.iteration_history.append(record)
        logger.info(f"Recorded AL iteration {iteration}: {n_samples_labeled} samples labeled")


__all__ = ["ActiveLearningStrategy", "ActiveLearningPipeline"]
