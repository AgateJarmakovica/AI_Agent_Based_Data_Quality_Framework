"""
Complete Human-in-the-Loop Demo
Author: Agate Jarmakoviča

Demonstrē pilnu HITL workflow, izmantojot visas jaunās komponentes:
- Model-label disagreement detection
- Annotation quality metrics
- Annotator management
- Active learning integration
- Workflow automation

Based on: "Managing the Human in the Loop" from Active Learning for ML
"""

import numpy as np
import pandas as pd
from pathlib import Path

# HITL components
from healthdq.hitl import (
    HITLWorkflow,
    AnnotatorManager,
    DisagreementDetector,
    AnnotationQualityMetrics,
    ActiveLearningStrategy,
    ActiveLearningPipeline,
    ApprovalManager,
    FeedbackCollector,
)

# Utilities
from healthdq.utils.logger import get_logger

logger = get_logger(__name__)


def demo_1_annotator_qualification():
    """
    Demo 1: Assess Annotator Skills

    From the book: "It is highly recommended that annotators undergo thorough
    training sessions and complete qualification tests before they can work
    independently."
    """
    print("\n" + "=" * 60)
    print("DEMO 1: ANNOTATOR QUALIFICATION & SKILL ASSESSMENT")
    print("=" * 60)

    # Initialize annotator manager
    manager = AnnotatorManager(storage_path="data/annotators")

    # Register annotators
    annotator1_id = manager.register_annotator(
        name="Dr. Anna Bērziņa",
        email="anna.berzina@hospital.lv",
        expertise_level="senior",
        specializations=["cardiology", "internal_medicine"],
    )

    annotator2_id = manager.register_annotator(
        name="Jānis Kalniņš",
        email="janis.kalnins@hospital.lv",
        expertise_level="junior",
        specializations=["radiology"],
    )

    print(f"\n✅ Registered 2 annotators")
    print(f"   - Dr. Anna Bērziņa (ID: {annotator1_id[:8]}...)")
    print(f"   - Jānis Kalniņš (ID: {annotator2_id[:8]}...)")

    # Create gold standard test data
    gold_standard_labels = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1] * 5)  # 50 samples

    # Simulate annotator 1 (experienced - high accuracy)
    annotator1_labels = gold_standard_labels.copy()
    # Add 5% errors
    error_indices = np.random.choice(len(annotator1_labels), size=int(0.05 * len(annotator1_labels)), replace=False)
    annotator1_labels[error_indices] = 1 - annotator1_labels[error_indices]

    # Simulate annotator 2 (junior - lower accuracy)
    annotator2_labels = gold_standard_labels.copy()
    # Add 20% errors
    error_indices = np.random.choice(len(annotator2_labels), size=int(0.20 * len(annotator2_labels)), replace=False)
    annotator2_labels[error_indices] = 1 - annotator2_labels[error_indices]

    # Assess annotator 1
    print("\n📊 Assessing Dr. Anna Bērziņa...")
    assessment1 = manager.assess_annotator(
        annotator_id=annotator1_id,
        annotator_labels=annotator1_labels.tolist(),
        gold_standard_labels=gold_standard_labels.tolist(),
        min_accuracy=0.90,
        min_kappa=0.80,
    )

    print(f"   Accuracy: {assessment1['accuracy']:.2%}")
    print(f"   Cohen's Kappa: {assessment1['kappa']:.3f}")
    print(f"   Interpretation: {assessment1['interpretation']}")
    print(f"   Qualified: {'✅ YES' if assessment1['qualified'] else '❌ NO'}")
    print(f"   Recommendation: {assessment1['recommendation']}")

    # Assess annotator 2
    print("\n📊 Assessing Jānis Kalniņš...")
    assessment2 = manager.assess_annotator(
        annotator_id=annotator2_id,
        annotator_labels=annotator2_labels.tolist(),
        gold_standard_labels=gold_standard_labels.tolist(),
        min_accuracy=0.90,
        min_kappa=0.80,
    )

    print(f"   Accuracy: {assessment2['accuracy']:.2%}")
    print(f"   Cohen's Kappa: {assessment2['kappa']:.3f}")
    print(f"   Interpretation: {assessment2['interpretation']}")
    print(f"   Qualified: {'✅ YES' if assessment2['qualified'] else '❌ NO'}")
    print(f"   Recommendation: {assessment2['recommendation']}")

    return manager


def demo_2_inter_annotator_agreement():
    """
    Demo 2: Inter-Annotator Agreement

    From the book: "If your budget allows, you can assign each data point to
    multiple annotators to identify conflicts."
    """
    print("\n" + "=" * 60)
    print("DEMO 2: INTER-ANNOTATOR AGREEMENT & MAJORITY VOTE")
    print("=" * 60)

    # Simulate 3 annotators labeling the same 20 samples
    annotator1_labels = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    annotator2_labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    annotator3_labels = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])

    # Calculate pairwise agreement
    print("\n📊 Calculating pairwise Cohen's Kappa...")
    kappa_12 = AnnotationQualityMetrics.cohen_kappa(annotator1_labels, annotator2_labels)
    kappa_13 = AnnotationQualityMetrics.cohen_kappa(annotator1_labels, annotator3_labels)
    kappa_23 = AnnotationQualityMetrics.cohen_kappa(annotator2_labels, annotator3_labels)

    print(f"   Annotator 1 vs 2: κ = {kappa_12['kappa']:.3f} ({kappa_12['interpretation']})")
    print(f"   Annotator 1 vs 3: κ = {kappa_13['kappa']:.3f} ({kappa_13['interpretation']})")
    print(f"   Annotator 2 vs 3: κ = {kappa_23['kappa']:.3f} ({kappa_23['interpretation']})")

    # Calculate inter-annotator agreement matrix
    print("\n📊 Inter-Annotator Agreement Matrix:")
    agreement_matrix = AnnotationQualityMetrics.inter_annotator_agreement_matrix({
        "Annotator 1": annotator1_labels.tolist(),
        "Annotator 2": annotator2_labels.tolist(),
        "Annotator 3": annotator3_labels.tolist(),
    })
    print(agreement_matrix)

    # Majority vote
    print("\n🗳️  Calculating majority vote...")
    majority_results = AnnotationQualityMetrics.majority_vote(
        annotations=[
            annotator1_labels.tolist(),
            annotator2_labels.tolist(),
            annotator3_labels.tolist(),
        ],
        return_confidence=True,
    )

    print(f"   Sample majority votes (first 5):")
    for i, result in enumerate(majority_results[:5]):
        print(f"   Sample {i+1}: Label={result['label']}, Confidence={result['confidence']:.2f}, Votes={result['votes']}")


def demo_3_model_label_disagreement():
    """
    Demo 3: Model-Label Disagreement Detection

    From the book: "Programmatically identifying mismatches - to identify
    discrepancies between the model's predictions and the human-annotated labels."
    """
    print("\n" + "=" * 60)
    print("DEMO 3: MODEL-LABEL DISAGREEMENT DETECTION")
    print("=" * 60)

    # Create sample data
    n_samples = 100
    data = pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n_samples)],
        "diagnosis": np.random.choice(["healthy", "sick"], n_samples),
        "confidence": np.random.uniform(0.5, 1.0, n_samples),
    })

    # Simulate human labels (ground truth)
    y_true = np.random.choice([0, 1], n_samples)

    # Simulate model predictions (with some errors)
    y_pred = y_true.copy()
    # Introduce 20% disagreement
    error_indices = np.random.choice(n_samples, size=int(0.20 * n_samples), replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]

    # Add confidence scores
    y_pred_proba = np.random.uniform(0.6, 0.95, n_samples)

    # Detect disagreements
    print("\n🔍 Detecting model-label disagreements...")
    detector = DisagreementDetector()
    analysis = detector.detect_mismatches_with_confidence(
        data=data,
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        confidence_threshold=0.8,
    )

    print(f"   Total samples: {analysis['total_samples']}")
    print(f"   Total mismatches: {analysis['total_mismatches']}")
    print(f"   Disagreement rate: {analysis['mismatch_rate']:.2%}")
    print(f"   High-confidence mismatches: {analysis.get('high_confidence_mismatches', 0)}")

    # Sample mismatches for review
    print("\n📋 Sampling mismatches for manual review...")
    review_samples = detector.sample_mismatches_for_review(
        n_samples=5,
        strategy="high_confidence",
        priority="high_confidence",
    )

    print(f"   Sampled {len(review_samples)} high-confidence mismatches:")
    for i, mismatch in enumerate(review_samples[:3]):
        print(f"   {i+1}. Sample {mismatch['sample_id']}: "
              f"True={mismatch['true_label']}, Pred={mismatch['predicted_label']}, "
              f"Confidence={mismatch.get('model_confidence', 'N/A')}")

    # Create re-labeling queue
    print("\n📝 Creating re-labeling queue for confusing cases...")
    relabel_queue = detector.create_relabeling_queue(
        priority="high_confidence",
        max_items=10,
    )
    print(f"   Created queue with {len(relabel_queue)} items for re-labeling")

    return detector


def demo_4_active_learning():
    """
    Demo 4: Active Learning with Uncertainty Sampling

    From the book: "Active ML promises more efficient ML by intelligently
    selecting the most informative samples for labeling by human oracles."
    """
    print("\n" + "=" * 60)
    print("DEMO 4: ACTIVE LEARNING - UNCERTAINTY & BALANCED SAMPLING")
    print("=" * 60)

    # Create unlabeled data pool
    n_unlabeled = 1000
    unlabeled_data = pd.DataFrame({
        "feature_1": np.random.randn(n_unlabeled),
        "feature_2": np.random.randn(n_unlabeled),
        "feature_3": np.random.randn(n_unlabeled),
    })

    # Simulate model predictions (3 classes)
    predictions = np.random.choice([0, 1, 2], n_unlabeled, p=[0.6, 0.3, 0.1])  # Imbalanced

    # Simulate prediction probabilities
    prediction_probs = np.random.dirichlet(alpha=[1, 1, 1], size=n_unlabeled)

    # Initialize Active Learning
    al_strategy = ActiveLearningStrategy()

    # 1. Uncertainty Sampling
    print("\n🎯 1. UNCERTAINTY SAMPLING (Least Confident)")
    selected_uncertain, indices = al_strategy.uncertainty_sampling(
        unlabeled_data=unlabeled_data,
        prediction_probabilities=prediction_probs,
        n_samples=50,
        strategy="least_confident",
    )
    print(f"   Selected {len(selected_uncertain)} most uncertain samples")
    print(f"   Mean uncertainty: {selected_uncertain['uncertainty_score'].mean():.3f}")

    # 2. Balanced Sampling (prevent imbalance)
    print("\n⚖️  2. BALANCED SAMPLING (Prevent Dataset Imbalance)")
    selected_balanced, indices = al_strategy.balanced_sampling(
        unlabeled_data=unlabeled_data,
        predictions=predictions,
        n_samples=90,  # 30 per class
        class_ratios={0: 0.33, 1: 0.33, 2: 0.34},
    )
    print(f"   Selected {len(selected_balanced)} balanced samples")
    class_dist = selected_balanced['predicted_class'].value_counts().to_dict()
    print(f"   Class distribution: {class_dist}")

    # 3. Combined Sampling (uncertainty + balance)
    print("\n🎯⚖️  3. COMBINED SAMPLING (Uncertainty + Balance)")
    selected_combined, indices = al_strategy.combined_sampling(
        unlabeled_data=unlabeled_data,
        prediction_probabilities=prediction_probs,
        predictions=predictions,
        n_samples=100,
        uncertainty_weight=0.6,
        balance_weight=0.4,
        minority_class_boost=2.0,
    )
    print(f"   Selected {len(selected_combined)} samples with combined strategy")
    print(f"   Mean uncertainty: {selected_combined['uncertainty_score'].mean():.3f}")
    class_dist = selected_combined['predicted_class'].value_counts().to_dict()
    print(f"   Class distribution: {class_dist}")


def demo_5_complete_workflow():
    """
    Demo 5: Complete HITL Workflow

    From the book: "The workflow is the end-to-end sequence of steps that's
    followed by the annotator to complete the labeling task."
    """
    print("\n" + "=" * 60)
    print("DEMO 5: COMPLETE HITL WORKFLOW")
    print("=" * 60)

    # Initialize workflow
    workflow = HITLWorkflow()

    # Create workflow session
    print("\n1️⃣  Creating workflow session...")
    session_id = workflow.create_workflow_session(
        session_name="Medical Image Annotation - Batch 1",
        description="Annotate chest X-rays for pneumonia detection",
    )
    print(f"   ✅ Session created: {session_id[:8]}...")

    # Create sample data
    n_samples = 20
    samples = pd.DataFrame({
        "image_id": [f"IMG_{i:04d}" for i in range(n_samples)],
        "image_path": [f"/data/images/img_{i:04d}.jpg" for i in range(n_samples)],
        "uncertainty_score": np.random.uniform(0.3, 0.9, n_samples),
    })

    # Create annotation tasks
    print("\n2️⃣  Creating annotation tasks...")
    task_ids = workflow.create_annotation_tasks(
        session_id=session_id,
        samples=samples,
        task_type="classification",
        priority="high",
        requires_multiple_annotators=True,
        n_annotators_per_sample=2,
    )
    print(f"   ✅ Created {len(task_ids)} tasks")

    # Register annotators (using the manager from demo 1)
    print("\n3️⃣  Registering annotators...")
    ann1_id = workflow.annotator_manager.register_annotator(
        name="Dr. Smith", expertise_level="senior"
    )
    ann2_id = workflow.annotator_manager.register_annotator(
        name="Dr. Jones", expertise_level="senior"
    )
    print(f"   ✅ Registered 2 annotators")

    # Mark them as qualified (skip qualification for demo)
    workflow.annotator_manager.annotators[ann1_id].is_qualified = True
    workflow.annotator_manager.annotators[ann2_id].is_qualified = True

    # Assign tasks
    print("\n4️⃣  Assigning tasks to annotators...")
    assignment = workflow.assign_tasks_to_annotators(
        task_ids=task_ids,
        strategy="balanced",
    )
    print(f"   ✅ Tasks assigned:")
    for annotator_id, assigned_tasks in assignment.items():
        annotator_name = workflow.annotator_manager.annotators[annotator_id].name
        print(f"      - {annotator_name}: {len(assigned_tasks)} tasks")

    # Simulate annotations
    print("\n5️⃣  Simulating annotations...")
    n_annotated = 0
    for task_id in task_ids[:10]:  # Annotate first 10 tasks
        task = workflow.tasks[task_id]
        for annotator_id in task["assigned_annotators"]:
            # Simulate annotation (random label)
            annotation = np.random.choice(["normal", "pneumonia"])
            workflow.submit_annotation(
                task_id=task_id,
                annotator_id=annotator_id,
                annotation=annotation,
                time_spent_seconds=np.random.uniform(30, 120),
            )
            n_annotated += 1
    print(f"   ✅ Submitted {n_annotated} annotations")

    # Review annotations
    print("\n6️⃣  Reviewing annotations...")
    completed_tasks = [tid for tid in task_ids if workflow.tasks[tid]["status"] == "annotated"]
    review_summary = workflow.review_annotations(
        task_ids=completed_tasks,
        reviewer_id="reviewer_001",
        auto_approve_threshold=0.95,
    )
    print(f"   ✅ Review summary:")
    print(f"      - Reviewed: {review_summary['reviewed']}")
    print(f"      - Auto-approved: {review_summary['auto_approved']}")
    print(f"      - Needs manual review: {review_summary['needs_manual_review']}")

    # Get workflow status
    print("\n7️⃣  Workflow status...")
    status = workflow.get_workflow_status(session_id)
    print(f"   Session: {status['session_name']}")
    print(f"   Status: {status['status']}")
    print(f"   Current stage: {status['current_stage']}")
    print(f"   Total tasks: {status['total_tasks']}")
    print(f"   Task statuses: {status['task_statuses']}")

    print("\n✅ WORKFLOW COMPLETED SUCCESSFULLY!")


def main():
    """Run all HITL demos."""
    print("\n" + "=" * 60)
    print("🚀 HUMAN-IN-THE-LOOP COMPLETE DEMO")
    print("Based on: 'Managing the Human in the Loop' (Active ML)")
    print("=" * 60)

    try:
        # Demo 1: Annotator qualification
        manager = demo_1_annotator_qualification()

        # Demo 2: Inter-annotator agreement
        demo_2_inter_annotator_agreement()

        # Demo 3: Model-label disagreement
        detector = demo_3_model_label_disagreement()

        # Demo 4: Active learning
        demo_4_active_learning()

        # Demo 5: Complete workflow
        demo_5_complete_workflow()

        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("1. ✅ Annotator qualification with accuracy & Cohen's Kappa")
        print("2. ✅ Inter-annotator agreement with majority vote")
        print("3. ✅ Model-label disagreement detection & re-labeling")
        print("4. ✅ Active learning with uncertainty & balanced sampling")
        print("5. ✅ Complete automated HITL workflow")
        print("\n📚 Implementation based on best practices from:")
        print("   'Managing the Human in the Loop' - Active Learning for ML")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
