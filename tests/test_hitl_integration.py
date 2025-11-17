"""
HITL Integration Tests
Author: Agate Jarmakoviča

Tests for the complete HITL system integration.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that all HITL components can be imported."""
    print("Testing HITL component imports...")

    try:
        from healthdq.hitl import (
            DisagreementDetector,
            AnnotationQualityMetrics,
            AnnotatorManager,
            AnnotatorProfile,
            ActiveLearningStrategy,
            ActiveLearningPipeline,
            HITLWorkflow,
            WorkflowStage,
            TaskStatus,
            ApprovalManager,
            FeedbackCollector,
        )
        print("✅ All HITL components imported successfully")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_disagreement_detector():
    """Test DisagreementDetector initialization."""
    print("\nTesting DisagreementDetector...")

    try:
        from healthdq.hitl import DisagreementDetector

        detector = DisagreementDetector()
        assert detector.mismatches == []
        assert detector.disagreement_history == []

        print("✅ DisagreementDetector initialized successfully")
        return True

    except Exception as e:
        print(f"❌ DisagreementDetector test failed: {e}")
        return False


def test_annotator_manager():
    """Test AnnotatorManager initialization."""
    print("\nTesting AnnotatorManager...")

    try:
        from healthdq.hitl import AnnotatorManager

        manager = AnnotatorManager()
        assert manager.annotators == {}

        print("✅ AnnotatorManager initialized successfully")
        return True

    except Exception as e:
        print(f"❌ AnnotatorManager test failed: {e}")
        return False


def test_active_learning():
    """Test ActiveLearningStrategy initialization."""
    print("\nTesting ActiveLearningStrategy...")

    try:
        from healthdq.hitl import ActiveLearningStrategy

        strategy = ActiveLearningStrategy()
        assert strategy.selection_history == []

        print("✅ ActiveLearningStrategy initialized successfully")
        return True

    except Exception as e:
        print(f"❌ ActiveLearningStrategy test failed: {e}")
        return False


def test_workflow():
    """Test HITLWorkflow initialization."""
    print("\nTesting HITLWorkflow...")

    try:
        from healthdq.hitl import HITLWorkflow

        workflow = HITLWorkflow()
        assert workflow.tasks == {}
        assert workflow.workflow_sessions == {}

        print("✅ HITLWorkflow initialized successfully")
        return True

    except Exception as e:
        print(f"❌ HITLWorkflow test failed: {e}")
        return False


def test_annotation_quality_metrics():
    """Test AnnotationQualityMetrics static methods."""
    print("\nTesting AnnotationQualityMetrics...")

    try:
        from healthdq.hitl import AnnotationQualityMetrics

        # Test that class can be instantiated
        metrics = AnnotationQualityMetrics()

        print("✅ AnnotationQualityMetrics initialized successfully")
        return True

    except Exception as e:
        print(f"❌ AnnotationQualityMetrics test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("HITL INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        test_imports,
        test_disagreement_detector,
        test_annotator_manager,
        test_active_learning,
        test_workflow,
        test_annotation_quality_metrics,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
