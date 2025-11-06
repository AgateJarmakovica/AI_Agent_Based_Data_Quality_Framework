"""
Adaptive Learning Components for Healthcare Data Quality
Author: Agate Jarmakoviča

This module provides AI-driven learning capabilities that enable the framework
to automatically discover, understand, and adapt to different healthcare data models.

Components:
- SchemaLearner: Learns data schemas and patterns from input data
- HealthcareDataDetector: Detects healthcare data models (FHIR, HL7, OMOP, EHR)
- SemanticAnalyzer: Interprets medical terminology and data semantics
"""

from healthdq.learners.schema_learner import SchemaLearner

try:
    from healthdq.learners.healthcare_detector import HealthcareDataDetector
except ImportError:
    HealthcareDataDetector = None

__all__ = ["SchemaLearner", "HealthcareDataDetector"]
