# Human-in-the-Loop (HITL) System

**Autors:** Agate Jarmakoviča
**Pamatojoties uz:** "Managing the Human in the Loop" (Active Learning for Machine Learning)

---

## 📋 Saturs

1. [Ievads](#ievads)
2. [Arhitektūra](#arhitektūra)
3. [Komponentes](#komponentes)
4. [Izmantošana](#izmantošana)
5. [Labākās prakses](#labākās-prakses)
6. [API Reference](#api-reference)

---

## Ievads

healthdq-ai HITL sistēma implementē pilnu cilvēka iesaistes ciklu datu kvalitātes novērtēšanā un uzlabošanā. Sistēma ir izstrādāta, pamatojoties uz Active Machine Learning labākajām praksēm, kas aprakstītas grāmatā **"Managing the Human in the Loop"**.

### Galvenās iespējas

- ✅ **Anotētāju kvalifikācija** - Annotator skill assessment ar accuracy un Cohen's Kappa
- ✅ **Inter-annotator agreement** - Fleiss' Kappa, majority vote, agreement matrices
- ✅ **Model-label disagreement** - Programmatic mismatch detection un re-labeling
- ✅ **Active Learning** - Uncertainty sampling, balanced sampling, diversity sampling
- ✅ **Workflow automation** - End-to-end automated labeling pipeline
- ✅ **Feedback learning** - Continuous improvement from human feedback
- ✅ **Quality metrics** - Publication-accurate annotation quality assessment

---

## Arhitektūra

```
┌─────────────────────────────────────────────────────────────┐
│                    HITL Workflow Manager                     │
│           (Orchestrates entire HITL pipeline)                │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┴────────┬──────────┬──────────┬──────────┐
    │                 │          │          │          │
┌───▼────┐  ┌────────▼───┐  ┌──▼──────┐ ┌─▼────────┐ ┌▼───────────┐
│ Active │  │ Annotator  │  │Disagree-│ │ Quality  │ │  Approval  │
│Learning│  │  Manager   │  │  ment   │ │ Metrics  │ │  Manager   │
│Strategy│  │            │  │Detector │ │          │ │            │
└────────┘  └────────────┘  └─────────┘ └──────────┘ └────────────┘
     │             │              │           │              │
     └─────────────┴──────────────┴───────────┴──────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Feedback System  │
                    │  (Learning Loop)  │
                    └───────────────────┘
```

### Workflow Stages

1. **Sample Selection** - Active learning izvēlas informatīvākos datus
2. **Task Assignment** - Automātiska uzdevumu sadalīšana anotētājiem
3. **Annotation** - Cilvēki veic anotācijas
4. **Quality Check** - Kvalitātes pārbaude un validācija
5. **Review** - Recenzēšana un konflikt resolution
6. **Approval** - Galīgā apstiprināšana
7. **Feedback** - Mācīšanās no rezultātiem

---

## Komponentes

### 1. DisagreementDetector

Identificē un analizē neatbilstības starp modeļa prognozēm un cilvēka anotācijām.

**Galvenās metodes:**
- `detect_mismatches()` - Atrod visas neatbilstības
- `detect_mismatches_with_confidence()` - Iekļauj modeļa confidence scores
- `sample_mismatches_for_review()` - Izvēlas samplus manual review
- `create_relabeling_queue()` - Izveido re-labeling queue

**Piemērs:**
```python
from healthdq.hitl import DisagreementDetector

detector = DisagreementDetector()

# Detect disagreements
analysis = detector.detect_mismatches_with_confidence(
    data=patient_data,
    y_true=human_labels,
    y_pred=model_predictions,
    y_pred_proba=prediction_probabilities,
    confidence_threshold=0.8
)

print(f"Disagreement rate: {analysis['mismatch_rate']:.2%}")
print(f"High-confidence mismatches: {analysis['high_confidence_mismatches']}")

# Sample for review
review_samples = detector.sample_mismatches_for_review(
    n_samples=10,
    strategy="high_confidence"
)
```

### 2. AnnotationQualityMetrics

Novērtē anotāciju kvalitāti, izmantojot dažādus metrikus.

**Galvenās metodes:**
- `calculate_accuracy()` - Accuracy pret gold standard
- `cohen_kappa()` - Cohen's Kappa divu anotētāju agreement
- `fleiss_kappa()` - Fleiss' Kappa vairāku anotētāju agreement
- `majority_vote()` - Majority vote ar confidence scores
- `assess_annotator_quality()` - Pilna kvalitātes novērtēšana

**Piemērs:**
```python
from healthdq.hitl import AnnotationQualityMetrics

# Assess annotator against gold standard
assessment = AnnotationQualityMetrics.assess_annotator_quality(
    annotator_labels=annotator_labels,
    gold_standard_labels=gold_labels,
    min_accuracy=0.90,
    min_kappa=0.80
)

print(f"Qualified: {assessment['qualified']}")
print(f"Accuracy: {assessment['accuracy']:.2%}")
print(f"Kappa: {assessment['kappa']:.3f}")
print(f"Recommendation: {assessment['recommendation']}")

# Calculate inter-annotator agreement
kappa = AnnotationQualityMetrics.cohen_kappa(
    annotator1_labels=labels1,
    annotator2_labels=labels2
)
print(f"Cohen's Kappa: {kappa['kappa']:.3f}")
print(f"Interpretation: {kappa['interpretation']}")
```

**Cohen's Kappa Interpretation:**
- < 0.00: No agreement (worse than chance)
- 0.00-0.20: Slight agreement
- 0.21-0.40: Fair agreement
- 0.41-0.60: Moderate agreement
- 0.61-0.80: Substantial agreement
- 0.81-1.00: Almost perfect agreement

### 3. AnnotatorManager

Pārvalda anotētājus un viņu performanci.

**Galvenās metodes:**
- `register_annotator()` - Reģistrē jaunu anotētāju
- `assess_annotator()` - Novērtē anotētāja kvalitāti
- `assign_task()` - Piešķir uzdevumu
- `distribute_tasks()` - Sadala uzdevumus starp anotētājiem
- `get_annotator_statistics()` - Iegūst detalizētu statistiku

**Piemērs:**
```python
from healthdq.hitl import AnnotatorManager

manager = AnnotatorManager(storage_path="data/annotators")

# Register annotator
annotator_id = manager.register_annotator(
    name="Dr. Anna Bērziņa",
    email="anna@hospital.lv",
    expertise_level="senior",
    specializations=["cardiology"]
)

# Assess annotator
assessment = manager.assess_annotator(
    annotator_id=annotator_id,
    annotator_labels=test_labels,
    gold_standard_labels=gold_labels,
    min_accuracy=0.90,
    min_kappa=0.80
)

# Distribute tasks
assignment = manager.distribute_tasks(
    task_ids=task_ids,
    strategy="balanced",  # balanced, expertise, speed
    max_tasks_per_annotator=10
)

# Get statistics
stats = manager.get_annotator_statistics(annotator_id)
print(f"Total annotations: {stats['total_annotations']}")
print(f"Average accuracy: {stats['average_accuracy']:.2%}")
```

### 4. ActiveLearningStrategy

Implementē Active Learning metodes informatīvāko samplu izvēlei.

**Galvenās metodes:**
- `uncertainty_sampling()` - Izvēlas visnedrošākos samplus
- `balanced_sampling()` - Nodrošina klašu balansu
- `diversity_sampling()` - Izvēlas dažādus samplus
- `combined_sampling()` - Kombinē vairākas stratēģijas

**Piemērs:**
```python
from healthdq.hitl import ActiveLearningStrategy

al_strategy = ActiveLearningStrategy()

# Uncertainty sampling
selected, indices = al_strategy.uncertainty_sampling(
    unlabeled_data=unlabeled_pool,
    prediction_probabilities=pred_proba,
    n_samples=100,
    strategy="least_confident"  # least_confident, margin, entropy
)

# Balanced sampling (prevent imbalance)
selected, indices = al_strategy.balanced_sampling(
    unlabeled_data=unlabeled_pool,
    predictions=model_predictions,
    n_samples=100,
    class_ratios={0: 0.3, 1: 0.3, 2: 0.4}
)

# Combined sampling
selected, indices = al_strategy.combined_sampling(
    unlabeled_data=unlabeled_pool,
    prediction_probabilities=pred_proba,
    predictions=model_predictions,
    n_samples=100,
    uncertainty_weight=0.6,
    balance_weight=0.4,
    minority_class_boost=2.0
)
```

**Uncertainty Strategies:**
1. **Least Confident** - Izvēlas samplus, kur modelim ir zemākā confidence
2. **Margin** - Izvēlas samplus ar mazu starpību starp top 2 predictions
3. **Entropy** - Izvēlas samplus ar augstāko prediction entropy

### 5. HITLWorkflow

Pilns workflow manager visam HITL procesam.

**Galvenās metodes:**
- `create_workflow_session()` - Izveido jaunu sesiju
- `create_annotation_tasks()` - Izveido anotācijas uzdevumus
- `assign_tasks_to_annotators()` - Piešķir uzdevumus
- `submit_annotation()` - Iesniedz anotāciju
- `review_annotations()` - Pārskata anotācijas
- `detect_and_resolve_disagreements()` - Atrisina konfliktus

**Piemērs:**
```python
from healthdq.hitl import HITLWorkflow

workflow = HITLWorkflow()

# Create session
session_id = workflow.create_workflow_session(
    session_name="Medical Image Annotation - Batch 1",
    description="Chest X-ray pneumonia detection"
)

# Create tasks
task_ids = workflow.create_annotation_tasks(
    session_id=session_id,
    samples=selected_samples,
    task_type="classification",
    priority="high",
    requires_multiple_annotators=True,
    n_annotators_per_sample=2
)

# Assign tasks
assignment = workflow.assign_tasks_to_annotators(
    task_ids=task_ids,
    strategy="balanced"
)

# Submit annotation
workflow.submit_annotation(
    task_id=task_id,
    annotator_id=annotator_id,
    annotation={"label": "pneumonia", "confidence": 0.95},
    time_spent_seconds=45
)

# Review
review_summary = workflow.review_annotations(
    task_ids=completed_tasks,
    reviewer_id="reviewer_001",
    auto_approve_threshold=0.95
)

# Get status
status = workflow.get_workflow_status(session_id)
```

---

## Izmantošana

### Pilns HITL Pipeline

```python
import pandas as pd
from healthdq.hitl import (
    HITLWorkflow,
    ActiveLearningStrategy,
    AnnotatorManager
)

# 1. Initialize components
workflow = HITLWorkflow()
al_strategy = ActiveLearningStrategy()

# 2. Select samples using Active Learning
unlabeled_data = pd.read_csv("unlabeled_pool.csv")
model_predictions = model.predict(unlabeled_data)
model_proba = model.predict_proba(unlabeled_data)

selected_samples, indices = al_strategy.combined_sampling(
    unlabeled_data=unlabeled_data,
    prediction_probabilities=model_proba,
    predictions=model_predictions,
    n_samples=100,
    uncertainty_weight=0.6,
    balance_weight=0.4
)

# 3. Create workflow session
session_id = workflow.create_workflow_session(
    session_name="Active Learning Iteration 1",
    description="High uncertainty + balanced sampling"
)

# 4. Create annotation tasks
task_ids = workflow.create_annotation_tasks(
    session_id=session_id,
    samples=selected_samples,
    task_type="classification",
    priority="high"
)

# 5. Assign to annotators
assignment = workflow.assign_tasks_to_annotators(
    task_ids=task_ids,
    strategy="balanced"
)

# 6. Wait for annotations...
# (Annotators complete their work)

# 7. Review and approve
review_summary = workflow.review_annotations(
    task_ids=task_ids,
    reviewer_id="senior_reviewer",
    auto_approve_threshold=0.95
)

# 8. Export results
workflow.export_annotations(
    session_id=session_id,
    output_path="annotations_batch1.csv",
    include_metadata=True
)

# 9. Detect disagreements for model improvement
disagreements = workflow.detect_and_resolve_disagreements(
    session_id=session_id,
    y_true=human_labels,
    y_pred=model_predictions,
    create_relabeling_tasks=True
)

print(f"Workflow completed!")
print(f"Annotations: {review_summary['reviewed']}")
print(f"Disagreements: {disagreements['total_mismatches']}")
```

---

## Labākās prakses

### 1. Anotētāju Kvalifikācija

**No grāmatas:** "It is highly recommended that annotators undergo thorough training sessions and complete qualification tests before they can work independently."

```python
# Qualification test
assessment = manager.assess_annotator(
    annotator_id=new_annotator,
    annotator_labels=test_labels,
    gold_standard_labels=gold_labels,
    min_accuracy=0.90,  # 90% accuracy
    min_kappa=0.80      # Substantial agreement
)

if not assessment['qualified']:
    print(f"Additional training needed: {assessment['recommendation']}")
```

**Ieteikumi:**
- Minimum accuracy: 90%
- Minimum Cohen's Kappa: 0.80 (substantial agreement)
- Periodic re-assessment (every 3 months)
- Use control samples for ongoing monitoring

### 2. Multi-Annotator Labeling

**No grāmatas:** "If your budget allows, you can assign each data point to multiple annotators to identify conflicts."

```python
# Require 2-3 annotators for critical data
task_ids = workflow.create_annotation_tasks(
    session_id=session_id,
    samples=critical_samples,
    requires_multiple_annotators=True,
    n_annotators_per_sample=3  # 3 annotators
)

# Resolve with majority vote
majority = AnnotationQualityMetrics.majority_vote(
    annotations=[ann1, ann2, ann3],
    return_confidence=True
)
```

**Kad izmantot:**
- Critical medical decisions
- Ambiguous cases
- Model training data quality assurance
- Inter-annotator agreement monitoring

### 3. Balanced Dataset Creation

**No grāmatas:** "To prevent imbalanced datasets, we can actively sample minority classes at higher rates during data collection."

```python
# Monitor class distribution
class_counts = current_dataset['label'].value_counts()

# Over-sample minority classes
selected, _ = al_strategy.balanced_sampling(
    unlabeled_data=unlabeled_pool,
    predictions=model_predictions,
    n_samples=100,
    class_ratios={
        'rare_disease': 0.4,      # 40% minority class
        'common_disease': 0.3,    # 30%
        'healthy': 0.3            # 30%
    }
)
```

### 4. Disagreement-Driven Re-labeling

**No grāmatas:** "By systematically identifying, understanding, and resolving model-label disagreements, the system improves over time."

```python
# Find high-confidence mismatches
detector = DisagreementDetector()
analysis = detector.detect_mismatches_with_confidence(
    data=data,
    y_true=human_labels,
    y_pred=model_predictions,
    y_pred_proba=model_confidence,
    confidence_threshold=0.8
)

# Create re-labeling queue
relabel_queue = detector.create_relabeling_queue(
    priority="high_confidence",
    max_items=50
)

# Assign for re-review (with 2+ annotators)
workflow.create_annotation_tasks(
    session_id=session_id,
    samples=relabel_queue,
    task_type="relabeling",
    requires_multiple_annotators=True,
    n_annotators_per_sample=2
)
```

### 5. Continuous Quality Monitoring

```python
# Track annotator performance over time
stats = manager.get_annotator_statistics(annotator_id)

if stats['average_accuracy'] < 0.85:
    # Trigger re-training
    print(f"Annotator needs re-training")

# Monitor agreement trends
recent_kappas = stats['recent_kappa_scores']
if len(recent_kappas) >= 3:
    trend = np.polyfit(range(len(recent_kappas)), recent_kappas, 1)[0]
    if trend < 0:
        print("Warning: Agreement declining")
```

---

## Konfigurācija

HITL sistēma izmanto `configs/hitl.yml` konfigurāciju:

```yaml
hitl:
  annotation_quality:
    min_accuracy: 0.90
    min_kappa: 0.80
    require_multiple_annotators: true

  active_learning:
    default_strategy: "combined"
    uncertainty_weight: 0.6
    balance_weight: 0.4

  workflow:
    auto_approve_threshold: 0.95
    max_active_tasks: 5
```

---

## API Reference

Pilna API dokumentācija pieejama:

- [DisagreementDetector API](../src/healthdq/hitl/disagreement.py)
- [AnnotationQualityMetrics API](../src/healthdq/hitl/quality_metrics.py)
- [AnnotatorManager API](../src/healthdq/hitl/annotator_manager.py)
- [ActiveLearningStrategy API](../src/healthdq/hitl/active_learning.py)
- [HITLWorkflow API](../src/healthdq/hitl/workflow.py)

---

## Piemēri

1. **Complete Demo** - `examples/hitl_complete_demo.py`
   - Annotator qualification
   - Inter-annotator agreement
   - Model-label disagreement
   - Active learning
   - Full workflow

2. **Adaptive Learning Demo** - `examples/adaptive_learning_demo.py`
   - Active learning integration
   - Healthcare data detection

---

## References

**Grāmata:** "Human-in-the-Loop Machine Learning: Active learning and annotation for human-centered AI"
**Autors:** Robert Munro
**Chapter 3:** Managing the Human in the Loop

**Key Concepts Implemented:**
- ✅ Interactive learning systems and workflows
- ✅ Labeling interface design
- ✅ Model-label disagreement handling
- ✅ Annotation quality control (Cohen's Kappa, accuracy)
- ✅ Multiple annotator workflows with majority vote
- ✅ Balanced sampling for dataset quality
- ✅ Workflow automation and optimization

---

## Autors

**Agate Jarmakoviča**
PhD Research - AI Agent-Based Data Quality Framework
Email: [your-email]
GitHub: [@AgateJarmakovica](https://github.com/AgateJarmakovica)

---

*Šī dokumentācija ir daļa no healthdq-ai v2.0 projekta.*
