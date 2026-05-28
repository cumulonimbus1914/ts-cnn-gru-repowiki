# Performance Metrics & Evaluation

<cite>
**Referenced Files in This Document**
- [utils_metrics_final.py](file://utils_metrics_final.py)
- [evaluate_ts_final.py](file://evaluate_ts_final.py)
- [config_ts_final.py](file://config_ts_final.py)
- [validation-cv-bootstrap.md](file://reports/validation-cv-bootstrap.md)
- [evaluation_report.md](file://reports/evaluation_report.md)
- [analyze_predictions.py](file://extras/analyze_predictions.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)
10. [Appendices](#appendices)

## Introduction
This document explains the comprehensive performance metrics system used to evaluate the thunderstorm nowcasting model. It covers:
- Distinctions between frame-level and event-level metrics (POD, FAR, CSI, ETS, SEDI, F1/F2)
- Weighted event metrics for severity-aware evaluation
- Bootstrap confidence intervals for statistical significance testing
- Lead time analysis (early detection rates and temporal accuracy)
- Confusion matrix, ROC-AUC, and PR-AUC computation from raw probabilities
- Persistence filter effectiveness measurement
- Guidance on metric interpretation, benchmarking, and significance testing

## Project Structure
The evaluation pipeline integrates model inference, post-processing, and comprehensive metrics computation:
- Inference and post-processing: [evaluate_ts_final.py](file://evaluate_ts_final.py)
- Metrics computation and bootstrapping: [utils_metrics_final.py](file://utils_metrics_final.py)
- Configuration controlling thresholds, lead time, and severity weights: [config_ts_final.py](file://config_ts_final.py)
- Validation and bootstrap report: [validation-cv-bootstrap.md](file://reports/validation-cv-bootstrap.md)
- Model comparison report: [evaluation_report.md](file://reports/evaluation_report.md)
- Quick prediction analysis: [analyze_predictions.py](file://extras/analyze_predictions.py)

```mermaid
graph TB
A["evaluate_ts_final.py<br/>Inference + Post-processing + Metrics"] --> B["utils_metrics_final.py<br/>Metrics + Bootstrapping"]
A --> C["config_ts_final.py<br/>Threshold + Lead + Weights"]
B --> D["validation-cv-bootstrap.md<br/>Bootstrap CI Plan"]
A --> E["evaluation_report.md<br/>Model Comparison"]
A --> F["analyze_predictions.py<br/>Quick Audit"]
```

**Diagram sources**
- [evaluate_ts_final.py:1-908](file://evaluate_ts_final.py#L1-L908)
- [utils_metrics_final.py:1-760](file://utils_metrics_final.py#L1-L760)
- [config_ts_final.py:1-208](file://config_ts_final.py#L1-L208)
- [validation-cv-bootstrap.md:1-89](file://reports/validation-cv-bootstrap.md#L1-L89)
- [evaluation_report.md:1-58](file://reports/evaluation_report.md#L1-L58)
- [analyze_predictions.py:1-64](file://extras/analyze_predictions.py#L1-L64)

**Section sources**
- [evaluate_ts_final.py:1-908](file://evaluate_ts_final.py#L1-L908)
- [utils_metrics_final.py:1-760](file://utils_metrics_final.py#L1-L760)
- [config_ts_final.py:1-208](file://config_ts_final.py#L1-L208)

## Core Components
- Frame-level metrics: POD, FAR, CSI, ETS, SEDI, F1, F2 computed from thresholded predictions
- Event-level metrics: IMD-style overlap-based POD, FAR, CSI, SEDI with lead-time constraints
- Weighted event metrics: Severity-weighted POD/FAR/CSI with lead-time bonuses
- Temporal post-processing: smoothing, persistence filter, Schmitt trigger hysteresis
- Statistical significance: temporal block bootstrap by calendar day
- Discriminative quality: ROC-AUC and PR-AUC from raw probabilities

**Section sources**
- [utils_metrics_final.py:120-190](file://utils_metrics_final.py#L120-L190)
- [utils_metrics_final.py:338-392](file://utils_metrics_final.py#L338-L392)
- [utils_metrics_final.py:575-650](file://utils_metrics_final.py#L575-L650)
- [utils_metrics_final.py:653-760](file://utils_metrics_final.py#L653-L760)
- [evaluate_ts_final.py:612-623](file://evaluate_ts_final.py#L612-L623)

## Architecture Overview
The evaluation workflow:
1. Inference on test set yields raw probabilities and severity labels
2. Optional Platt scaling calibration
3. Temporal smoothing and threshold selection (single or dual thresholds)
4. Persistence filter and severe fast-track
5. Metrics computation (frame, event, weighted event, lead times)
6. Bootstrap confidence intervals on test set

```mermaid
sequenceDiagram
participant Eval as "evaluate_ts_final.py"
participant Utils as "utils_metrics_final.py"
participant Sk as "scikit-learn"
participant Np as "numpy/pandas"
Eval->>Eval : "run_inference()"
Eval->>Eval : "apply temporal smoothing"
Eval->>Utils : "find_best_threshold() or find_best_dual_threshold()"
Eval->>Utils : "apply_persistence() / apply_schmitt_trigger()"
Eval->>Utils : "compute_binary_metrics()"
Eval->>Sk : "roc_curve(), roc_auc_score(), precision_recall_curve(), auc()"
Eval->>Utils : "compute_event_metrics()"
Eval->>Utils : "compute_weighted_event_metrics()"
Eval->>Utils : "compute_lead_times() + summarize_lead_times()"
Eval->>Utils : "bootstrap_test_metrics()"
Utils->>Np : "temporal grouping by date"
Utils->>Utils : "compute_all_metrics() per bootstrap sample"
Utils-->>Eval : "CI summaries"
```

**Diagram sources**
- [evaluate_ts_final.py:500-800](file://evaluate_ts_final.py#L500-L800)
- [utils_metrics_final.py:192-314](file://utils_metrics_final.py#L192-L314)
- [utils_metrics_final.py:653-760](file://utils_metrics_final.py#L653-L760)

## Detailed Component Analysis

### Frame-Level Metrics (POD, FAR, CSI, ETS, SEDI, F1/F2)
- Computation from thresholded predictions using true positives, false positives, false negatives, true negatives
- ETS adjusts for random chance; SEDI is base-rate independent for rare events
- F1 and F2 emphasize recall with different penalties for false positives

```mermaid
flowchart TD
Start(["Thresholded Predictions"]) --> TP["TP = (y_true==1) & (y_pred==1)"]
TP --> FP["FP = (y_true==0) & (y_pred==1)"]
TP --> FN["FN = (y_true==1) & (y_pred==0)"]
TP --> TN["TN = (y_true==0) & (y_pred==0)"]
TP --> POD["POD = TP/(TP+FN)"]
FP --> FAR["FAR = FP/(TP+FP)"]
TP --> CSI["CSI = TP/(TP+FP+FN)"]
TP --> ETS["ETS = (TP-H_rand)/(TP+FP+FN-H_rand)"]
TP --> SEDI["SEDI from H,PoFD"]
TP --> F1["F1 = 2TP/(2TP+FP+FN)"]
TP --> F2["F2 = 5TP/(5TP+4FN+FP)"]
POD --> End(["Frame Metrics"])
FAR --> End
CSI --> End
ETS --> End
SEDI --> End
F1 --> End
F2 --> End
```

**Diagram sources**
- [utils_metrics_final.py:120-152](file://utils_metrics_final.py#L120-L152)

**Section sources**
- [utils_metrics_final.py:120-152](file://utils_metrics_final.py#L120-L152)
- [evaluate_ts_final.py:607](file://evaluate_ts_final.py#L607)

### Event-Level Metrics (IMD-style)
- Events extracted as contiguous 1-segments; short events filtered by minimum length
- Overlap-based matching with lead-time constraint (prediction must occur within a maximum lead window)
- POD, FAR, CSI computed from matched hits, misses, false alarms
- SEDI estimated using hit rate and POFD approximated from counts

```mermaid
flowchart TD
A["Binary Predictions"] --> B["extract_events()"]
A2["Binary Truth"] --> B
B --> C["Filter by min_event_len"]
C --> D["Match true vs pred events"]
D --> E{"Overlap AND within max_lead?"}
E --> |Yes| H["Hit"]
E --> |No| M["Miss or FA"]
H --> POD["POD = hits/(hits+misses)"]
M --> FAR["FAR = false_alarms/(hits+false_alarms)"]
H --> CSI["CSI = hits/(hits+misses+false_alarms)"]
```

**Diagram sources**
- [utils_metrics_final.py:322-392](file://utils_metrics_final.py#L322-L392)

**Section sources**
- [utils_metrics_final.py:338-392](file://utils_metrics_final.py#L338-L392)

### Weighted Event Metrics (Severity-Aware)
- Severity weights applied to hits and misses; false alarms weighted uniformly
- Lead-time bonus: +10% per step early detection (up to a cap)
- Additional bonuses: early detection rate ≥ 60% adds +0.05; safe POD bonus ≥ 0.60 adds +0.05
- Outputs: weighted POD, weighted FAR, weighted CSI, and lead-time corrected wCSI

```mermaid
flowchart TD
Start(["Event Matches + Severity Labels"]) --> W["Assign base weights by severity"]
W --> L["Compute lead steps per hit"]
L --> B["Apply 10% lead bonus per step"]
B --> S["Sum weighted hits / (hits+misses)"]
S --> POD["wPOD"]
B --> F["Sum weighted false alarms"]
F --> FAR["wFAR"]
POD --> CSI["wCSI"]
FAR --> CSI
CSI --> LT["lt_wCSI = min(1.0, CSI + bonus)"]
```

**Diagram sources**
- [utils_metrics_final.py:575-650](file://utils_metrics_final.py#L575-L650)

**Section sources**
- [utils_metrics_final.py:479-518](file://utils_metrics_final.py#L479-L518)
- [utils_metrics_final.py:575-650](file://utils_metrics_final.py#L575-L650)

### Threshold Selection and Post-Processing
- Single threshold grid-search optimizing chosen metric (default: lead-time corrected weighted CSI)
- Dual-threshold Schmitt trigger hysteresis with separate high/low thresholds
- Temporal smoothing (EMA or rolling mean)
- Persistence filter removes short isolated false alarms; severe fast-track keeps runs with high-prob severe events

```mermaid
flowchart TD
Prob["Smoothed Probabilities"] --> T["Grid-search best threshold"]
Prob --> SCH["Dual-threshold Schmitt Trigger"]
T --> P["Persistence Filter"]
SCH --> P
P --> Pred["Final Binary Predictions"]
```

**Diagram sources**
- [utils_metrics_final.py:192-314](file://utils_metrics_final.py#L192-L314)
- [utils_metrics_final.py:23-47](file://utils_metrics_final.py#L23-L47)
- [utils_metrics_final.py:50-77](file://utils_metrics_final.py#L50-L77)

**Section sources**
- [utils_metrics_final.py:192-314](file://utils_metrics_final.py#L192-L314)
- [evaluate_ts_final.py:508-600](file://evaluate_ts_final.py#L508-L600)
- [config_ts_final.py:92-94](file://config_ts_final.py#L92-L94)

### Lead Time Analysis
- Lead time per true event: event_start minus earliest valid prediction within lead window
- Positive lead indicates early detection; negative indicates late detection; None indicates miss
- Summary statistics: mean, median, early detection rate, late detection rate, miss rate
- Optional breakdown by thunderstorm category

```mermaid
flowchart TD
A["True Events"] --> B["Pred Events"]
B --> C{"Overlap within max_lead?"}
C --> |Yes| D["Pick earliest valid prediction"]
C --> |No| E["Mark as miss"]
D --> F["Lead = true_start - pred_start"]
F --> G["Summarize stats"]
E --> G
```

**Diagram sources**
- [utils_metrics_final.py:395-477](file://utils_metrics_final.py#L395-L477)

**Section sources**
- [utils_metrics_final.py:395-477](file://utils_metrics_final.py#L395-L477)
- [evaluate_ts_final.py:629-641](file://evaluate_ts_final.py#L629-L641)

### Discriminative Quality: ROC-AUC and PR-AUC
- Computed from raw probabilities (threshold-independent)
- ROC-AUC measures TPR vs FPR across thresholds
- PR-AUC measures precision-recall trade-offs

```mermaid
sequenceDiagram
participant Eval as "evaluate_ts_final.py"
participant Sk as "sklearn.metrics"
Eval->>Sk : "roc_curve(y_true, y_prob)"
Eval->>Sk : "roc_auc_score(y_true, y_prob)"
Eval->>Sk : "precision_recall_curve(y_true, y_prob)"
Eval->>Sk : "auc(recall, precision)"
Sk-->>Eval : "fpr,tpr,precision,recall,auc"
```

**Diagram sources**
- [evaluate_ts_final.py:612-623](file://evaluate_ts_final.py#L612-L623)

**Section sources**
- [evaluate_ts_final.py:612-623](file://evaluate_ts_final.py#L612-L623)

### Confusion Matrix Analysis
- Counts of true positives, false positives, false negatives, true negatives
- Percentages shown for interpretability

```mermaid
flowchart TD
A["y_true, y_pred"] --> CM["confusion_matrix()"]
CM --> P["Normalize to percentages"]
P --> V["Visualize heatmap"]
```

**Diagram sources**
- [evaluate_ts_final.py:41-56](file://evaluate_ts_final.py#L41-L56)

**Section sources**
- [evaluate_ts_final.py:41-56](file://evaluate_ts_final.py#L41-L56)

### Bootstrap Confidence Intervals (Temporal Block Bootstrap)
- Samples calendar days with replacement; recomputes all metrics on resampled sequences
- Produces point estimates and 95% confidence intervals for frame, event, and weighted metrics
- Uses step-minutes conversion for lead-time summaries

```mermaid
flowchart TD
A["Timestamps"] --> B["Parse dates"]
B --> C["Group indices by date"]
C --> D["Resample dates with replacement"]
D --> E["Rebuild sequences"]
E --> F["Compute all metrics"]
F --> G["Collect bootstrap samples"]
G --> H["Percentiles 2.5% and 97.5%"]
H --> I["Report CI summaries"]
```

**Diagram sources**
- [utils_metrics_final.py:653-760](file://utils_metrics_final.py#L653-L760)

**Section sources**
- [utils_metrics_final.py:653-760](file://utils_metrics_final.py#L653-L760)
- [evaluate_ts_final.py:744-799](file://evaluate_ts_final.py#L744-L799)

### Persistence Filter Effectiveness Measurement
- Counts short false alarms (≤ min_len frames) that are not matched to true events
- Helps quantify impact of persistence filter on reducing spurious detections

**Section sources**
- [utils_metrics_final.py:80-94](file://utils_metrics_final.py#L80-L94)
- [evaluate_ts_final.py:608](file://evaluate_ts_final.py#L608)

## Dependency Analysis
- Evaluation depends on metrics utilities for computations and bootstrapping
- Configuration controls threshold metric, lead time limits, smoothing, and severity weights
- Validation report documents bootstrap CI plan and threshold metric updates

```mermaid
graph TB
Eval["evaluate_ts_final.py"] --> UM["utils_metrics_final.py"]
Eval --> CFG["config_ts_final.py"]
UM --> VC["validation-cv-bootstrap.md"]
Eval --> ER["evaluation_report.md"]
Eval --> AP["analyze_predictions.py"]
```

**Diagram sources**
- [evaluate_ts_final.py:1-908](file://evaluate_ts_final.py#L1-L908)
- [utils_metrics_final.py:1-760](file://utils_metrics_final.py#L1-L760)
- [config_ts_final.py:1-208](file://config_ts_final.py#L1-L208)
- [validation-cv-bootstrap.md:1-89](file://reports/validation-cv-bootstrap.md#L1-L89)
- [evaluation_report.md:1-58](file://reports/evaluation_report.md#L1-L58)
- [analyze_predictions.py:1-64](file://extras/analyze_predictions.py#L1-L64)

**Section sources**
- [evaluate_ts_final.py:1-908](file://evaluate_ts_final.py#L1-L908)
- [utils_metrics_final.py:1-760](file://utils_metrics_final.py#L1-L760)
- [config_ts_final.py:1-208](file://config_ts_final.py#L1-L208)
- [validation-cv-bootstrap.md:1-89](file://reports/validation-cv-bootstrap.md#L1-L89)
- [evaluation_report.md:1-58](file://reports/evaluation_report.md#L1-L58)
- [analyze_predictions.py:1-64](file://extras/analyze_predictions.py#L1-L64)

## Performance Considerations
- Use lead-time corrected weighted CSI as the primary selection metric to balance detection, false alarm control, and timeliness
- Prefer temporal smoothing and Schmitt trigger to reduce temporal chatter without over-aggressive persistence
- Calibrate probabilities (Platt scaling) when applicable to improve discriminative quality
- Monitor short false alarms to assess persistence filter effectiveness
- Use bootstrap CIs to assess statistical significance of differences across models or folds

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
- If bootstrap CI fails to compute, verify timestamps and ensure at least one unique date exists
- If lead-time statistics appear off, confirm step-minutes computation and max-lead steps alignment
- If weighted metrics seem inconsistent, verify severity label mapping and minimum event length
- If ROC/PR curves look unexpected, check that raw probabilities are used (not thresholded predictions)

**Section sources**
- [utils_metrics_final.py:653-760](file://utils_metrics_final.py#L653-L760)
- [evaluate_ts_final.py:602-604](file://evaluate_ts_final.py#L602-L604)
- [evaluate_ts_final.py:629-641](file://evaluate_ts_final.py#L629-L641)

## Conclusion
The evaluation framework combines rigorous frame-level and event-level metrics with severity-aware weighting and lead-time corrections. Temporal block bootstrapping provides robust confidence intervals for statistical comparisons. The system balances detection skill, false alarm control, and operational timeliness—essential for reliable nowcasting.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices

### Metric Interpretation Guidelines
- POD: Fraction of observed events correctly detected
- FAR: Fraction of predicted events that were false alarms
- CSI: Critical success index; balanced measure of hits vs. misses and false alarms
- ETS: Equitable threat score; adjusted for random chance
- SEDI: Base-rate independent skill for rare events
- F1/F2: Harmonic means emphasizing recall; F2 heavier penalty on false negatives
- Event POD/FAR/CSI: Overlap-based; accounts for event duration and timing
- Weighted metrics: Emphasize severe categories; incorporate lead-time bonuses
- ROC-AUC/PR-AUC: Discriminative ability independent of threshold

[No sources needed since this section provides general guidance]

### Benchmarking and Significance Testing Procedures
- Compare models using weighted event metrics and lead-time corrected CSI as primary criteria
- Use temporal block bootstrap to compute 95% confidence intervals for key metrics
- Report point estimates alongside lower and upper bounds for robust comparison
- For model selection, prefer lead-time corrected weighted CSI as the default threshold metric

**Section sources**
- [validation-cv-bootstrap.md:10-14](file://reports/validation-cv-bootstrap.md#L10-L14)
- [evaluate_ts_final.py:756-789](file://evaluate_ts_final.py#L756-L789)
- [config_ts_final.py:92](file://config_ts_final.py#L92)