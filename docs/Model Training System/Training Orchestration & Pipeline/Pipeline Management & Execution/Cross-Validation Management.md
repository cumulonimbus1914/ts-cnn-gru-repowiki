# Cross-Validation Management

<cite>
**Referenced Files in This Document**
- [master.py](file://master.py)
- [train_ts_final.py](file://train_ts_final.py)
- [evaluate_ts_final.py](file://evaluate_ts_final.py)
- [evaluate_ensemble.py](file://evaluate_ensemble.py)
- [evaluate_ablation.py](file://evaluate_ablation.py)
- [dataset_ts_final.py](file://dataset_ts_final.py)
- [config_ts_final.py](file://config_ts_final.py)
- [model_ts_final.py](file://model_ts_final.py)
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

## Introduction
This document explains the walk-forward cross-validation (CV) system used in the Nagpur TS Nowcasting pipeline. It details the three-fold validation strategy with time-based splits, dynamic date boundary determination, and temporal consistency requirements. It also covers fold selection via command-line arguments, dataset splitting logic, and the evaluation workflow across folds. Finally, it highlights the importance of chronological data splitting for realistic weather forecasting evaluation and provides guidance for fold-specific configurations and performance comparisons.

## Project Structure
The cross-validation system spans several scripts:
- A master orchestration script that runs training and evaluation phases for a selected fold.
- A training script that builds time-based splits and trains models per fold.
- An evaluation script that computes metrics on the test set for a selected fold.
- An ensemble evaluation script that combines best and SWA models for a selected fold.
- An ablation study script that evaluates feature contributions on the test set for a selected fold.
- A dataset module that provides time-stamped samples and supports time-based filtering.
- A configuration module that defines training and evaluation defaults and fold boundaries.

```mermaid
graph TB
Master["master.py<br/>Orchestration"] --> Train["train_ts_final.py<br/>Train per fold"]
Master --> Eval["evaluate_ts_final.py<br/>Evaluate Best per fold"]
Master --> Ensem["evaluate_ensemble.py<br/>Evaluate Ensemble per fold"]
Master --> Ablat["evaluate_ablation.py<br/>Ablation per fold"]
Train --> DS["dataset_ts_final.py<br/>Timestamped samples"]
Eval --> DS
Ensem --> DS
Ablat --> DS
Train --> CFG["config_ts_final.py<br/>Defaults & fold dates"]
Eval --> CFG
Ensem --> CFG
Ablat --> CFG
Train --> Model["model_ts_final.py<br/>CNN-GRU architecture"]
Eval --> Model
Ensem --> Model
Ablat --> Model
```

**Diagram sources**
- [master.py:39-98](file://master.py#L39-L98)
- [train_ts_final.py:142-234](file://train_ts_final.py#L142-L234)
- [evaluate_ts_final.py:361-424](file://evaluate_ts_final.py#L361-L424)
- [evaluate_ensemble.py:84-153](file://evaluate_ensemble.py#L84-L153)
- [evaluate_ablation.py:172-255](file://evaluate_ablation.py#L172-L255)
- [dataset_ts_final.py:57-101](file://dataset_ts_final.py#L57-L101)
- [config_ts_final.py:186-188](file://config_ts_final.py#L186-L188)
- [model_ts_final.py:83-200](file://model_ts_final.py#L83-L200)

**Section sources**
- [master.py:39-108](file://master.py#L39-L108)
- [train_ts_final.py:142-234](file://train_ts_final.py#L142-L234)
- [evaluate_ts_final.py:361-424](file://evaluate_ts_final.py#L361-L424)
- [evaluate_ensemble.py:84-153](file://evaluate_ensemble.py#L84-L153)
- [evaluate_ablation.py:172-255](file://evaluate_ablation.py#L172-L255)
- [dataset_ts_final.py:57-101](file://dataset_ts_final.py#L57-L101)
- [config_ts_final.py:186-188](file://config_ts_final.py#L186-L188)
- [model_ts_final.py:83-200](file://model_ts_final.py#L83-L200)

## Core Components
- Fold selection via command-line argument: The master pipeline accepts a fold argument (1, 2, or 3) and passes it downstream to training and evaluation scripts.
- Time-based dataset partitioning: Both training and evaluation scripts build a timestamp-indexed metadata DataFrame and select train/validation/test partitions based on fold-specific or configuration-defined end dates.
- Dynamic date boundary determination: For folds 1 and 2, fixed end dates are used; for fold 3, the end dates are taken from configuration.
- Temporal consistency: The dataset ensures chronological ordering and uses timestamps to define splits, preventing leakage across time.

**Section sources**
- [master.py:40-43](file://master.py#L40-L43)
- [train_ts_final.py:213-229](file://train_ts_final.py#L213-L229)
- [evaluate_ts_final.py:410-426](file://evaluate_ts_final.py#L410-L426)
- [evaluate_ensemble.py:135-151](file://evaluate_ensemble.py#L135-L151)
- [evaluate_ablation.py:229-247](file://evaluate_ablation.py#L229-L247)

## Architecture Overview
The cross-validation architecture follows a walk-forward strategy:
- Fold 1: Train end date is 2025-03-01; Validation end date is 2025-05-01.
- Fold 2: Train end date is 2025-05-01; Validation end date is 2025-07-01.
- Fold 3: Train end date and Validation end date are taken from configuration.

```mermaid
sequenceDiagram
participant User as "User"
participant Master as "master.py"
participant Train as "train_ts_final.py"
participant Eval as "evaluate_ts_final.py"
participant Ensem as "evaluate_ensemble.py"
participant Ablat as "evaluate_ablation.py"
User->>Master : "--fold 1|2|3"
Master->>Train : "--fold <fold>"
Train->>Train : "Build timestamp metadata"
Train->>Train : "Select train/val indices by fold dates"
Train->>Train : "Train model per fold"
Train-->>Master : "Save best and SWA models"
Master->>Eval : "<best_model_fold_<fold>.pth> --fold <fold>"
Eval->>Eval : "Load dataset and build timestamp metadata"
Eval->>Eval : "Select val/test indices by fold dates"
Eval->>Eval : "Find threshold on validation set"
Eval->>Eval : "Evaluate test set and produce metrics"
Master->>Ensem : "<best_model_fold_<fold>.pth> <swa_model_fold_<fold>.pth> --fold <fold>"
Ensem->>Ensem : "Load best and SWA models"
Ensem->>Ensem : "Select val/test indices by fold dates"
Ensem->>Ensem : "Ensemble predictions and evaluate"
Master->>Ablat : "<best_model_fold_<fold>.pth> <swa_model_fold_<fold>.pth> --fold <fold>"
Ablat->>Ablat : "Select test indices by fold dates"
Ablat->>Ablat : "Run ablation suite on test set"
```

**Diagram sources**
- [master.py:76-98](file://master.py#L76-L98)
- [train_ts_final.py:213-229](file://train_ts_final.py#L213-L229)
- [evaluate_ts_final.py:410-426](file://evaluate_ts_final.py#L410-L426)
- [evaluate_ensemble.py:135-151](file://evaluate_ensemble.py#L135-L151)
- [evaluate_ablation.py:229-247](file://evaluate_ablation.py#L229-L247)

## Detailed Component Analysis

### Fold Selection and Orchestration
- The master pipeline parses the fold argument and runs training and evaluation phases accordingly. It constructs model paths per fold and orchestrates downstream evaluation and ablation steps.

```mermaid
flowchart TD
Start(["Start"]) --> ParseArgs["Parse --fold argument"]
ParseArgs --> RunTrain["Run train_ts_final.py --fold <fold>"]
RunTrain --> EvalBest["Run evaluate_ts_final.py <best_fold.pth> --fold <fold>"]
EvalBest --> EvalSWA["Run evaluate_ts_final.py <swa_fold_swa.pth> --fold <fold>"]
EvalSWA --> EvalEns["Run evaluate_ensemble.py <best_fold.pth> <swa_fold_swa.pth> --fold <fold>"]
EvalEns --> RunAblat["Run evaluate_ablation.py <best_fold.pth> <swa_fold_swa.pth> --fold <fold>"]
RunAblat --> End(["End"])
```

**Diagram sources**
- [master.py:39-108](file://master.py#L39-L108)

**Section sources**
- [master.py:39-108](file://master.py#L39-L108)

### Training Script: Time-Based Splitting and Model Selection
- The training script builds a timestamp-indexed metadata DataFrame from the dataset’s samples and selects train and validation indices based on the chosen fold.
- For folds 1 and 2, fixed end dates are used; for fold 3, configuration-defined dates are used.
- The script applies pre-event labeling and class-balanced sampling, then trains the model and saves the best-performing checkpoint per fold.

```mermaid
flowchart TD
StartTrain(["Start Training"]) --> LoadMeta["Load dataset and build timestamp metadata"]
LoadMeta --> SelectDates{"Fold == 1 or 2?"}
SelectDates --> |Yes| FixedDates["Use fixed end dates"]
SelectDates --> |No| ConfigDates["Use TRAIN_END/VAL_END from config"]
FixedDates --> Split["Split indices by train/val"]
ConfigDates --> Split
Split --> PreEvent["Apply pre-event labeling"]
PreEvent --> Sampler["Build class-balanced sampler"]
Sampler --> TrainLoop["Training loop with validation metrics"]
TrainLoop --> SaveBest["Save best model per fold"]
SaveBest --> EndTrain(["End Training"])
```

**Diagram sources**
- [train_ts_final.py:204-234](file://train_ts_final.py#L204-L234)
- [train_ts_final.py:236-279](file://train_ts_final.py#L236-L279)
- [train_ts_final.py:674-683](file://train_ts_final.py#L674-L683)

**Section sources**
- [train_ts_final.py:204-234](file://train_ts_final.py#L204-L234)
- [train_ts_final.py:236-279](file://train_ts_final.py#L236-L279)
- [train_ts_final.py:674-683](file://train_ts_final.py#L674-L683)

### Evaluation Script: Threshold Derivation and Test Set Metrics
- The evaluation script mirrors the training split logic to select validation and test sets.
- It derives a threshold on the validation set and applies temporal smoothing and persistence filtering to produce final predictions on the test set.
- It computes frame-level and event-level metrics, lead-time statistics, and severity breakdowns.

```mermaid
sequenceDiagram
participant Eval as "evaluate_ts_final.py"
participant DS as "UpgradedTSDataset"
participant Model as "CNN_GRU_TS"
Eval->>DS : "Build timestamp metadata"
Eval->>Eval : "Select val/test indices by fold dates"
Eval->>Model : "Load best model"
Eval->>Eval : "Run inference on val"
Eval->>Eval : "Derive threshold on val"
Eval->>Eval : "Calibrate probabilities (optional)"
Eval->>Eval : "Temporal smoothing + persistence"
Eval->>Eval : "Compute frame/event/lead-time metrics"
```

**Diagram sources**
- [evaluate_ts_final.py:398-426](file://evaluate_ts_final.py#L398-L426)
- [evaluate_ts_final.py:505-601](file://evaluate_ts_final.py#L505-L601)
- [evaluate_ts_final.py:628-714](file://evaluate_ts_final.py#L628-L714)

**Section sources**
- [evaluate_ts_final.py:398-426](file://evaluate_ts_final.py#L398-L426)
- [evaluate_ts_final.py:505-601](file://evaluate_ts_final.py#L505-L601)
- [evaluate_ts_final.py:628-714](file://evaluate_ts_final.py#L628-L714)

### Ensemble Evaluation: Best + SWA Averaging
- The ensemble script loads both the best and SWA models, selects the same validation and test partitions, and averages their predictions to reduce false alarms while preserving detection rates.

```mermaid
flowchart TD
StartEns(["Start Ensemble Evaluation"]) --> LoadModels["Load Best and SWA models"]
LoadModels --> SelectDatesEns["Select val/test indices by fold dates"]
SelectDatesEns --> InferBest["Infer on val/test (Best)"]
InferBest --> InferSWA["Infer on val/test (SWA)"]
InferSWA --> EnsembleAvg["Average predictions (60/40)"]
EnsembleAvg --> Calibrate["Optional Platt scaling on val"]
Calibrate --> PostProc["Temporal smoothing + persistence"]
PostProc --> Metrics["Compute metrics and save outputs"]
Metrics --> EndEns(["End Ensemble Evaluation"])
```

**Diagram sources**
- [evaluate_ensemble.py:155-173](file://evaluate_ensemble.py#L155-L173)
- [evaluate_ensemble.py:174-200](file://evaluate_ensemble.py#L174-L200)

**Section sources**
- [evaluate_ensemble.py:155-173](file://evaluate_ensemble.py#L155-L173)
- [evaluate_ensemble.py:174-200](file://evaluate_ensemble.py#L174-L200)

### Ablation Study: Feature Contribution Analysis
- The ablation script evaluates the contribution of each input feature by zeroing it out and measuring the impact on weighted event-level metrics on the test set for a selected fold.

```mermaid
flowchart TD
StartAblat(["Start Ablation"]) --> SelectDatesAblat["Select test indices by fold dates"]
SelectDatesAblat --> IterateTargets["Iterate over ablation targets"]
IterateTargets --> ZeroTarget["Zero out target feature"]
ZeroTarget --> InferAblat["Infer on test"]
InferAblat --> PostProcAblat["Temporal smoothing + persistence"]
PostProcAblat --> MetricsAblat["Compute weighted event metrics"]
MetricsAblat --> SaveAblat["Save results to CSV"]
SaveAblat --> EndAblat(["End Ablation"])
```

**Diagram sources**
- [evaluate_ablation.py:229-247](file://evaluate_ablation.py#L229-L247)
- [evaluate_ablation.py:134-148](file://evaluate_ablation.py#L134-L148)

**Section sources**
- [evaluate_ablation.py:229-247](file://evaluate_ablation.py#L229-L247)
- [evaluate_ablation.py:134-148](file://evaluate_ablation.py#L134-L148)

### Dataset and Timestamp-Based Partitioning
- The dataset module provides timestamped samples and supports building a metadata DataFrame for fast, RAM-based time-based filtering.
- The training and evaluation scripts rely on this metadata to select train, validation, and test partitions without re-reading disk data.

```mermaid
classDiagram
class UpgradedTSDataset {
+samples
+timestamps
+__getitem__(idx)
+_build_samples()
+_compute_severity_labels()
}
class IRSequenceDataset {
+files
+timestamps
+_parse_ts(filename)
+_build_samples()
}
UpgradedTSDataset --|> IRSequenceDataset : "extends"
```

**Diagram sources**
- [dataset_ts_final.py:57-101](file://dataset_ts_final.py#L57-L101)
- [dataset_ts_final.py:367-403](file://dataset_ts_final.py#L367-L403)

**Section sources**
- [dataset_ts_final.py:57-101](file://dataset_ts_final.py#L57-L101)
- [dataset_ts_final.py:367-403](file://dataset_ts_final.py#L367-L403)

### Configuration and Fold Date Definitions
- Fold 1 and 2 use fixed end dates embedded in the evaluation scripts.
- Fold 3 uses TRAIN_END and VAL_END from configuration.
- The configuration module centralizes defaults for training and evaluation.

```mermaid
flowchart TD
FoldSel["Fold Selection"] --> F1["Fold 1: fixed dates"]
FoldSel --> F2["Fold 2: fixed dates"]
FoldSel --> F3["Fold 3: config dates"]
F1 --> Dates1["Train end: 2025-03-01<br/>Val end: 2025-05-01"]
F2 --> Dates2["Train end: 2025-05-01<br/>Val end: 2025-07-01"]
F3 --> Dates3["Train end: TRAIN_END<br/>Val end: VAL_END"]
```

**Diagram sources**
- [evaluate_ts_final.py:411-419](file://evaluate_ts_final.py#L411-L419)
- [train_ts_final.py:214-222](file://train_ts_final.py#L214-L222)
- [config_ts_final.py:186-188](file://config_ts_final.py#L186-L188)

**Section sources**
- [evaluate_ts_final.py:411-419](file://evaluate_ts_final.py#L411-L419)
- [train_ts_final.py:214-222](file://train_ts_final.py#L214-L222)
- [config_ts_final.py:186-188](file://config_ts_final.py#L186-L188)

## Dependency Analysis
- The master pipeline depends on training and evaluation scripts and passes the fold argument downstream.
- Training and evaluation scripts depend on the dataset module for timestamped samples and on the configuration module for defaults and fold dates.
- The model module is shared across training and evaluation.

```mermaid
graph TB
Master["master.py"] --> Train["train_ts_final.py"]
Master --> Eval["evaluate_ts_final.py"]
Master --> Ensem["evaluate_ensemble.py"]
Master --> Ablat["evaluate_ablation.py"]
Train --> DS["dataset_ts_final.py"]
Eval --> DS
Ensem --> DS
Ablat --> DS
Train --> CFG["config_ts_final.py"]
Eval --> CFG
Ensem --> CFG
Ablat --> CFG
Train --> Model["model_ts_final.py"]
Eval --> Model
Ensem --> Model
Ablat --> Model
```

**Diagram sources**
- [master.py:76-98](file://master.py#L76-L98)
- [train_ts_final.py:213-229](file://train_ts_final.py#L213-L229)
- [evaluate_ts_final.py:410-426](file://evaluate_ts_final.py#L410-L426)
- [evaluate_ensemble.py:135-151](file://evaluate_ensemble.py#L135-L151)
- [evaluate_ablation.py:229-247](file://evaluate_ablation.py#L229-L247)
- [dataset_ts_final.py:57-101](file://dataset_ts_final.py#L57-L101)
- [config_ts_final.py:186-188](file://config_ts_final.py#L186-L188)
- [model_ts_final.py:83-200](file://model_ts_final.py#L83-L200)

**Section sources**
- [master.py:76-98](file://master.py#L76-L98)
- [train_ts_final.py:213-229](file://train_ts_final.py#L213-L229)
- [evaluate_ts_final.py:410-426](file://evaluate_ts_final.py#L410-L426)
- [evaluate_ensemble.py:135-151](file://evaluate_ensemble.py#L135-L151)
- [evaluate_ablation.py:229-247](file://evaluate_ablation.py#L229-L247)
- [dataset_ts_final.py:57-101](file://dataset_ts_final.py#L57-L101)
- [config_ts_final.py:186-188](file://config_ts_final.py#L186-L188)
- [model_ts_final.py:83-200](file://model_ts_final.py#L83-L200)

## Performance Considerations
- Time-based splits ensure that validation and test sets are temporally ahead of training, preventing leakage and enabling realistic forecasting evaluation.
- Using a fixed validation period for folds 1 and 2 allows fair comparison across folds.
- For fold 3, dynamic end dates enable ongoing evaluation as new data becomes available.
- The evaluation pipeline includes temporal smoothing and persistence filtering to stabilize predictions and reduce short-lived false alarms.

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
- If a model checkpoint is not found for a given fold, ensure the training phase completed successfully and saved the expected files.
- If validation or test indices appear empty, verify that the dataset contains timestamps after the selected train end date.
- If threshold derivation fails, confirm that the validation set contains both positive and negative samples.
- If ensemble evaluation does not improve over individual models, review the calibration and smoothing steps.

**Section sources**
- [evaluate_ts_final.py:431-446](file://evaluate_ts_final.py#L431-L446)
- [evaluate_ensemble.py:160-172](file://evaluate_ensemble.py#L160-L172)

## Conclusion
The Nagpur TS Nowcasting pipeline employs a robust walk-forward cross-validation strategy with time-based splits. Folds 1 and 2 use fixed end dates to provide stable, comparable evaluations, while fold 3 leverages configuration-defined dates for dynamic coverage. The training and evaluation scripts consistently partition datasets by timestamps, derive thresholds on validation sets, and apply temporal smoothing and persistence filtering to produce reliable metrics. This approach ensures realistic weather forecasting evaluation and enables meaningful performance comparisons across folds.