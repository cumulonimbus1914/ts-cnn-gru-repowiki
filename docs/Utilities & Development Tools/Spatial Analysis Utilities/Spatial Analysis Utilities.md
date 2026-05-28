# Spatial Analysis Utilities

<cite>
**Referenced Files in This Document**
- [utils_spatial_final.py](file://utils_spatial_final.py)
- [generate_static_masks.py](file://extras/generate_static_masks.py)
- [find_global_shifts.py](file://extras/find_global_shifts.py)
- [auto_find_centers.py](file://extras/auto_find_centers.py)
- [find_exact_centers.py](file://extras/find_exact_centers.py)
- [check_mask_center.py](file://extras/check_mask_center.py)
- [preprocess_ts.py](file://preprocess_ts.py)
- [utils_preprocessing.py](file://utils_preprocessing.py)
- [config_ts_final.py](file://config_ts_final.py)
- [prepare_data.py](file://prepare_data.py)
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
This document describes the spatial analysis utilities used for geometric operations, mask generation, and spatial transformation tools in the IR (infrared) satellite imagery processing pipeline. It focuses on:
- Static mask generation for region-of-interest extraction and spatial filtering
- Center detection algorithms for storm identification and tracking
- Global shift detection tools for spatial alignment and coordinate transformation
- Spatial coordinate systems, geometric operations, and preprocessing workflows
- Performance considerations, mask optimization, and center detection accuracy

The goal is to help both technical and non-technical users understand how spatial masks and centers are computed, validated, and integrated into the broader preprocessing and modeling workflow.

## Project Structure
The spatial utilities span several modules:
- Spatial utilities for mask creation and visualization
- Static mask generation scripts for dimension-specific masks
- Center detection and global shift detection tools
- Preprocessing pipeline integrating masks and centers
- Configuration and preparation pipeline for datasets

```mermaid
graph TB
subgraph "Spatial Utilities"
SU["utils_spatial_final.py"]
end
subgraph "Static Mask Generation"
GSM["extras/generate_static_masks.py"]
end
subgraph "Center Detection"
AGC["extras/auto_find_centers.py"]
GSH["extras/find_global_shifts.py"]
FEC["extras/find_exact_centers.py"]
CMZ["extras/check_mask_center.py"]
end
subgraph "Preprocessing Pipeline"
PRE["preprocess_ts.py"]
UPR["utils_preprocessing.py"]
CFG["config_ts_final.py"]
PRP["prepare_data.py"]
end
SU --> PRE
PRE --> PRP
CFG --> PRE
GSM --> PRE
AGC --> PRE
GSH --> PRE
FEC --> PRE
CMZ --> PRE
UPR --> PRP
```

**Diagram sources**
- [utils_spatial_final.py:12-80](file://utils_spatial_final.py#L12-L80)
- [generate_static_masks.py:1-150](file://extras/generate_static_masks.py#L1-L150)
- [auto_find_centers.py:1-101](file://extras/auto_find_centers.py#L1-L101)
- [find_global_shifts.py:1-76](file://extras/find_global_shifts.py#L1-L76)
- [find_exact_centers.py:1-65](file://extras/find_exact_centers.py#L1-L65)
- [check_mask_center.py:1-78](file://extras/check_mask_center.py#L1-L78)
- [preprocess_ts.py:17-112](file://preprocess_ts.py#L17-L112)
- [utils_preprocessing.py:86-162](file://utils_preprocessing.py#L86-L162)
- [config_ts_final.py:106-117](file://config_ts_final.py#L106-L117)
- [prepare_data.py:39-132](file://prepare_data.py#L39-L132)

**Section sources**
- [utils_spatial_final.py:12-80](file://utils_spatial_final.py#L12-L80)
- [generate_static_masks.py:1-150](file://extras/generate_static_masks.py#L1-L150)
- [auto_find_centers.py:1-101](file://extras/auto_find_centers.py#L1-L101)
- [find_global_shifts.py:1-76](file://extras/find_global_shifts.py#L1-L76)
- [find_exact_centers.py:1-65](file://extras/find_exact_centers.py#L1-L65)
- [check_mask_center.py:1-78](file://extras/check_mask_center.py#L1-L78)
- [preprocess_ts.py:17-112](file://preprocess_ts.py#L17-L112)
- [utils_preprocessing.py:86-162](file://utils_preprocessing.py#L86-L162)
- [config_ts_final.py:106-117](file://config_ts_final.py#L106-L117)
- [prepare_data.py:39-132](file://prepare_data.py#L39-L132)

## Core Components
- Spatial mask utilities: Gaussian weight mask and distance map creation for spatial attention and target-zone highlighting
- Static mask generation: Dimension-aware mask creation from median crops and color thresholds
- Center detection: Automatic and exact center-finding via template matching and pixel-wise similarity
- Global shift detection: Cross-dimension alignment using normalized cross-correlation
- Preprocessing integration: Cropping, inpainting, resizing, normalization, and optional optical flow computation
- Configuration: Spatial mask parameters and center coordinates for consistent processing

Key capabilities:
- ROI extraction using masks and centers
- Spatial filtering with attention-weighted regions
- Coordinate transformation and alignment across image dimensions
- Robust preprocessing with contrast enhancement and outlier normalization

**Section sources**
- [utils_spatial_final.py:12-80](file://utils_spatial_final.py#L12-L80)
- [generate_static_masks.py:17-147](file://extras/generate_static_masks.py#L17-L147)
- [auto_find_centers.py:22-98](file://extras/auto_find_centers.py#L22-L98)
- [find_exact_centers.py:13-62](file://extras/find_exact_centers.py#L13-L62)
- [find_global_shifts.py:13-73](file://extras/find_global_shifts.py#L13-L73)
- [preprocess_ts.py:27-112](file://preprocess_ts.py#L27-L112)
- [utils_preprocessing.py:16-162](file://utils_preprocessing.py#L16-L162)
- [config_ts_final.py:106-117](file://config_ts_final.py#L106-L117)

## Architecture Overview
The spatial analysis pipeline integrates mask generation, center detection, and preprocessing into a cohesive workflow. Static masks are generated per image dimension and cached for reuse. During preprocessing, masks are combined with detected overlays and inpainted to remove artifacts. Optional optical flow is computed for motion features. The final dataset is stored in HDF5 for efficient training.

```mermaid
sequenceDiagram
participant Raw as "Raw IR/WV Images"
participant Gen as "Static Mask Generator"
participant Pre as "Preprocess Module"
participant Cfg as "Config"
participant Prep as "Prepare Data"
participant Out as "HDF5 Dataset"
Raw->>Gen : "Median crop + HSV thresholds"
Gen-->>Raw : "Static mask per dimension"
Raw->>Pre : "Load image + CENTER_MAP"
Pre->>Cfg : "Read MASK_CENTER, STATION_RADIUS_PX"
Pre->>Pre : "Crop around center<br/>Combine masks + inpaint"
Pre-->>Prep : "Normalized IR/WV arrays"
Prep->>Prep : "Compute texture, cooling, flow"
Prep-->>Out : "Write HDF5 with channels"
```

**Diagram sources**
- [generate_static_masks.py:116-147](file://extras/generate_static_masks.py#L116-L147)
- [preprocess_ts.py:27-112](file://preprocess_ts.py#L27-L112)
- [config_ts_final.py:106-117](file://config_ts_final.py#L106-L117)
- [prepare_data.py:64-125](file://prepare_data.py#L64-L125)

## Detailed Component Analysis

### Spatial Mask Utilities
- Gaussian weight mask: Creates a normalized 2D Gaussian centered at a given pixel with configurable spread. Mean normalization ensures no overall brightness change.
- Distance map: Produces a normalized Euclidean distance map from a center, with a sharp peak at the station radius to highlight the target zone.

```mermaid
flowchart TD
Start(["Inputs: H, W, center, sigma"]) --> Grid["Create coordinate grids"]
Grid --> Dist["Compute squared distance from center"]
Dist --> Exp["Apply Gaussian exp(-d^2 / 2σ^2)"]
Exp --> Norm["Normalize so mean == 1"]
Norm --> Out["Return float32 mask"]
```

**Diagram sources**
- [utils_spatial_final.py:12-34](file://utils_spatial_final.py#L12-L34)

Accuracy and performance:
- Uses vectorized NumPy operations for speed
- Normalization prevents illumination bias in attention weighting

**Section sources**
- [utils_spatial_final.py:12-34](file://utils_spatial_final.py#L12-L34)
- [utils_spatial_final.py:36-65](file://utils_spatial_final.py#L36-L65)

### Static Mask Generation
- Purpose: Generate dimension-specific masks to filter out non-storm regions and overlays
- Method:
  - Group images by dimension and prioritize seasonal clear conditions
  - Compute median crop and convert to HSV
  - Thresholds for grid, cyan, and boundary regions; combine with OR operation
  - Dilate once to keep mask tight and prevent information loss
  - Save per-dimension static masks

```mermaid
flowchart TD
Scan["Scan raw images by dimension"] --> Score["Score clarity (mean intensity in crop)"]
Score --> Select["Select top clear crops"]
Select --> Median["Compute median crop"]
Median --> HSV["Convert to HSV"]
HSV --> Thresh["Apply thresholds (grid/cyan/boundary)"]
Thresh --> Combine["Bitwise OR to form overlay mask"]
Combine --> Dilate["Single dilation"]
Dilate --> Save["Save static mask PNG"]
```

**Diagram sources**
- [generate_static_masks.py:86-147](file://extras/generate_static_masks.py#L86-L147)

Optimization:
- Early exit when sufficient files per dimension are collected
- Two-pass scoring prioritizes seasonal clear images
- Minimal IO overhead by caching masks and reusing

**Section sources**
- [generate_static_masks.py:17-147](file://extras/generate_static_masks.py#L17-L147)

### Center Detection Algorithms
- Automatic center detection:
  - Extract overlay mask from reference dimension
  - Define a template around Nagpur center
  - Perform normalized cross-correlation on binary mask across dimensions
  - Record center with confidence and optionally save verification crops

```mermaid
sequenceDiagram
participant Ref as "Reference Image"
participant Dim as "Other Dimensions"
participant TM as "Template Matching"
participant Out as "Centers Dict"
Ref->>Ref : "Build overlay mask"
Ref->>Ref : "Extract template around center"
loop For each dimension
Dim->>TM : "Compute normalized cross-correlation"
TM-->>Out : "Record (x,y) + confidence"
end
```

**Diagram sources**
- [auto_find_centers.py:46-90](file://extras/auto_find_centers.py#L46-L90)

- Exact center refinement:
  - Compare small crops around the reference center using MSE
  - Search a small neighborhood to refine center coordinates

```mermaid
flowchart TD
Ref["Reference crop around (cx,cy)"] --> Loop["Iterate dx, dy in small range"]
Loop --> Crop["Extract crop at (cx+dx, cy+dy)"]
Crop --> MSE["Compute MSE vs reference"]
MSE --> Best["Track best (cx,cy) with lowest MSE"]
```

**Diagram sources**
- [find_exact_centers.py:32-61](file://extras/find_exact_centers.py#L32-L61)

- Interactive mask center tool:
  - Opens a raw image and allows clicking to convert between raw and processed coordinates
  - Prints recommended configuration updates for MASK_CENTER

```mermaid
sequenceDiagram
participant User as "User"
participant GUI as "Interactive Plot"
participant Conv as "Coordinate Converter"
User->>GUI : "Click on raw image"
GUI->>Conv : "Map (x_raw, y_raw) -> (x_proc, y_proc)"
Conv-->>User : "Print processed coordinates"
User-->>User : "Update config with MASK_CENTER"
```

**Diagram sources**
- [check_mask_center.py:45-71](file://extras/check_mask_center.py#L45-L71)

Accuracy considerations:
- Template matching on binary masks improves robustness to illumination changes
- MSE-based refinement reduces sub-pixel jitter
- Confidence scores from normalized cross-correlation guide trust in detections

**Section sources**
- [auto_find_centers.py:22-98](file://extras/auto_find_centers.py#L22-L98)
- [find_exact_centers.py:13-62](file://extras/find_exact_centers.py#L13-L62)
- [check_mask_center.py:5-71](file://extras/check_mask_center.py#L5-L71)

### Global Shift Detection Tools
- Goal: Align different image dimensions by detecting shifts in coastline/grid patterns
- Method:
  - Build boundary mask from HSV ranges
  - Use normalized cross-correlation (template matching) with the reference dimension
  - Derive shifted centers by offset from match location
  - Optionally save verification crops

```mermaid
flowchart TD
Load["Load reference and candidate masks"] --> Match["Normalized cross-correlation"]
Match --> Locate["Find best match location"]
Locate --> Offset["Compute center offsets"]
Offset --> Report["Report aligned centers"]
```

**Diagram sources**
- [find_global_shifts.py:13-73](file://extras/find_global_shifts.py#L13-L73)

Integration:
- Used to populate CENTER_MAP for each dimension
- Ensures consistent cropping across heterogeneous image sizes

**Section sources**
- [find_global_shifts.py:13-73](file://extras/find_global_shifts.py#L13-L73)

### Spatial Coordinate Systems and Geometric Operations
- Pixel coordinate mapping:
  - Raw image coordinates are mapped to processed coordinates using resize ratios
  - Used for translating user-selected points into model-ready coordinates
- Cropping geometry:
  - Fixed-size crops around centers for consistent input tensors
  - Padding and resizing preserve aspect ratios and center alignment

```mermaid
flowchart TD
Raw["Raw (w_raw, h_raw)"] --> Scale["Scale to processed (224, 224)"]
Scale --> Proc["Processed (w_proc, h_proc)"]
Proc --> Map["x_proc = x_raw * 224 / w_raw<br/>y_proc = y_raw * 224 / h_raw"]
```

**Diagram sources**
- [check_mask_center.py:57-59](file://extras/check_mask_center.py#L57-L59)

**Section sources**
- [check_mask_center.py:23-30](file://extras/check_mask_center.py#L23-L30)
- [check_mask_center.py:57-64](file://extras/check_mask_center.py#L57-L64)

### Preprocessing and Spatial Filtering Workflow
- Overlay cleaning:
  - Build masks from HSV ranges and static masks
  - Combine masks and inpaint to remove artifacts
- Grayscale proxy and feature extraction:
  - Convert to grayscale proxy where brighter indicates colder clouds
  - Compute CCD metrics and connected components
- Resizing and padding:
  - Resize maintaining aspect ratio, then pad to square
- Optional optical flow:
  - Compute motion magnitude using a lightweight method

```mermaid
flowchart TD
Read["Read image"] --> Crop["Crop around center"]
Crop --> HSV["Convert to HSV"]
HSV --> Static["Load or build static mask"]
Static --> Overlay["Build overlay mask"]
Overlay --> Inpaint["Inpaint artifacts"]
Inpaint --> Gray["Grayscale proxy"]
Gray --> Features["Compute CCD + complexity"]
Features --> Resize["Resize maintaining aspect ratio"]
Resize --> Pad["Pad to square"]
Pad --> Output["Final normalized image"]
```

**Diagram sources**
- [preprocess_ts.py:27-112](file://preprocess_ts.py#L27-L112)

**Section sources**
- [preprocess_ts.py:27-112](file://preprocess_ts.py#L27-L112)
- [utils_preprocessing.py:16-162](file://utils_preprocessing.py#L16-L162)

### Configuration and Dataset Preparation
- Spatial configuration:
  - MASK_CENTER defines the center for attention weighting
  - STATION_RADIUS_PX sets the target zone boundary radius
  - USE_MASK toggles spatial attention
- Dataset preparation:
  - Matches IR and WV images by timestamp
  - Computes texture, cooling, flow, and differences
  - Writes channels to HDF5 for training

```mermaid
graph TB
CFG["config_ts_final.py<br/>MASK_CENTER, STATION_RADIUS_PX, USE_MASK"] --> PRE["preprocess_ts.py<br/>Cropping + masking"]
PRE --> PRP["prepare_data.py<br/>Feature engineering + HDF5 write"]
```

**Diagram sources**
- [config_ts_final.py:106-117](file://config_ts_final.py#L106-L117)
- [preprocess_ts.py:27-112](file://preprocess_ts.py#L27-L112)
- [prepare_data.py:64-125](file://prepare_data.py#L64-L125)

**Section sources**
- [config_ts_final.py:106-117](file://config_ts_final.py#L106-L117)
- [prepare_data.py:39-132](file://prepare_data.py#L39-L132)

## Dependency Analysis
- Spatial utilities depend on NumPy and Matplotlib/PIL for visualization
- Static mask generation depends on OpenCV for image I/O, HSV conversion, and thresholding
- Center detection scripts depend on OpenCV for template matching and image operations
- Preprocessing depends on OpenCV, Pandas, and CV utilities
- Dataset preparation depends on HDF5, PyTorch, and CV utilities

```mermaid
graph TB
SU["utils_spatial_final.py"] --> NP["NumPy"]
SU --> MPL["Matplotlib/PIL"]
GSM["generate_static_masks.py"] --> CV["OpenCV"]
AGC["auto_find_centers.py"] --> CV
GSH["find_global_shifts.py"] --> CV
FEC["find_exact_centers.py"] --> CV
CMZ["check_mask_center.py"] --> MPL
PRE["preprocess_ts.py"] --> CV
PRE --> PD["Pandas"]
PRP["prepare_data.py"] --> CV
PRP --> H5["HDF5"]
PRP --> TOR["PyTorch"]
UPR["utils_preprocessing.py"] --> CV
UPR --> TOR
```

**Diagram sources**
- [utils_spatial_final.py:6-8](file://utils_spatial_final.py#L6-L8)
- [generate_static_masks.py:2-6](file://extras/generate_static_masks.py#L2-L6)
- [auto_find_centers.py:1-5](file://extras/auto_find_centers.py#L1-L5)
- [find_global_shifts.py:1-4](file://extras/find_global_shifts.py#L1-L4)
- [find_exact_centers.py:1-4](file://extras/find_exact_centers.py#L1-L4)
- [check_mask_center.py:1-3](file://extras/check_mask_center.py#L1-L3)
- [preprocess_ts.py:1-5](file://preprocess_ts.py#L1-L5)
- [prepare_data.py:1-12](file://prepare_data.py#L1-L12)
- [utils_preprocessing.py:8-12](file://utils_preprocessing.py#L8-L12)

**Section sources**
- [utils_spatial_final.py:6-8](file://utils_spatial_final.py#L6-L8)
- [generate_static_masks.py:2-6](file://extras/generate_static_masks.py#L2-L6)
- [auto_find_centers.py:1-5](file://extras/auto_find_centers.py#L1-L5)
- [find_global_shifts.py:1-4](file://extras/find_global_shifts.py#L1-L4)
- [find_exact_centers.py:1-4](file://extras/find_exact_centers.py#L1-L4)
- [check_mask_center.py:1-3](file://extras/check_mask_center.py#L1-L3)
- [preprocess_ts.py:1-5](file://preprocess_ts.py#L1-L5)
- [prepare_data.py:1-12](file://prepare_data.py#L1-L12)
- [utils_preprocessing.py:8-12](file://utils_preprocessing.py#L8-L12)

## Performance Considerations
- Vectorized operations: Gaussian and distance computations use NumPy’s broadcasting for speed
- Mask generation optimization: Early exits and two-pass scoring reduce unnecessary reads
- Template matching: Normalized cross-correlation is efficient and robust; restrict search ranges to improve speed
- Inpainting and resizing: Use appropriate interpolation and padding to minimize artifacts
- Dataset I/O: HDF5 compression and chunking improve throughput during training
- Optical flow: Lightweight method is used to balance accuracy and compute cost

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
- Static mask quality:
  - If masks over-segment or under-segment, adjust HSV thresholds and dilation iterations
  - Ensure median crop captures representative scene without excessive cloud cover
- Center detection failures:
  - Verify CENTER_MAP entries align with reference dimension
  - Increase template size or confidence thresholds if matching is noisy
- Coordinate mapping errors:
  - Confirm raw image dimensions and processed size ratios
  - Use interactive tool to validate conversions
- Preprocessing artifacts:
  - Inspect overlay masks and inpainting results
  - Adjust thresholds or expand static mask coverage

**Section sources**
- [generate_static_masks.py:122-140](file://extras/generate_static_masks.py#L122-L140)
- [auto_find_centers.py:66-74](file://extras/auto_find_centers.py#L66-L74)
- [check_mask_center.py:57-64](file://extras/check_mask_center.py#L57-L64)
- [preprocess_ts.py:50-67](file://preprocess_ts.py#L50-L67)

## Conclusion
The spatial analysis utilities provide a robust foundation for ROI extraction, spatial filtering, and coordinate alignment in IR imagery preprocessing. By combining static masks, precise center detection, and efficient preprocessing, the pipeline supports accurate storm identification and tracking while maintaining computational efficiency. Proper tuning of thresholds, templates, and masks ensures high accuracy and reproducibility across diverse image dimensions.