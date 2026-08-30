# SeetaPsych Gaze Follow

> Gaze following estimation modules for SeetaPsych. Detects heads in scene and estimates where each person is looking (gaze point on image plane, and social gaze relations between people).

## Usage

This project is already included in the seetapsych-lib default configuration. Download and use it via `seetapsych-manager download`.

For usage, refer to [SeetaPsych](https://github.com/seetapsych/seetapsych-lib).

Install optional algorithm dependencies:

```bash
uv pip install seetapsych-gaze-follow[all]
```

The algorithm modules automatically download model weights from ModelScope on first use, so the initial run may be slower due to model downloading.

You can additionally add this algorithm module using the following methods.

### WebUI

Run `seetapsych-webui` with the `--files` argument to use it.

```
seetapsych-webui --files \
  seetapsych_gaze_follow/modules/head_detection.yml \
  seetapsych_gaze_follow/modules/cosi.yml
```

### Programmatic Usage

Add the following code in your program to use this algorithm module.

```python
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline

factory = Factory()
factory.load_file_modules("seetapsych_gaze_follow/modules/head_detection.yml")
factory.load_file_modules("seetapsych_gaze_follow/modules/cosi.yml")

pipeline = Pipeline(factory, ...)

pipeline.add_attributes("head/detection", "head/gaze_point")
# Or for dyadic social gaze:
# pipeline.add_attributes('head/detection', 'head/social_gaze')
```

## Module Pipeline

The gaze following pipeline consists of two stages loaded from separate module configs:

1. **HeadDetection** ([head_detection.yml](seetapsych_gaze_follow/modules/head_detection.yml)) — YOLO-based multi-head detector, plus an optional HeadSelection post-processor.
2. **CoSI** ([cosi.yml](seetapsych_gaze_follow/modules/cosi.yml)) — confidence-coordinated spatial integration model for gaze point and social gaze relation prediction.

Dependency graph:
- `head/detection` → `head/gaze_point` (single-person gaze following)
- `head/detection` → `head/social_gaze` (dyadic social gaze, requires ≥ 2 detected heads)

## Introduction

### HeadDetection (YOLO)

Ultralytics YOLO-based head detection module. Detects multiple human heads per frame with per-box confidence scores. Includes a built-in HeadSelection post-package for filtering and sorting detections before gaze estimation.

Module config: [head_detection.yml](seetapsych_gaze_follow/modules/head_detection.yml).

**Packages:**

#### 1. HeadDetection(CoSI)
- Provides Attributes: `head/detection`
- Requires: *(none)*
- Entry: `seetapsych_gaze_follow.head_detection.package.load`
- Available model: `seeta-gaze-follow-yolo_head.pt`
- Parameters:
  - `img_size` (integer, default `640`): Input image size. Affects detection speed and accuracy.
  - `conf` (number, default `0.25`): Detection confidence threshold. Filters low-confidence boxes.
  - `iou` (number, default `0.45`): IoU threshold for NMS deduplication.
  - `max_det` (integer, default `20`): Maximum detection boxes per frame, for multi-person scenarios.

#### 2. HeadSelection
- Provides Attributes: `head/selection`, `head/detection` (overwrites with sorted/filtered list)
- Requires: `head/detection`
- Entry: `seetapsych_gaze_follow.head_selection.package.load`
- Priority: 100 (runs after HeadDetection)
- Parameters:
  - `count` (integer, default `1`): Number of head detections to keep.
  - `method` (selection, default `max_size`): `max_size` or `max_confidence`. Selection criterion for top detections.
  - `sort` (selection, default `left-right`): `left-right`, `right-left`, `top-bottom`, `bottom-top`. Sort order for selected detections.

---

### CoSI — Confidence-Coordinated Spatial Integration Gaze Follow

PyTorch/timm-based CoSI gaze following model. Takes the full image plus each detected head bounding box, and predicts the per-head 2D gaze point on the image plane with an associated attention heatmap. Supports two inference modes: single-person gaze point prediction per head, and dyadic social gaze relation classification between two people (principal ↔ associate). Built on Hydra config with `eval_dyadic` stage and `confidence_coordinated` integration. Input resolution: 448×448.

Module config: [cosi.yml](seetapsych_gaze_follow/modules/cosi.yml).

**Packages:**

#### 1. HeadGazePoint(CoSI)
- Provides Attributes: `head/gaze_point`
- Requires: `head/detection`
- Entry: `seetapsych_gaze_follow.cosi.gaze_point.load`
- Available model: `seeta-gaze-follow-cosi_weights.pth`
- Per-head output fields:
  - `head_location_xyxy`: The source head box used for prediction.
  - `gaze_point_px`: Predicted 2D gaze point (pixel coordinates on the original image).
  - `heatmap`: Floating-point attention heatmap (can be visualized with `cv2.COLORMAP_JET`).

#### 2. HeadSocialRelation(CoSI)
- Provides Attributes: `head/social_gaze`
- Requires: `head/detection` (requires ≥ 2 detected heads)
- Entry: `seetapsych_gaze_follow.cosi.social_gaze.load`
- Available model: `seeta-gaze-follow-cosi_weights.pth` (shared with HeadGazePoint)
- Behaviour: Selects the top-2 highest-confidence detections, orders them horizontally (left = principal, right = associate), and runs dyadic prediction.
- Output fields for each of `principal` and `associate`:
  - `head_location_xyxy`: The source head box.
  - `gaze_point_px`: Predicted gaze point in pixel coordinates.
  - `heatmap`: Per-person attention heatmap.
  - `social_gaze_id`: Integer class ID of the social gaze relation. Ordered mapping: 0=share, 1=mutual, 2=single, 3=miss, 4=void.
  - `social_gaze_label`: Human-readable social gaze relation label. Possible values: share, mutual, single, miss, void. Index of the value matches social_gaze_id.
