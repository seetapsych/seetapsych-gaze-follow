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

### Module Catalog

| YAML Path | Packages |
|---|---|
| [head_detection.yml](seetapsych_gaze_follow/modules/head_detection.yml) | HeadDetection-CoSIGaze, HeadSelection |
| [cosi.yml](seetapsych_gaze_follow/modules/cosi.yml) | SceneGazeFollow-CoSIGaze, SocialGaze-CoSIGaze |

## Module Pipeline

The gaze following pipeline consists of two stages loaded from separate module configs:

1. **HeadDetection** ([head_detection.yml](seetapsych_gaze_follow/modules/head_detection.yml)) — multi-head detector, plus an optional HeadSelection post-processor.
2. **CoSI** ([cosi.yml](seetapsych_gaze_follow/modules/cosi.yml)) — confidence-coordinated spatial integration model for gaze point and social gaze relation prediction.

Dependency graph:
- `head/detection` → `head/gaze_point` (single-person gaze following)
- `head/detection` → `head/social_gaze` (dyadic social gaze, requires ≥ 2 detected heads)

### HeadDetection

Ultralytics multi-person head detector with pluggable selection/sorting post-process, used as the front-end for CoSI gaze-following models.

Module config: [head_detection.yml](seetapsych_gaze_follow/modules/head_detection.yml)

| Package Name | Provides Attributes | Requires Attributes |
|---|---|---|
| HeadDetection-CoSIGaze | head/detection | *(none)* |
| HeadSelection | head/selection, head/detection | head/detection |

#### HeadDetection-CoSIGaze

**Description**: multi-person head detector with configurable confidence and NMS thresholds; produces head bounding boxes consumed by CoSI gaze-following and social-gaze packages.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| img_size | integer | 640 | Input image size for inference, larger values improve small-head recall but increase latency and VRAM usage; keep as multiples of 32 (640 is a balanced default). |
| conf | number | 0.25 | Minimum detection confidence threshold, raise to reduce false positives in crowded scenes (0.35-0.5 typical), lower to recover faraway/occluded heads (0.15-0.2). |
| iou | number | 0.45 | NMS IoU threshold for duplicate suppression, lower (0.3-0.4) removes more overlapping boxes for dense crowds, higher (0.5-0.6) keeps more candidates for closely-spaced heads. |
| max_det | integer | 20 | Maximum detection boxes kept per frame after NMS; set to the expected upper bound of simultaneous people in the scene (e.g. 2-4 for dyads, 10-20 for audiences) to avoid noisy downstream cost. |

**Models**

| Name | Recommended |
|---|---|
| seeta-gaze-follow-yolo_head.pt | ✓ |

#### HeadSelection

**Description**: Post-process that selects top-N head detections by size or confidence, then reorders them spatially (left-right/top-bottom) before passing to downstream gaze-following or social-gaze modules.

**Parameters**

| Name | Type | Default | Selection | Description |
|---|---|---|---|---|
| count | integer | 1 | — | Number of heads to keep after selection; match the number of tracked people in the scene, e.g. 1 for single-target, 2 for dyadic social-gaze analysis. |
| method | selection | max_size | max_size, max_confidence | Criterion used to pick the top-N heads. Use max_size to prefer closest/largest heads (dominant foreground person); use max_confidence when occlusion is rare and detector score is trustworthy. |
| sort | selection | left-right | left-right, right-left, top-bottom, bottom-top | Spatial order applied after selection. left-right matches screen reading order and is recommended for dyadic social-gaze (left = principal, right = associate); top-bottom is better for vertically stacked layouts. |

**Models**: *(none)*

### CoSIGaze

Confidence-coordinated Spatial Integration (CoSI) transformer for multi-person gaze following and dyadic social-gaze relation classification from a single RGB scene image.

Module config: [cosi.yml](seetapsych_gaze_follow/modules/cosi.yml)

| Package Name | Provides Attributes | Requires Attributes |
|---|---|---|
| SceneGazeFollow-CoSIGaze | head/gaze_point | head/detection |
| SocialGaze-CoSIGaze | head/social_gaze | head/detection |

#### SceneGazeFollow-CoSIGaze

**Description**: Per-head scene-level gaze-following with CoSI transformer; for every input head box returns a 2D gaze target point (gaze_point_px) and a per-pixel gaze heatmap on the original scene image.

**Parameters**: *(none)*

**Models**

| Name | Recommended |
|---|---|
| seeta-gaze-follow-cosi_weights.pth | ✓ |

#### SocialGaze-CoSIGaze

**Description**: Dyadic social-gaze relation classifier using the shared CoSI transformer backbone; picks the top-2 most confident heads ordered horizontally (left = principal, right = associate) and predicts a 5-class relation, plus per-person gaze point and heatmap.

**Parameters**: *(none)*

**Models**

| Name | Recommended |
|---|---|
| seeta-gaze-follow-cosi_weights.pth | ✓ |

**SocialGaze class mapping** (social_gaze_id → social_gaze_label):

| ID | Label |
|---|---|
| 0 | share |
| 1 | mutual |
| 2 | single |
| 3 | miss |
| 4 | void |
