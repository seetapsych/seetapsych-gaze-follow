# -*- coding: utf-8 -*-
import os
from typing import Any

import cv2
import numpy as np
from seetapsych_lib.runtime.factory import Factory

# from seetapsych_lib.runtime.runner import Runner
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

module_root = os.path.join(os.path.dirname(__file__), "../seetapsych_gaze_follow/modules")

PRINCIPAL_COLOR = (0, 255, 0)
HEATMAP_ALPHA = 0.40


def _heatmap_to_colormap(heatmap: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size

    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        hm = np.squeeze(hm)
    if hm.ndim != 2:
        raise ValueError(f"Heatmap must be 2D after squeeze, got shape {hm.shape}")

    if hm.shape != (height, width):
        hm = cv2.resize(hm, (width, height), interpolation=cv2.INTER_LINEAR)

    finite = np.isfinite(hm)
    if not finite.any():
        hm_u8 = np.zeros((height, width), dtype=np.uint8)
    else:
        valid_values = hm[finite]
        hm_min = float(valid_values.min())
        hm_max = float(valid_values.max())

        hm = np.nan_to_num(hm, nan=hm_min, posinf=hm_max, neginf=hm_min)

        if hm_max > hm_min:
            hm = (hm - hm_min) / (hm_max - hm_min)
        else:
            hm = np.zeros_like(hm)

        hm_u8 = np.clip(hm * 255.0, 0, 255).astype(np.uint8)

    return cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)


def _overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, *, alpha: float = HEATMAP_ALPHA) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    heatmap_bgr = _heatmap_to_colormap(heatmap, (width, height))
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)


def _draw_head_gaze(canvas: np.ndarray, head: dict[str, Any], gaze_point: dict[str, Any] | None = None):
    x1, y1, x2, y2 = head["xyxy"]
    score = head["score"]
    height, width = canvas.shape[:2]

    cv2.rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        PRINCIPAL_COLOR,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    label = f"{score:.2f}"
    text_y = max(24, y1 - 10)
    cv2.putText(
        canvas,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        PRINCIPAL_COLOR,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    if gaze_point is not None:
        gx, gy = gaze_point["gaze_point_px"]
        gx = int(round(gx))
        gy = int(round(gy))
        gx = int(max(0, min(gx, width - 1)))
        gy = int(max(0, min(gy, height - 1)))

        head_center = (
            int(round((x1 + x2) / 2.0)),
            int(round((y1 + y2) / 2.0)),
        )

        cv2.line(
            canvas,
            head_center,
            (gx, gy),
            PRINCIPAL_COLOR,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(
            canvas,
            (gx, gy),
            6,
            PRINCIPAL_COLOR,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )


def main():
    factory = Factory()
    factory.load_dir_modules(module_root)

    pipeline = Pipeline(factory, attributes=["head/gaze_point"])

    pipeline.solve()
    pipeline.install_requirements()
    pipeline.cache_models()

    runner = Runner(pipeline)

    image = cv2.imread("test_img.jpg")
    print(image.shape)
    report = runner.run(data={"default": image})
    # print(json.dumps(report, indent=2, ensure_ascii=False))

    heads = report.get("head_detection", [])
    gaze_points = report.get("head_gaze_point", [])

    for i, head in enumerate(heads):
        gp = gaze_points[i] if i < len(gaze_points) else None

        if gp is not None and gp.get("heatmap") is not None:
            canvas = _overlay_heatmap(image.copy(), gp["heatmap"])
        else:
            canvas = image.copy()

        _draw_head_gaze(canvas, head, gp)
        cv2.imshow(f"Head #{i + 1}", canvas)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
