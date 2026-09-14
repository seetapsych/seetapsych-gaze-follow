# -*- coding: utf-8 -*-

import os
from typing import Any

import cv2
import numpy
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline
from seetapsych_lib.utils.json import dumps_sanitized

override_modules = [
    os.path.join(os.path.dirname(__file__), "../seetapsych_gaze_follow/modules"),
]

image_path = os.path.join(os.path.dirname(__file__), "test_img.jpg")

COLOR_HEAD: tuple[int, int, int] = (0, 255, 0)
COLOR_SCORE: tuple[int, int, int] = (255, 255, 255)
LABEL_BG_ALPHA: float = 0.45


def fit_image(image: numpy.ndarray, max_width: int = 1280, max_height: int = 960) -> tuple[numpy.ndarray, float]:
    """Proportionally downscale image to fit within max dimensions.

    Args:
        image: Input BGR image.
        max_width: Width cap in pixels.
        max_height: Height cap in pixels.

    Returns:
        (resized_image, scale). ``scale <= 1.0``; 1.0 if no resize.
    """
    h, w = image.shape[:2]
    if w <= max_width and h <= max_height:
        return image, 1.0
    scale = min(max_width / w, max_height / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def put_text_with_shadow(
    image: numpy.ndarray,
    text: str,
    org: tuple[int, int],
    font: int,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    *,
    shadow_color: tuple[int, int, int] = (96, 96, 96),
    shadow_base_offset_px: int = 1,
    line_type: int = cv2.LINE_AA,
) -> None:
    """Draw text with a down-right shadow that scales with ``font_scale``.

    Args:
        image: Target image (in-place).
        text: String to render.
        org: Top-left anchor (x, y) of the primary text.
        font: OpenCV FONT_HERSHEY_* constant.
        font_scale: Font scale (same semantics as ``cv2.putText``).
        color: Primary text BGR color.
        thickness: Stroke thickness applied to both shadow and text.
        shadow_color: Shadow BGR color. Defaults to mid-gray.
        shadow_base_offset_px: Shadow offset at ``font_scale == 1.0``.
        line_type: Line type for ``cv2.putText``. Defaults to anti-aliased.
    """
    offset = int(round(shadow_base_offset_px * max(font_scale, 1.0)))
    x, y = org
    cv2.putText(image, text, (x + offset, y + offset), font, font_scale, shadow_color, thickness, line_type)
    cv2.putText(image, text, org, font, font_scale, color, thickness, line_type)


def draw_label_rect(
    image: numpy.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    *,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    alpha: float = LABEL_BG_ALPHA,
) -> None:
    """Draw a semi-transparent filled rectangle used as a label background.

    Coordinates outside ``image`` are clamped safely, and a zero-area rect
    is silently skipped.

    Args:
        image: Target image (in-place).
        pt1: Top-left corner (x, y).
        pt2: Bottom-right corner (x, y).
        bg_color: Fill BGR color. Defaults to black.
        alpha: Opacity (0 = fully transparent, 1 = fully opaque).
    """
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = image.shape[:2]
    if x2 <= x1 or y2 <= y1 or x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
        return
    rx1 = int(max(0, min(x1, w - 1)))
    ry1 = int(max(0, min(y1, h - 1)))
    rx2 = int(max(1, min(x2, w)))
    ry2 = int(max(1, min(y2, h)))
    overlay = image[ry1:ry2, rx1:rx2]
    if overlay.size == 0:
        return
    filled = numpy.full_like(overlay, bg_color, dtype=numpy.uint8)
    image[ry1:ry2, rx1:rx2] = cv2.addWeighted(overlay, 1.0 - alpha, filled, alpha, 0.0)


def draw_results(image: numpy.ndarray, report: dict[str, Any]) -> numpy.ndarray:
    """Render head detection bboxes with per-box confidence labels.

    Args:
        image: Input BGR image.
        report: Algorithm result dict with ``head_detection`` list, each
            entry contains ``xyxy`` (4 floats) and ``score``.

    Returns:
        Annotated BGR image sized to fit_image limits.
    """
    vis, scale = fit_image(image)

    head_detection = report.get("head_detection", [])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    for _head_idx, bbox in enumerate(head_detection):
        xyxy = [int(round(v * scale)) for v in bbox["xyxy"]]
        pt1, pt2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
        cv2.rectangle(vis, pt1, pt2, COLOR_HEAD, thickness=1, lineType=cv2.LINE_AA)

        score = bbox.get("score", 0.0)
        label = f"{score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_bg_h = text_h + 8
        bbox_gap = 2
        room_above = xyxy[1] - label_bg_h - bbox_gap
        if room_above >= 0:
            label_y1 = room_above
            label_inside = False
        else:
            label_y1 = xyxy[1] + 3
            label_inside = True
        label_y2 = label_y1 + label_bg_h
        label_x1 = xyxy[0] + (0 if not label_inside else 3)
        label_x2 = label_x1 + text_w + 8
        draw_label_rect(vis, (label_x1, label_y1), (label_x2, label_y2))
        put_text_with_shadow(
            vis,
            label,
            (label_x1 + 4, label_y1 + text_h + 4),
            font,
            font_scale,
            COLOR_SCORE,
            thickness,
            shadow_base_offset_px=1,
        )

    return vis


def main():
    factory = Factory()
    # load_dir_modules tolerates missing dirs — example runs fine without local overrides.
    for root in override_modules:
        factory.load_dir_modules(root)

    pipeline = Pipeline(
        factory,
        packages=[
            "b73cc0cf-3300-498f-8795-d1f95a471075",  # HeadDetection-CoSIGaze
        ],
        attributes=[
            # 'head/detection',
        ],
    )

    pipeline.solve()
    pipeline.install_requirements()
    pipeline.cache_models()

    runner = Runner(pipeline)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    report = runner.run(data={"default": image})
    print(dumps_sanitized(report, indent=2, ensure_ascii=False))

    vis = draw_results(image, report)

    stem, ext = os.path.splitext(image_path)
    out_path = f"{stem}_headdet_result{ext}"
    cv2.imwrite(out_path, vis)
    print(f"Saved result to: {out_path}")

    cv2.imshow("head_detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    runner.dispose()


if __name__ == "__main__":
    main()
