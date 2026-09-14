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
COLOR_GAZE_LINE: tuple[int, int, int] = (0, 255, 0)
COLOR_GAZE_POINT: tuple[int, int, int] = (0, 255, 0)
COLOR_SCORE: tuple[int, int, int] = (255, 255, 255)
HEATMAP_ALPHA: float = 0.85
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


def _heatmap_to_colormap(heatmap: numpy.ndarray, image_size: tuple[int, int]) -> numpy.ndarray:
    """Normalize a 2D heatmap and render as a JET colormap BGR image.

    Non-finite values are replaced with the finite min before normalization.
    An all-invalid input returns a zero-value colormap.

    Args:
        heatmap: 2D float array.
        image_size: Target (width, height) in pixels.

    Returns:
        BGR uint8 JET colormap of ``image_size``.
    """
    width, height = image_size

    hm: numpy.ndarray = numpy.asarray(heatmap, dtype=numpy.float32)
    if hm.ndim != 2:
        hm = numpy.squeeze(hm)
    if hm.ndim != 2:
        raise ValueError(f"Heatmap must be 2D after squeeze, got shape {hm.shape}")

    if hm.shape != (height, width):
        hm = cv2.resize(hm, (width, height), interpolation=cv2.INTER_LINEAR)

    finite = numpy.isfinite(hm)
    if not finite.any():
        hm_u8 = numpy.zeros((height, width), dtype=numpy.uint8)
    else:
        valid_values = hm[finite]
        hm_min = float(valid_values.min())
        hm_max = float(valid_values.max())

        hm = numpy.nan_to_num(hm, nan=hm_min, posinf=hm_max, neginf=hm_min)

        if hm_max > hm_min:
            hm = (hm - hm_min) / (hm_max - hm_min)
        else:
            hm = numpy.zeros_like(hm)

        hm_u8 = numpy.clip(hm * 255.0, 0, 255).astype(numpy.uint8)

    return cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)


def _overlay_heatmap(
    image_bgr: numpy.ndarray,
    heatmap: numpy.ndarray,
    *,
    alpha: float = HEATMAP_ALPHA,
    gamma: float = 0.45,
) -> numpy.ndarray:
    """Blend a JET-colormap heatmap overlay with intensity-weighted alpha.

    Per-pixel alpha is ``alpha * normalize(heatmap) ** gamma``, so the zero
    tail of the heatmap leaves the base image untouched while the peak
    receives full ``alpha`` blending.

    Args:
        image_bgr: Base BGR image.
        heatmap: 2D heatmap array; resized to ``image_bgr`` shape.
        alpha: Peak-layer opacity (max 1.0).
        gamma: Weight-curve exponent. ``< 1.0`` softens mid-tone falloff,
            ``> 1.0`` concentrates opacity more narrowly around the peak.

    Returns:
        Blended BGR image at the same shape as ``image_bgr``.
    """
    height, width = image_bgr.shape[:2]
    heatmap_bgr = _heatmap_to_colormap(heatmap, (width, height))

    hm: numpy.ndarray = numpy.asarray(heatmap, dtype=numpy.float32)
    if hm.ndim != 2:
        hm = numpy.squeeze(hm)
    if hm.ndim != 2:
        hm = numpy.reshape(hm, (height, width))
    if hm.shape != (height, width):
        hm = cv2.resize(hm, (width, height), interpolation=cv2.INTER_LINEAR)

    finite = numpy.isfinite(hm)
    if finite.any():
        valid_values = hm[finite]
        hm_min = float(valid_values.min())
        hm_max = float(valid_values.max())
        hm = numpy.nan_to_num(hm, nan=hm_min, posinf=hm_max, neginf=hm_min)
        if hm_max > hm_min:
            hm_norm = (hm - hm_min) / (hm_max - hm_min)
        else:
            hm_norm = numpy.zeros_like(hm)
    else:
        hm_norm = numpy.zeros((height, width), dtype=numpy.float32)

    if gamma != 1.0:
        hm_norm = numpy.power(hm_norm.clip(0.0, 1.0), gamma)

    per_pixel_alpha = (alpha * hm_norm).clip(0.0, 1.0).astype(numpy.float32)
    alpha3 = per_pixel_alpha[:, :, numpy.newaxis]

    base = image_bgr.astype(numpy.float32)
    overlay = heatmap_bgr.astype(numpy.float32)
    blended = base * (1.0 - alpha3) + overlay * alpha3
    return numpy.clip(blended, 0, 255).astype(numpy.uint8)


def _draw_single_head(
    image: numpy.ndarray,
    gaze_point: dict[str, Any] | None,
    head_idx: int,
    scale: float,
) -> numpy.ndarray:
    """Render one head panel with bbox, optional heatmap, gaze vector.

    Args:
        image: Original input BGR image, at native resolution.
        gaze_point: Per-head entry with ``head_location_xyxy``,
            ``gaze_point_px``, and optional ``heatmap``. ``None`` indicates
            a placeholder entry where the upstream pipeline produced fewer
            gaze results than heads.
        head_idx: Zero-based index rendered as the ``Head#N`` label.
        scale: fit_image scale factor mapping ``image`` coords to canvas.

    Returns:
        Annotated canvas image (fit_image sized).
    """
    if gaze_point is not None and gaze_point.get("heatmap") is not None:
        canvas_base = _overlay_heatmap(image.copy(), gaze_point["heatmap"])
    else:
        canvas_base = image.copy()

    vis, _ = fit_image(canvas_base)
    fh_vis, fw_vis = vis.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    if gaze_point is not None:
        xyxy = [int(round(v * scale)) for v in gaze_point["head_location_xyxy"]]
    else:
        xyxy = [0, 0, 0, 0]
    pt1, pt2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
    cv2.rectangle(vis, pt1, pt2, COLOR_HEAD, thickness=1, lineType=cv2.LINE_AA)

    label = f"Head#{head_idx}"
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

    if gaze_point is not None:
        gx, gy = gaze_point["gaze_point_px"]
        gx = int(round(gx * scale))
        gy = int(round(gy * scale))
        gx = int(max(0, min(gx, fw_vis - 1)))
        gy = int(max(0, min(gy, fh_vis - 1)))

        head_center = (
            int(round((xyxy[0] + xyxy[2]) / 2.0)),
            int(round((xyxy[1] + xyxy[3]) / 2.0)),
        )

        cv2.line(vis, head_center, (gx, gy), COLOR_GAZE_LINE, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (gx, gy), 5, COLOR_GAZE_POINT, thickness=-1, lineType=cv2.LINE_AA)

    return vis


def _vstack_panels(panels: list[numpy.ndarray]) -> numpy.ndarray:
    """Vertically stack BGR panels, padding widths to match the widest panel.

    Args:
        panels: Non-empty list of BGR uint8 images.

    Raises:
        ValueError: If ``panels`` is empty.

    Returns:
        Single vertically stacked BGR image. Widths are normalized to the
        widest panel via aspect-ratio-preserving resize.
    """
    if not panels:
        raise ValueError("panels must be non-empty")
    if len(panels) == 1:
        return panels[0]

    max_w = max(p.shape[1] for p in panels)
    normalized: list[numpy.ndarray] = []
    for p in panels:
        h, w = p.shape[:2]
        if w == max_w:
            normalized.append(p)
            continue
        new_h = int(round(h * max_w / w))
        resized = cv2.resize(p, (max_w, new_h), interpolation=cv2.INTER_AREA)
        normalized.append(resized)

    total_h = sum(p.shape[0] for p in normalized)
    canvas = numpy.zeros((total_h, max_w, 3), dtype=numpy.uint8)
    y_cursor = 0
    for p in normalized:
        h = p.shape[0]
        canvas[y_cursor : y_cursor + h, : p.shape[1]] = p
        y_cursor += h
    return canvas


def draw_results(image: numpy.ndarray, report: dict[str, Any]) -> numpy.ndarray:
    """Render per-head gaze-following panels stacked vertically.

    Args:
        image: Input BGR image.
        report: Algorithm result dict with ``head_gaze_point`` list.

    Returns:
        Vertically stacked BGR image of N per-head panels.
    """
    _, scale = fit_image(image)

    head_gaze_point = report.get("head_gaze_point", [])

    panels: list[numpy.ndarray] = []
    for head_idx, gp in enumerate(head_gaze_point):
        panel = _draw_single_head(image, gp, head_idx, scale)
        panels.append(panel)

    if not panels:
        vis, _ = fit_image(image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        label = "Gaze point: no head detections"
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_bg_h = text_h + 8
        label_x1, label_y1 = 8, 8
        label_x2 = label_x1 + text_w + 16
        label_y2 = label_y1 + label_bg_h + 8
        draw_label_rect(vis, (label_x1, label_y1), (label_x2, label_y2))
        put_text_with_shadow(
            vis,
            label,
            (label_x1 + 8, label_y1 + text_h + 12),
            font,
            font_scale,
            (0, 165, 255),
            thickness,
            shadow_base_offset_px=1,
        )
        return vis

    return _vstack_panels(panels)


def main():
    factory = Factory()
    # load_dir_modules tolerates missing dirs — example runs fine without local overrides.
    for root in override_modules:
        factory.load_dir_modules(root)

    pipeline = Pipeline(
        factory,
        packages=[
            "b73cc0cf-3300-498f-8795-d1f95a471075",  # HeadDetection-CoSIGaze
            "9cfe3557-1ec4-4802-93bf-a8547e9e1eb8",  # SceneGazeFollow-CoSIGaze
        ],
        attributes=[
            # 'head/detection',
            # 'head/gaze_point',
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
    out_path = f"{stem}_gazepoint_result{ext}"
    cv2.imwrite(out_path, vis)
    print(f"Saved result to: {out_path}")

    cv2.imshow("gaze_point", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    runner.dispose()


if __name__ == "__main__":
    main()
