# -*- coding: utf-8 -*-

import os
import urllib.request
from typing import Any, Iterator

import cv2
import numpy
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.parallel_utils import stream_ordered
from seetapsych_lib.runtime.pipeline import Pipeline
from tqdm import tqdm

override_modules = [
    os.path.join(os.path.dirname(__file__), "../seetapsych_gaze_follow/modules"),
]

TEST_VIDEO_URL = "https://github.com/user-attachments/assets/89eeee2d-65b6-43c1-88da-f71abc41839f"

video_path = os.path.join(os.path.dirname(__file__), "test_video.mp4")

PRINCIPAL_COLOR: tuple[int, int, int] = (0, 255, 0)
ASSOCIATE_COLOR: tuple[int, int, int] = (0, 0, 255)
COLOR_LABEL_TEXT: tuple[int, int, int] = (255, 255, 255)
HEATMAP_ALPHA: float = 0.85
LABEL_BG_ALPHA: float = 0.45


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


def _draw_person(
    canvas: numpy.ndarray,
    person_name: str,
    person_data: dict[str, Any],
    color: tuple[int, int, int],
) -> None:
    """Draw one person's head box, gaze vector, and labeled chip.

    Args:
        canvas: Target image (in-place), at frame resolution.
        person_name: Display name rendered as the ``{name}: label`` chip.
        person_data: Person entry with ``head_location_xyxy``, ``gaze_point_px``,
            and ``social_gaze_label``.
        color: BGR stroke color for this person.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in person_data["head_location_xyxy"]]
    height, width = canvas.shape[:2]

    gx, gy = [int(round(v)) for v in person_data["gaze_point_px"]]
    gx = int(max(0, min(gx, width - 1)))
    gy = int(max(0, min(gy, height - 1)))

    head_center = (
        int(round((x1 + x2) / 2.0)),
        int(round((y1 + y2) / 2.0)),
    )

    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=1, lineType=cv2.LINE_AA)
    cv2.line(canvas, head_center, (gx, gy), color, thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (gx, gy), 5, color, thickness=-1, lineType=cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    label = person_data["social_gaze_label"]
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    label_bg_h = text_h + 8
    bbox_gap = 2
    room_above = y1 - label_bg_h - bbox_gap
    if room_above >= 0:
        label_y1 = room_above
        label_inside = False
    else:
        label_y1 = y1 + 3
        label_inside = True
    label_y2 = label_y1 + label_bg_h
    label_x1 = x1 + (0 if not label_inside else 3)
    label_x2 = label_x1 + text_w + 8
    draw_label_rect(canvas, (label_x1, label_y1), (label_x2, label_y2))
    put_text_with_shadow(
        canvas,
        label,
        (label_x1 + 4, label_y1 + text_h + 4),
        font,
        font_scale,
        COLOR_LABEL_TEXT,
        thickness,
        shadow_base_offset_px=1,
    )


def draw_results(frame: numpy.ndarray, report: dict[str, Any]) -> numpy.ndarray:
    """Render dyadic social-gaze relation on a single canvas at frame resolution.

    Args:
        frame: Input BGR video frame, at original resolution.
        report: Algorithm result dict with ``head_social_gaze`` containing
            ``principal`` and ``associate`` entries.

    Returns:
        Annotated BGR image at the same resolution as ``frame``.
    """
    social_gaze = report.get("head_social_gaze", {})
    has_pair = "principal" in social_gaze and "associate" in social_gaze

    if not has_pair:
        canvas = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        label = "Social gaze requires 2+ detected heads"
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_bg_h = text_h + 8
        label_x1, label_y1 = 8, 8
        label_x2 = label_x1 + text_w + 16
        label_y2 = label_y1 + label_bg_h + 8
        draw_label_rect(canvas, (label_x1, label_y1), (label_x2, label_y2))
        put_text_with_shadow(
            canvas,
            label,
            (label_x1 + 8, label_y1 + text_h + 12),
            font,
            font_scale,
            (0, 165, 255),
            thickness,
            shadow_base_offset_px=1,
        )
        return canvas

    principal = social_gaze["principal"]
    associate = social_gaze["associate"]

    combined_heatmap: numpy.ndarray | None = None
    for person in (principal, associate):
        hm = person.get("heatmap")
        if hm is None:
            continue
        if combined_heatmap is None:
            combined_heatmap = numpy.asarray(hm, dtype=numpy.float32).copy()
        else:
            combined_heatmap += numpy.asarray(hm, dtype=numpy.float32)

    if combined_heatmap is not None:
        canvas = _overlay_heatmap(frame.copy(), combined_heatmap)
    else:
        canvas = frame.copy()

    _draw_person(canvas, "principal", principal, PRINCIPAL_COLOR)
    _draw_person(canvas, "associate", associate, ASSOCIATE_COLOR)

    return canvas


def _frame_iter(cap: cv2.VideoCapture) -> Iterator[tuple[int, numpy.ndarray]]:
    """Yield ``(index, frame)`` pairs from an open ``VideoCapture``.

    Stops silently on read failure (end of stream or I/O error), matching
    the semantics of a standard ``cap.read()`` loop.

    Args:
        cap: An already-opened ``cv2.VideoCapture``.

    Yields:
        Zero-based index and the decoded BGR frame.
    """
    idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            return
        yield idx, frame
        idx += 1


def _ensure_test_video(path: str, url: str) -> None:
    """Download the bundled example asset for the unmodified demo path.

    Auto-download is guarded by two hard-coded conditions derived from the
    script location itself.  Both must match, otherwise ``path`` is treated
    as a user-owned file and this function is a strict no-op:

    * the base filename must equal ``test_video.mp4``
    * the parent directory must equal the directory containing this script

    Args:
        path: Resolved video path (the script-level ``video_path``).
        url: Remote URL for the bundled example video.
    """
    script_dir = os.path.abspath(os.path.dirname(__file__))
    target_dir = os.path.abspath(os.path.dirname(path))
    if target_dir != script_dir:
        return
    if os.path.basename(path) != "test_video.mp4":
        return
    if os.path.isfile(path):
        return

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if _hook.progress is None:
            _hook.progress = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading test_video.mp4")
        downloaded = block_num * block_size
        _hook.progress.update(min(downloaded - _hook.progress.n, total_size - _hook.progress.n))
        if downloaded >= total_size:
            _hook.progress.close()
            _hook.progress = None

    _hook.progress = None
    try:
        urllib.request.urlretrieve(url, path, reporthook=_hook)
    finally:
        if _hook.progress is not None:
            _hook.progress.close()


def main():
    _ensure_test_video(video_path, TEST_VIDEO_URL)

    factory = Factory()
    for root in override_modules:
        factory.load_dir_modules(root)

    pipeline = Pipeline(
        factory,
        packages=[
            "b73cc0cf-3300-498f-8795-d1f95a471075",  # HeadDetection-CoSIGaze
            "1f77a2b5-9c4a-44c9-98c7-d357caec3178",  # SocialGaze-CoSIGaze
        ],
        attributes=[],
    )

    pipeline.solve()
    pipeline.install_requirements()
    pipeline.cache_models()

    runner = Runner(pipeline)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    src_fps: float | None = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or not numpy.isfinite(src_fps) or src_fps <= 0:
        src_fps = 25.0
    frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames_raw: float | None = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames: int = (
        int(round(total_frames_raw))
        if (total_frames_raw is not None and numpy.isfinite(total_frames_raw) and total_frames_raw > 0)
        else 0
    )

    stem, _ext = os.path.splitext(video_path)
    out_path = f"{stem}_socialgaze_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, src_fps, (frame_width, frame_height))

    written = 0
    progress_total = total_frames if total_frames > 0 else None
    try:
        with tqdm(
            total=progress_total,
            desc="video_social_gaze",
            unit="frame",
            dynamic_ncols=True,
        ) as progress:
            frames = _frame_iter(cap)
            for item in stream_ordered(
                runner,
                frames,
                max_inflight=4,
                input_key="default",
                indexed=True,
                cancel_on_error=True,
                on_early_exit="cancel",
            ):
                vis = draw_results(item.input, item.output)
                writer.write(vis)
                written += 1
                progress.update(1)
                if progress_total is None:
                    progress.total = item.index + 1
    finally:
        cap.release()
        writer.release()
        runner.dispose()

    print(f"Saved video to: {out_path} (fps={src_fps:.3f}, size={frame_width}x{frame_height}, frames={written})")


if __name__ == "__main__":
    main()
