# -*- coding: utf-8 -*-
"""Capture mirrored scene-gaze snapshots without pausing the live camera preview."""

import os
import platform
import time
from typing import Any

import cv2
import numpy
from camera_window import CameraWindow, draw_text, fit_canvas, fit_geometry
from image_gaze_point import _draw_single_head, _overlay_heatmap, fit_image
from image_gaze_point import draw_results as draw_gaze_results
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.parallel.future import Future
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

override_modules = [
    os.path.join(os.path.dirname(__file__), "../seetapsych_gaze_follow/modules"),
]


def open_camera(camera_id: int = 0) -> cv2.VideoCapture:
    """Open a platform camera backend, falling back to automatic selection.

    Args:
        camera_id: Camera device index.

    Returns:
        Capture whose open state must be checked by the caller.
    """
    backend = {
        "Windows": cv2.CAP_DSHOW,
        "Linux": cv2.CAP_V4L2,
        "Darwin": cv2.CAP_AVFOUNDATION,
    }.get(platform.system(), cv2.CAP_ANY)
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id)
    return cap


def fit_panel(image: numpy.ndarray) -> numpy.ndarray:
    """Letterbox a scene image into a fixed-size display panel.

    Args:
        image: BGR scene or annotated snapshot.

    Returns:
        A 640 by 480 BGR panel.
    """
    return fit_canvas(image, (640, 480))


def render_snapshot(
    snapshot: numpy.ndarray, report: dict[str, Any], head_index: int, *, native: bool = False
) -> numpy.ndarray:
    """Reuse image-example heatmap styling for one selected snapshot head.

    Args:
        snapshot: Exact mirrored frame submitted for inference.
        report: Completed scene-gaze report.
        head_index: Selected head index; wrapped to the available head count.
        native: Preserve source resolution and defer vector/text rendering to the viewport.

    Returns:
        Letterboxed annotated snapshot, or a native-resolution heatmap image.
    """
    heads = report.get("head_gaze_point", [])
    if native:
        heatmap = heads[head_index % len(heads)].get("heatmap") if heads else None
        return _overlay_heatmap(snapshot, heatmap) if heatmap is not None else snapshot.copy()
    if not heads:
        return fit_panel(draw_gaze_results(snapshot, {"head_gaze_point": []}))
    head_index %= len(heads)
    _, scale = fit_image(snapshot)
    return fit_panel(_draw_single_head(snapshot, heads[head_index], head_index, scale))


def draw_processing(panel: numpy.ndarray, elapsed_seconds: float) -> None:
    """Draw a rotating arc and elapsed time over a pending snapshot.

    The animation indicates a pending request, not measured algorithm progress.

    Args:
        panel: Snapshot display region, modified in place without changing its source.
        elapsed_seconds: Monotonic time since submission; also drives the animation.
    """
    height, width = panel.shape[:2]
    scale = min(width / 640, height / 480)
    card_width = max(1, round(360 * scale))
    card_height = max(1, round(100 * scale))
    left, top = (width - card_width) // 2, (height - card_height) // 2
    card = panel[top : top + card_height, left : left + card_width]
    background = numpy.full_like(card, (28, 25, 23))
    card[:] = cv2.addWeighted(card, 0.65, background, 0.35, 0)
    center = (round(34 * scale), round(50 * scale))
    radius = max(1, round(18 * scale))
    thickness = max(1, round(3 * scale))
    angle = (max(0.0, elapsed_seconds) * 300) % 360
    cv2.circle(card, center, radius, (85, 80, 75), thickness, cv2.LINE_AA)
    cv2.ellipse(card, center, (radius, radius), angle, 0, 260, (100, 220, 130), thickness, cv2.LINE_AA)
    draw_text(card, "Processing snapshot", (round(70 * scale), round(40 * scale)), scale=0.6 * scale)
    draw_text(
        card,
        f"Elapsed {elapsed_seconds:.1f}s",
        (round(70 * scale), round(70 * scale)),
        (100, 220, 130),
        scale=0.55 * scale,
    )


def compose_view(
    frame: numpy.ndarray,
    result_panel: numpy.ndarray,
    status: str,
    display_size: tuple[int, int] = (1280, 580),
    gaze_point: dict[str, Any] | None = None,
    head_index: int = 0,
    *,
    processing_seconds: float | None = None,
) -> numpy.ndarray:
    """Compose a live scene and frozen inference result without mixing coordinates.

    Args:
        frame: Current live frame, already mirrored before capture and inference.
        result_panel: Cached mirrored snapshot visualization, preferably at source resolution.
        status: Capture or inference status.
        display_size: Current viewport width and height.
        gaze_point: Optional mirrored-source result for native-resolution annotation.
        head_index: Displayed head index.
        processing_seconds: Elapsed pending-request time, or None to hide the loading overlay.

    Returns:
        Side-by-side OpenCV dashboard.
    """
    width, height = display_size
    ui_scale = min(width / 1280, height / 580)
    header, footer = round(44 * ui_scale), round(56 * ui_scale)
    panel_width, panel_height = width // 2, max(1, height - header - footer)
    canvas = numpy.full((height, width, 3), (28, 25, 23), dtype=numpy.uint8)
    canvas[header : header + panel_height, :panel_width] = fit_canvas(frame, (panel_width, panel_height))
    canvas[header : header + panel_height, panel_width:] = fit_canvas(result_panel, (width - panel_width, panel_height))
    if gaze_point is not None:
        source_height, source_width = result_panel.shape[:2]
        ox, oy, rw, rh = fit_geometry((source_width, source_height), (width - panel_width, panel_height))
        sx, sy = rw / source_width, rh / source_height
        x1, y1, x2, y2 = gaze_point["head_location_xyxy"]
        pt1 = (panel_width + ox + round(x1 * sx), header + oy + round(y1 * sy))
        pt2 = (panel_width + ox + round(x2 * sx), header + oy + round(y2 * sy))
        gx, gy = gaze_point["gaze_point_px"]
        point = (
            panel_width + ox + round(numpy.clip(gx, 0, source_width - 1) * sx),
            header + oy + round(numpy.clip(gy, 0, source_height - 1) * sy),
        )
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(canvas, pt1, pt2, (0, 255, 0), max(1, round(ui_scale)), cv2.LINE_AA)
        cv2.line(canvas, center, point, (0, 255, 0), max(1, round(ui_scale)), cv2.LINE_AA)
        cv2.circle(canvas, point, max(2, round(5 * ui_scale)), (0, 255, 0), -1, cv2.LINE_AA)
        draw_text(
            canvas,
            f"Head#{head_index}",
            (pt1[0], max(header + round(20 * ui_scale), pt1[1] - 8)),
            scale=0.6 * ui_scale,
        )
    if processing_seconds is not None:
        draw_processing(canvas[header : header + panel_height, panel_width:], processing_seconds)
    for text, point, color in (
        ("LIVE SCENE", (round(16 * ui_scale), round(29 * ui_scale)), (100, 220, 130)),
        ("CAPTURED GAZE POINT", (panel_width + round(16 * ui_scale), round(29 * ui_scale)), (100, 220, 130)),
        (status, (round(16 * ui_scale), height - round(32 * ui_scale)), (235, 235, 235)),
        (
            "F = Fullscreen | Space = Capture | N = Next head | Esc / Q / X = Exit",
            (round(16 * ui_scale), height - round(6 * ui_scale)),
            (235, 235, 235),
        ),
    ):
        draw_text(canvas, text, point, color, scale=0.6 * ui_scale, thickness=max(1, round(ui_scale)))
    return canvas


def run_camera(cap: cv2.VideoCapture) -> None:
    """Submit snapshots asynchronously, allowing only one capture in flight.

    Frames are mirrored before display and inference so heatmaps and gaze
    coordinates need no additional reflection. Text is drawn after mirroring.

    Args:
        cap: Open camera owned by the caller.
    """
    factory = Factory()
    for root in override_modules:
        factory.load_dir_modules(root)
    pipeline = Pipeline(
        factory,
        packages=[
            "b73cc0cf-3300-498f-8795-d1f95a471075",  # HeadDetection-CoSIGaze
            "9cfe3557-1ec4-4802-93bf-a8547e9e1eb8",  # SceneGazeFollow-CoSIGaze
        ],
        attributes=[],
    )
    pipeline.solve()
    pipeline.install_requirements()
    pipeline.cache_models()
    runner = Runner(pipeline)
    pending: Future[dict[str, Any]] | None = None
    snapshot: numpy.ndarray | None = None
    report: dict[str, Any] = {}
    head_index = 0
    submitted_at = 0.0
    elapsed = 0.0
    result_panel: numpy.ndarray = numpy.full((480, 640, 3), (28, 25, 23), dtype=numpy.uint8)
    status = "Space captures a scene. Results stay attached to that snapshot."
    window = "Camera Gaze Point"
    try:
        display = CameraWindow(window, (1280, 580))
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped delivering frames. Check the connection and camera permissions.")
                break
            frame = cv2.flip(frame, 1)
            if pending is not None and pending.done():
                report = pending.get()
                pending = None
                elapsed = time.monotonic() - submitted_at
                if snapshot is not None:
                    result_panel = render_snapshot(snapshot, report, head_index, native=True)
                count = len(report.get("head_gaze_point", []))
                status = (
                    f"Head {head_index + 1}/{count} | {elapsed:.2f}s | N changes the heatmap"
                    if count
                    else ("No heads detected in this snapshot. Capture again with a head visible.")
                )
            heads = report.get("head_gaze_point", [])
            selected_head = heads[head_index % len(heads)] if heads and pending is None else None
            processing_seconds = max(0.0, time.monotonic() - submitted_at) if pending is not None else None
            display.show(
                compose_view(
                    frame,
                    result_panel,
                    status,
                    display.size(),
                    selected_head,
                    head_index,
                    processing_seconds=processing_seconds,
                )
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q"), ord("x"), ord("X")):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            display.handle_key(key)
            if key == ord(" "):
                if pending is None:
                    snapshot = frame.copy()
                    report = {}
                    head_index = 0
                    result_panel = snapshot.copy()
                    submitted_at = time.monotonic()
                    # Snapshot geometry must not be reused as an overlay on subsequent live frames.
                    pending = runner.run_async(data={"default": snapshot.copy()}, timestamp=time.time())
                    status = "Processing snapshot... live camera continues."
                else:
                    status = "Still processing this snapshot; capture again when it finishes."
            if key in (ord("n"), ord("N")) and pending is None and snapshot is not None:
                count = len(report.get("head_gaze_point", []))
                if count:
                    head_index = (head_index + 1) % count
                    result_panel = render_snapshot(snapshot, report, head_index, native=True)
                    status = f"Head {head_index + 1}/{count} | {elapsed:.2f}s | N changes the heatmap"
    finally:
        if pending is not None:
            pending.cancel()
        runner.dispose()


def main() -> None:
    """Open the camera before preparing models and release resources on exit."""
    cap = open_camera(0)
    try:
        if not cap.isOpened():
            print("Could not open camera 0. Check the device index, camera permissions, and other camera apps.")
            return
        run_camera(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
