"""Self-contained, aspect-preserving OpenCV window support for camera demos."""

import ctypes
import sys

import cv2
import numpy


def desktop_bounds() -> tuple[int, int, int, int]:
    """Find the primary desktop work area without changing process DPI settings.

    Returns:
        Left, top, width, and height in desktop coordinates. Headless systems
        or unavailable desktop APIs fall back to a conservative 1280 by 720 area.
    """
    if sys.platform == "win32":
        from ctypes import wintypes

        rect = wintypes.RECT()
        spi_get_work_area = 0x0030
        if ctypes.windll.user32.SystemParametersInfoW(spi_get_work_area, 0, ctypes.byref(rect), 0):
            return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    else:
        try:
            import tkinter
        except ImportError:
            return 0, 0, 1280, 720
        try:
            root = tkinter.Tk()
            root.withdraw()
            try:
                return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()
            finally:
                root.destroy()
        except tkinter.TclError:
            pass
    return 0, 0, 1280, 720


def initial_window_size(content_size: tuple[int, int], desktop_size: tuple[int, int]) -> tuple[int, int]:
    """Choose a substantial initial viewport that leaves room for desktop chrome.

    Args:
        content_size: Preferred content width and height.
        desktop_size: Available desktop width and height.

    Returns:
        Aspect-preserving viewport size, bounded by the desktop even for 4K cameras.

    Raises:
        ValueError: If any dimension is not positive.
    """
    if min(*content_size, *desktop_size) <= 0:
        raise ValueError("Content and desktop dimensions must be positive.")
    available = (max(1, int(desktop_size[0] * 0.85) - 32), max(1, int(desktop_size[1] * 0.85) - 64))
    width, height = content_size
    scale = min(available[0] / width, available[1] / height)
    return max(1, int(width * scale)), max(1, int(height * scale))


def fit_geometry(source_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Compute centered, aspect-preserving display bounds.

    Args:
        source_size: Source width and height.
        target_size: Viewport width and height.

    Returns:
        Left, top, resized width, and resized height within the viewport.

    Raises:
        ValueError: If any dimension is not positive.
    """
    if min(*source_size, *target_size) <= 0:
        raise ValueError("Source and target dimensions must be positive.")
    width, height = source_size
    tw, th = target_size
    scale = min(tw / width, th / height)
    rw, rh = max(1, min(tw, int(round(width * scale)))), max(1, min(th, int(round(height * scale))))
    return (tw - rw) // 2, (th - rh) // 2, rw, rh


def fit_canvas(image: numpy.ndarray, size: tuple[int, int]) -> numpy.ndarray:
    """Fit an entire BGR image without distortion or cropping.

    Args:
        image: Source BGR image.
        size: Target width and height.

    Returns:
        Target-sized image with dark letterboxing. An exact-size input is
        returned unchanged; shrinking uses area filtering and enlargement uses cubic.
    """
    height, width = image.shape[:2]
    if (width, height) == size:
        return image
    x, y, rw, rh = fit_geometry((width, height), size)
    interpolation = cv2.INTER_AREA if rw < width or rh < height else cv2.INTER_CUBIC
    resized = image if (rw, rh) == (width, height) else cv2.resize(image, (rw, rh), interpolation=interpolation)
    canvas = numpy.full((size[1], size[0], 3), (28, 25, 23), dtype=numpy.uint8)
    canvas[y : y + rh, x : x + rw] = resized
    return canvas


def draw_text(
    image: numpy.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int] = (235, 235, 235),
    *,
    scale: float = 0.6,
    thickness: int = 1,
) -> None:
    """Draw native-resolution text, shrinking long lines to the available width.

    Args:
        image: Destination BGR image.
        text: Display text.
        org: Left anchor and baseline in viewport pixels.
        color: BGR text color.
        scale: Desired OpenCV font scale.
        thickness: Desired stroke width.
    """
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    scale *= min(1.0, max(1, image.shape[1] - org[0] - 12) / max(1, width))
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class CameraWindow:
    """Manage default, resized, and fullscreen viewports without stretching content."""

    def __init__(self, name: str, content_size: tuple[int, int], *, fullscreen: bool = False) -> None:
        """Create a centered, screen-bounded window.

        Args:
            name: HighGUI window name.
            content_size: Preferred content width and height.
            fullscreen: Whether to start fullscreen.
        """
        self.name = name
        self.fullscreen = False
        left, top, width, height = desktop_bounds()
        self.windowed_size = initial_window_size(content_size, (width, height))
        # Explicit letterboxing avoids backend-specific WINDOW_KEEPRATIO behavior.
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(name, *self.windowed_size)
        cv2.moveWindow(name, left + (width - self.windowed_size[0]) // 2, top + (height - self.windowed_size[1]) // 2)
        if fullscreen:
            self.toggle_fullscreen()

    def size(self) -> tuple[int, int]:
        """Read the current drawable area, including maximized and fullscreen states.

        Returns:
            Viewport width and height, or the initial size before HighGUI lays it out.
        """
        try:
            _, _, width, height = cv2.getWindowImageRect(self.name)
            if width > 0 and height > 0:
                return width, height
        except cv2.error:
            pass
        return self.windowed_size

    def toggle_fullscreen(self) -> None:
        """Toggle fullscreen, preserving the previous windowed viewport size."""
        if not self.fullscreen:
            self.windowed_size = self.size()
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(
            self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        )
        if not self.fullscreen:
            cv2.resizeWindow(self.name, *self.windowed_size)

    def handle_key(self, key: int) -> None:
        """Handle the shared fullscreen shortcut.

        Args:
            key: Low eight bits of the HighGUI key code.
        """
        if key in (ord("f"), ord("F")):
            self.toggle_fullscreen()

    def show(self, image: numpy.ndarray) -> None:
        """Display an image in the current viewport without changing its aspect ratio.

        Args:
            image: BGR image or native-resolution dashboard.
        """
        cv2.imshow(self.name, fit_canvas(image, self.size()))
