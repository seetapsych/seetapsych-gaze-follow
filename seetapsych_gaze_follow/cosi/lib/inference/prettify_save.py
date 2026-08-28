from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from .preprocess import ImageInput, load_rgb_image
from .types import DyadicPrediction


def save_prediction_visualizations(
    image: ImageInput,
    head_boxes: Sequence[Sequence[float]],
    prediction: DyadicPrediction,
    output_dir,
):
    """
    Save:
      - principal_heatmap.png
      - associate_heatmap.png
      - gaze_points.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(load_rgb_image(image))

    people = (
        ("principal", prediction.principal, head_boxes[0]),
        ("associate", prediction.associate, head_boxes[1]),
    )

    for name, pred, box in people:
        fig, ax = plt.subplots()
        ax.imshow(rgb)
        ax.imshow(pred.heatmap, alpha=0.45)
        x1, y1, x2, y2 = map(float, box)
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2))
        ax.scatter([pred.gaze_point_px[0]], [pred.gaze_point_px[1]], s=60)
        ax.set_title(f"{name}: {pred.social_gaze_label}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}_heatmap.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.imshow(rgb)
    for name, pred, box in people:
        x1, y1, x2, y2 = map(float, box)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        gx, gy = pred.gaze_point_px
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2))
        ax.plot([cx, gx], [cy, gy], linewidth=2)
        ax.scatter([gx], [gy], s=60, label=f"{name}: {pred.social_gaze_label}")
    ax.legend()
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "gaze_points.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
