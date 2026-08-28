# -*- coding: utf-8 -*-
import json
import os

import cv2
import numpy as np

from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline
# from seetapsych_lib.runtime.runner import Runner
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner

module_root = os.path.join(os.path.dirname(__file__), '../seetapsych_gaze_follow/modules')

PRINCIPAL_COLOR = (0, 255, 0)
ASSOCIATE_COLOR = (0, 0, 255)
HEATMAP_ALPHA = 0.40


def _heatmap_to_colormap(heatmap, image_size):
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


def _overlay_heatmap(image_bgr, heatmap, *, alpha=HEATMAP_ALPHA):
    height, width = image_bgr.shape[:2]
    heatmap_bgr = _heatmap_to_colormap(heatmap, (width, height))
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)


def _draw_person(canvas, person_name, person_data, color):
    x1, y1, x2, y2 = person_data['head_location_xyxy']
    height, width = canvas.shape[:2]

    gx, gy = person_data['gaze_point_px']
    gx = int(round(gx))
    gy = int(round(gy))
    gx = int(max(0, min(gx, width - 1)))
    gy = int(max(0, min(gy, height - 1)))

    head_center = (
        int(round((x1 + x2) / 2.0)),
        int(round((y1 + y2) / 2.0)),
    )

    cv2.rectangle(
        canvas, (x1, y1), (x2, y2), color,
        thickness=1, lineType=cv2.LINE_AA,
    )

    cv2.line(
        canvas, head_center, (gx, gy), color,
        thickness=1, lineType=cv2.LINE_AA,
    )

    cv2.circle(
        canvas, (gx, gy), 5, color,
        thickness=-1, lineType=cv2.LINE_AA,
    )

    label = f"{person_name}: {person_data['social_gaze_label']}"
    text_y = max(24, y1 - 10)
    cv2.putText(
        canvas, label, (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color,
        thickness=1, lineType=cv2.LINE_AA,
    )


def main():
    factory = Factory()
    factory.load_dir_modules(module_root)

    pipeline = Pipeline(factory, attributes=['head/social_gaze'])

    pipeline.solve()
    pipeline.install_requirements()
    pipeline.cache_models()

    runner = Runner(pipeline)

    image = cv2.imread('test_img.jpg')
    print(image.shape)
    report = runner.run(data={
        'default': image
    })
    # print(json.dumps(report, indent=2, ensure_ascii=False))

    social_gaze = report.get('head_social_gaze', {})
    has_principal = 'principal' in social_gaze and 'associate' in social_gaze

    if not has_principal:
        print('Failed to get social gaze (need at least 2 detected heads)')
        return

    principal = social_gaze['principal']
    associate = social_gaze['associate']

    combined_heatmap = None
    for person in (principal, associate):
        hm = person.get('heatmap')
        if hm is None:
            continue
        if combined_heatmap is None:
            combined_heatmap = np.asarray(hm, dtype=np.float32).copy()
        else:
            combined_heatmap += np.asarray(hm, dtype=np.float32)

    if combined_heatmap is not None:
        canvas = _overlay_heatmap(image.copy(), combined_heatmap)
    else:
        canvas = image.copy()

    _draw_person(canvas, 'principal', principal, PRINCIPAL_COLOR)
    _draw_person(canvas, 'associate', associate, ASSOCIATE_COLOR)

    cv2.imshow('Social Gaze - Combined Heatmap', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
