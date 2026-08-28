# -*- coding: utf-8 -*-
import json
import os

import cv2

from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline
from seetapsych_lib.runtime.runner import Runner

module_root = os.path.join(os.path.dirname(__file__), '../seetapsych_gaze_follow/modules')


def main():
    factory = Factory()
    factory.load_dir_modules(module_root)

    pipeline = Pipeline(factory, attributes=['head/detection'])

    # print(pipeline.config.model_dump_json(indent=2, exclude_none=True))
    pipeline.solve()

    # print(pipeline.config.model_dump_json(indent=2, exclude_none=True))
    # print(pipeline.problem())
    # print(pipeline.satisfied())
    pipeline.install_requirements()
    pipeline.cache_models()

    runner = Runner(pipeline)

    image = cv2.imread('test_img.jpg')
    print(image.shape)
    report = runner.run(data={
        'default': image
    })
    print(json.dumps(report, indent=2, ensure_ascii=False))

    vis_image = image.copy()
    heads = report.get('head_detection', [])
    for head in heads:
        x1, y1, x2, y2 = head['xyxy']
        score = head['score']
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis_image, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(vis_image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('Head Detection', vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
