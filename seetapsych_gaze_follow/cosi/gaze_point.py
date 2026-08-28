# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import Any

import numpy
import torch
from hydra import compose, initialize_config_dir

from seetapsych_lib import api

from .lib.models import build_model
from .lib.inference.predictor import SinglePersonGazePredictor

def build_gaze_model(args, gaze_device):
    config_dir = os.path.abspath(args.config_dir)
    overrides = [
        f"stage={args.stage}",
        f"model={args.model}",
        f"pretrained_weights={args.pretrained}",
    ]

    if args.model == "cosi":
        overrides.extend(
            [
                f"model.integration={args.integration}",
                "data.transform.input_resolution=448",
            ]
        )

    if args.device is not None:
        overrides.append(f"device={args.device}")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    model = build_model(gaze_device, cfg, verbose=False)
    return model, cfg

class Instance(api.Instance):
    def __init__(
            self, pretrained: str, device: api.Device,
        ):
        gaze_device = str(device)
        torch_device = torch.device(gaze_device)

        args = argparse.Namespace(
            config_dir=Path(__file__).parent / 'lib' / 'config',
            stage="eval_dyadic",
            model="cosi",
            pretrained=pretrained,
            integration="confidence_coordinated",
            device=gaze_device,
        )

        gaze_model, cfg = build_gaze_model(args, gaze_device)
        predictor = SinglePersonGazePredictor(
            gaze_model,
            input_size=cfg.data.transform.input_resolution,
            device=gaze_device,
            person_key="principal",
        )

        self.__torch_device = torch_device
        self.__predictor = predictor

    def inference(self, *,
                  data: dict[str, Any],
                  report: dict[str, Any],
                  **kwargs) -> dict[str, Any]:
        input_data = data['default']
        input_data = numpy.ascontiguousarray(input_data)  # [H, W, C] format, BGR layout
        image_rgb = input_data[:, :, ::-1]

        head_detection = report.get('head_detection', [])

        head_gaze_point = []
        for bbox in head_detection:
            xyxy = bbox['xyxy'] # [x1, y1, x2, y2]
            prediction = self.__predictor.predict(image_rgb, xyxy)
            prediction_json = prediction.to_dict(include_heatmap=False)
            head_gaze_point.append({
                'head_location_xyxy': xyxy,
                'gaze_point_px': prediction_json['gaze_point_px'],
                'heatmap': prediction.heatmap,
            })
        return {
            'head_gaze_point': head_gaze_point,
        }

    def detect_heads(self, image, *, conf=None, iou=None, max_det=None, imgsz=None):
        height, width = image.shape[:2]

        predict_kwargs = {"verbose": False}
        if conf is not None:
            predict_kwargs["conf"] = float(conf)
        if iou is not None:
            predict_kwargs["iou"] = float(iou)
        if max_det is not None:
            predict_kwargs["max_det"] = int(max_det)
        if imgsz is not None:
            predict_kwargs["imgsz"] = int(imgsz)

        results = self.__model(image, **predict_kwargs)
        boxes = results[0].boxes

        heads = []

        for det in boxes:
            x1, y1, x2, y2 = det.xyxy[0].detach().cpu().numpy()
            conf = float(det.conf[0])

            heads.append({
                "xyxy": [
                    int(round(numpy.clip(x1, 0, width - 1))),
                    int(round(numpy.clip(y1, 0, height - 1))),
                    int(round(numpy.clip(x2, 0, width - 1))),
                    int(round(numpy.clip(y2, 0, height - 1)))
                ],
                "score": float(conf),
            })

        return heads


class Package(api.Package):
    def create(self, *,
               models: list[api.UsageModel],
               parameters: dict[str, Any],
               device: api.Device | None,
               **kwargs) -> Instance:
        assert len(models) >= 1, api.MissingModelError('At least one model required')

        pretrained = models[0].cache()
        return Instance(
            pretrained,
            api.Device('cpu') if device is None else device,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == '__main__':
    main()
