# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy
import torch
from hydra import compose, initialize_config_dir
from seetapsych_lib import api

from .lib.inference import DyadicGazePredictor
from .lib.models import build_model


def build_gaze_model(args: argparse.Namespace, gaze_device: str) -> tuple[Any, Any]:
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

    torch_device = torch.device(gaze_device)
    model = build_model(torch_device, cfg, verbose=False)
    return model, cfg


def order_two_heads(detections: Sequence[Dict]) -> Tuple[Dict, Dict]:
    """
    1. Keep the two highest-confidence heads.
    2. Order those two by horizontal center.
    3. Left = principal, right = associate.
    """
    if len(detections) < 2:
        raise RuntimeError(
            f"Expected at least 2 detected heads, but YOLOv5 returned {len(detections)}. Try lowering --head-conf."
        )

    best_two = sorted(
        detections,
        key=lambda item: item["score"],
        reverse=True,
    )[:2]

    best_two.sort(key=lambda item: (item["xyxy"][0] + item["xyxy"][2]) / 2.0)

    return best_two[0], best_two[1]


class Instance(api.Instance):
    def __init__(
        self,
        pretrained: str,
        device: api.Device,
    ):
        gaze_device = str(device)
        torch_device = torch.device(gaze_device)

        args = argparse.Namespace(
            config_dir=Path(__file__).parent / "lib" / "config",
            stage="eval_dyadic",
            model="cosi",
            pretrained=pretrained,
            integration="confidence_coordinated",
            device=gaze_device,
        )

        gaze_model, cfg = build_gaze_model(args, gaze_device)
        predictor = DyadicGazePredictor(
            gaze_model,
            input_size=cfg.data.transform.input_resolution,
            device=gaze_device,
        )

        self.__torch_device = torch_device
        self.__predictor = predictor

    def inference(self, *, data: dict[str, Any], report: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        input_data = data["default"]
        input_data = numpy.ascontiguousarray(input_data)  # [H, W, C] format, BGR layout
        image_rgb = input_data[:, :, ::-1]

        head_detection = report.get("head_detection", [])

        if len(head_detection) < 2:
            return {
                "head_social_relation": {
                    "success": False,
                }
            }

        principal_det, associate_det = order_two_heads(head_detection)
        head_boxes = [principal_det["xyxy"], associate_det["xyxy"]]

        prediction = self.__predictor.predict(image_rgb, head_boxes)
        prediction_json = prediction.to_dict(include_heatmaps=False)

        return {
            "head_social_gaze": {
                "principal": {
                    "head_location_xyxy": principal_det["xyxy"],
                    "gaze_point_px": prediction_json["principal"]["gaze_point_px"],
                    "heatmap": prediction.principal.heatmap,
                    "social_gaze_id": int(prediction.principal.social_gaze_id),
                    "social_gaze_label": str(prediction.principal.social_gaze_label),
                },
                "associate": {
                    "head_location_xyxy": associate_det["xyxy"],
                    "gaze_point_px": prediction_json["associate"]["gaze_point_px"],
                    "heatmap": prediction.associate.heatmap,
                    "social_gaze_id": int(prediction.associate.social_gaze_id),
                    "social_gaze_label": str(prediction.associate.social_gaze_label),
                },
            }
        }


class Package(api.Package):
    def create(
        self,
        *,
        models: list[api.UsageModel],
        parameters: dict[str, Any],
        device: api.Device | None,
        **kwargs: Any,
    ) -> Instance:
        assert len(models) >= 1, api.MissingModelError("At least one model required")

        pretrained = models[0].cache()
        return Instance(
            pretrained,
            api.Device("cpu") if device is None else device,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == "__main__":
    main()
