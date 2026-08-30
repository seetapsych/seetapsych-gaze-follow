# -*- coding: utf-8 -*-

from typing import Any, cast

import numpy
import torch
from seetapsych_lib import api
from ultralytics import YOLO


class Instance(api.Instance):
    def __init__(
        self,
        weights: str,
        device: api.Device,
        img_size: int | None = None,
        conf: float | None = None,
        iou: float | None = None,
        max_det: int | None = None,
    ):
        if img_size is None:
            img_size = 640
        if conf is None:
            conf = 0.25
        if iou is None:
            iou = 0.45
        if max_det is None:
            max_det = 20

        torch_device = torch.device(str(device))
        model = YOLO(weights, verbose=False)
        model.to(torch_device)
        model_ = cast(Any, model)
        model_.conf = conf
        model_.iou = iou
        model_.max_det = max_det

        self.__torch_device = torch_device
        self.__model = model
        self.__img_size = img_size
        self.__conf = conf
        self.__iou = iou
        self.__max_det = max_det

    def inference(self, *, data: dict[str, Any], report: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        input_data = data["default"]
        input_data = numpy.ascontiguousarray(input_data)  # [H, W, C] format, BGR layout

        image_bgr = input_data
        heads = self.detect_heads(
            image_bgr,
            conf=self.__conf,
            iou=self.__iou,
            max_det=self.__max_det,
            imgsz=self.__img_size,
        )

        return {
            "head_detection": heads,
        }

    def detect_heads(
        self,
        image: numpy.ndarray,
        *,
        conf: float | None = None,
        iou: float | None = None,
        max_det: int | None = None,
        imgsz: int | None = None,
    ) -> list[dict[str, Any]]:
        height, width = image.shape[:2]

        predict_kwargs: dict[str, Any] = {"verbose": False}
        if conf is not None:
            predict_kwargs["conf"] = float(conf)
        if iou is not None:
            predict_kwargs["iou"] = float(iou)
        if max_det is not None:
            predict_kwargs["max_det"] = int(max_det)
        if imgsz is not None:
            predict_kwargs["imgsz"] = int(imgsz)

        results = self.__model(image, **predict_kwargs)
        first = cast(Any, list(results)[0])
        boxes = cast(Any, first.boxes)

        heads: list[dict[str, Any]] = []

        if boxes is None:
            return heads

        for det in cast(Any, list(boxes)):
            x1, y1, x2, y2 = det.xyxy[0].detach().cpu().numpy()
            det_conf = float(det.conf[0])

            heads.append(
                {
                    "xyxy": [
                        int(round(numpy.clip(x1, 0, width - 1))),
                        int(round(numpy.clip(y1, 0, height - 1))),
                        int(round(numpy.clip(x2, 0, width - 1))),
                        int(round(numpy.clip(y2, 0, height - 1))),
                    ],
                    "score": float(det_conf),
                }
            )

        return heads


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

        img_size = parameters.get("img_size", None)
        conf = parameters.get("conf", None)
        iou = parameters.get("iou", None)
        max_det = parameters.get("max_det", None)

        weights = models[0].cache()
        return Instance(
            weights,
            api.Device("cpu") if device is None else device,
            img_size=img_size,
            conf=conf,
            iou=iou,
            max_det=max_det,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == "__main__":
    main()
