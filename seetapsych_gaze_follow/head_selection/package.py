# -*- coding: utf-8 -*-

from typing import Any, Literal

from seetapsych_lib import api
from seetapsych_lib.utils.logger import logger

SelectionMethod = Literal["max_size", "max_confidence"]
SortMethod = Literal["left-right", "right-left", "top-bottom", "bottom-top"]


def _box_size(xyxy: list[int]) -> int:
    x1, y1, x2, y2 = xyxy
    return (x2 - x1) * (y2 - y1)


def _box_center(xyxy: list[int]) -> tuple[int, int]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _sort_detections(detections: list[dict], sort_method: SortMethod) -> list[dict]:
    if not detections:
        return detections

    if sort_method == "left-right":
        return sorted(detections, key=lambda d: _box_center(d["xyxy"])[0])
    elif sort_method == "right-left":
        return sorted(detections, key=lambda d: _box_center(d["xyxy"])[0], reverse=True)
    elif sort_method == "top-bottom":
        return sorted(detections, key=lambda d: _box_center(d["xyxy"])[1])
    elif sort_method == "bottom-top":
        return sorted(detections, key=lambda d: _box_center(d["xyxy"])[1], reverse=True)
    else:
        logger.warning("Unknown sort method: %s", sort_method)
        return detections


def _select_top(detections: list[dict], count: int, method: SelectionMethod) -> list[dict]:
    if not detections or count <= 0:
        return []

    if method == "max_size":
        sorted_by = sorted(detections, key=lambda d: _box_size(d["xyxy"]), reverse=True)
    elif method == "max_confidence":
        sorted_by = sorted(detections, key=lambda d: d.get("score", 0.0), reverse=True)
    else:
        logger.warning("Unknown selection method: %s", method)
        sorted_by = detections

    return sorted_by[:count]


class Instance(api.Instance):
    def __init__(self, count: int = 1, method: SelectionMethod = "max_size", sort: SortMethod = "left-right"):
        self.__count = count
        self.__method = method
        self.__sort = sort

    def reset(self):
        pass

    def inference(self, *, data: dict[str, Any], report: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        head_detection = report.get("head_detection", [])

        if not head_detection:
            report["head_selection"] = {
                "count": 0,
                "selected_indices": [],
            }
            return report

        top_selected = _select_top(head_detection, self.__count, self.__method)
        sorted_selected = _sort_detections(top_selected, self.__sort)

        selected_indices = []
        for item in sorted_selected:
            try:
                idx = head_detection.index(item)
                selected_indices.append(idx)
            except ValueError:
                selected_indices.append(-1)

        report["head_selection"] = {
            "count": len(sorted_selected),
            "selected_indices": selected_indices,
        }

        report["head_detection"] = sorted_selected

        return report


class Package(api.Package):
    def create(
        self,
        *,
        models: list[api.UsageModel],
        parameters: dict[str, Any],
        device: api.Device | None,
        **kwargs: Any,
    ) -> Instance:
        count = parameters.get("count", 1)
        method = parameters.get("method", "max_size")
        sort = parameters.get("sort", "left-right")

        return Instance(count=count, method=method, sort=sort)


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == "__main__":
    main()
