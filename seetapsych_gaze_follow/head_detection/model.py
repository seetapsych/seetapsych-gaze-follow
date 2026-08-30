# -*- coding: utf-8 -*-

import os.path

from seetapsych_lib import api

ROOT = os.path.dirname(os.path.abspath(__file__))


class Model(api.Model):
    def __init__(self, name: str):
        self.__name = name
        self.__path = os.path.join(ROOT, name)

    def exists(self) -> bool:
        return os.path.exists(self.__path)

    def cache(self) -> str:
        if self.exists():
            return self.__path

        raise RuntimeError(
            f"Unable to download on my own. Need to contact the developer "
            f"to obtain {self.__name} and place it in the directory {ROOT}"
        )


def load() -> api.Model:
    return Model("seeta-gaze-follow-yolo_head.pt")


def main():
    pass


if __name__ == "__main__":
    main()
