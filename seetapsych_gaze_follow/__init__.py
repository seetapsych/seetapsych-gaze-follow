# -*- coding: utf-8 -*-

from typing import Optional


def _load_version() -> Optional[str]:
    try:
        from ._version import __version__  # type: ignore[import-not-found]
        return __version__.strip() or None
    except Exception:
        return None


__version__ = _load_version() or "0.0.0.dev0"
