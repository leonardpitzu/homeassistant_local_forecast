"""Evaluate the trained forecast models — gradient-boosted trees, pure Python.

Each model is a gzipped JSON file in ``models/``: the input names it expects
(see features.py), a link function and a list of trees.  A tree is six flat
arrays: split feature (-1 on a leaf), threshold, left child, right child,
whether a missing input goes left, and the leaf value.  The training harness
exports them from scikit-learn and checks this exact walk reproduces it.

Loading reads disk, so it runs in an executor once at setup.  Prediction is a
few thousand comparisons per call and needs no third-party library.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

MODELS_DIR = Path(__file__).parent / "models"

HOURLY_TARGETS = ("t", "td", "ws", "p", "wet", "rain", "cloud", "fog", "ts")
DAILY_TARGETS = ("tmax", "tmin", "wetday", "rainday", "cloudday", "wsday", "tsday")


class Model:
    """One exported tree ensemble."""

    __slots__ = ("base", "features", "link", "trees")

    def __init__(self, data: dict[str, Any]) -> None:
        self.features: tuple[str, ...] = tuple(data["features"])
        self.link: str = data["link"]
        self.base: float = float(data["base"])
        self.trees: list[tuple[list[int], list[float], list[int], list[int], list[int], list[float]]] = [
            tuple(t) for t in data["trees"]
        ]

    def raw(self, row: list[float]) -> float:
        total = self.base
        for feat, thr, left, right, miss, value in self.trees:
            i = 0
            while (f := feat[i]) >= 0:
                x = row[f]
                if x != x:  # NaN: a channel this station does not have
                    i = left[i] if miss[i] else right[i]
                else:
                    i = left[i] if x <= thr[i] else right[i]
            total += value[i]
        return total

    def predict(self, row: list[float]) -> float:
        raw = self.raw(row)
        if self.link == "logit":
            return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))
        if self.link == "log":
            return math.exp(min(10.0, raw))
        return raw


class ModelSet:
    """Every hourly and daily model, keyed by target name."""

    def __init__(self, hourly: dict[str, Model], daily: dict[str, Model]) -> None:
        self.hourly = hourly
        self.daily = daily

    @classmethod
    def load(cls, directory: Path = MODELS_DIR) -> ModelSet:
        """Blocking: call from an executor."""

        def read(name: str) -> Model:
            with gzip.open(directory / f"{name}.json.gz", "rt", encoding="utf-8") as fp:
                return Model(json.load(fp))

        return cls(
            {k: read(f"hourly_{k}") for k in HOURLY_TARGETS},
            {k: read(f"daily_{k}") for k in DAILY_TARGETS},
        )
