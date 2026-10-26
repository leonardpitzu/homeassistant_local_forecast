"""Evaluate the trained forecast models — gradient-boosted trees, pure Python.

Each model is a gzipped binary file in ``models/``: a JSON header with the
input names it expects (see features.py), a link function and the base score,
then the trees.  A tree is six packed little-endian arrays: split feature (-1
on a leaf), left child, right child, whether a missing input goes left, the
threshold and the leaf value.  The training harness exports them from
scikit-learn and checks this exact walk reproduces it.

Loading reads disk, so it runs in an executor once at setup; the arrays are
taken straight from the bytes, so no number is parsed (JSON took 4.6 s for the
same models, this ~0.3 s).  Trees are held in typed arrays (~30 MB for all of
them, as fast to walk as lists), and prediction is a few thousand comparisons
per call with no third-party library.
"""

from __future__ import annotations

from array import array
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
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
        self.trees: list[tuple[array, array, array, array, array, array]] = [
            (array("h", f), array("d", t), array("h", lo), array("h", hi), array("b", m), array("d", v))
            for f, t, lo, hi, m, v in data["trees"]
        ]

    @classmethod
    def from_bytes(cls, blob: bytes) -> Model:
        """Read the packed binary format (see train.write_model in the harness)."""
        if blob[:4] != b"LFM1":
            raise ValueError("not a local_forecast model")
        (n,) = struct.unpack_from("<I", blob, 4)
        head = json.loads(blob[8 : 8 + n])
        pos = 8 + n
        trees = []
        for _ in range(head["trees"]):
            (nodes,) = struct.unpack_from("<H", blob, pos)
            pos += 2
            arrays = []
            for code, size in (("h", 2), ("h", 2), ("h", 2), ("b", 1), ("d", 8), ("d", 8)):
                a = array(code)
                a.frombytes(blob[pos : pos + nodes * size])
                if sys.byteorder != "little":
                    a.byteswap()
                arrays.append(a)
                pos += nodes * size
            feat, left, right, miss, thr, value = arrays
            trees.append((feat, thr, left, right, miss, value))
        model = cls.__new__(cls)
        model.features, model.link, model.base = tuple(head["features"]), head["link"], float(head["base"])
        model.trees = trees
        return model

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

    def __init__(self, hourly: dict[str, Model], daily: dict[str, Model], version: str = "") -> None:
        self.hourly = hourly
        self.daily = daily
        # Fingerprint of the model files: a self-correction learned on other
        # models does not apply to these.
        self.version = version

    @classmethod
    def load(cls, directory: Path = MODELS_DIR) -> ModelSet:
        """Blocking: call from an executor."""
        digest = hashlib.sha256()

        def read(name: str) -> Model:
            blob = gzip.decompress((directory / f"{name}.bin.gz").read_bytes())
            digest.update(blob)
            return Model.from_bytes(blob)

        hourly = {k: read(f"hourly_{k}") for k in HOURLY_TARGETS}
        daily = {k: read(f"daily_{k}") for k in DAILY_TARGETS}
        return cls(hourly, daily, digest.hexdigest()[:12])
