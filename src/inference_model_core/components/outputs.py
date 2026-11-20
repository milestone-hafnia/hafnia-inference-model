"""Outputs shared by components of several modules."""

from typing import List, NamedTuple

from hafnia.dataset.primitives import Primitive


class Predictions(NamedTuple):
    predictions: List[Primitive]
