"""Where the framework weights of pretrained models are downloaded to.

'ModelBundle.from_pretrained' downloads the weights of a model's framework into 'models/<framework>' of the
working directory, not into the framework's cache in the home folder. A program that runs from its own root folder,
such as a trainer package, keeps the weights there, so they can be shipped with it and the model is exported
without internet access. Weights that are already there are not downloaded again.
"""

import contextlib
import os
from pathlib import Path
from typing import Iterator

import torch

PATH_MODELS = Path("models")


def framework_weights_dir(framework: str) -> Path:
    return (PATH_MODELS / framework).absolute()


@contextlib.contextmanager
def torchvision_weights_dir() -> Iterator[Path]:
    """Download torchvision weights into 'models/torchvision/checkpoints' (torch hub's layout) inside the context."""
    path_dir = framework_weights_dir("torchvision")
    previous = torch.hub.get_dir()
    torch.hub.set_dir(str(path_dir))
    try:
        yield path_dir
    finally:
        torch.hub.set_dir(previous)


@contextlib.contextmanager
def rfdetr_weights_dir() -> Iterator[Path]:
    """Download RF-DETR weights into 'models/rfdetr' inside the context. RF-DETR reads its cache folder from
    'RF_HOME' when it resolves a weights file name, which covers the fallback of 'RFDETRLarge' to older weights."""
    path_dir = framework_weights_dir("rfdetr")
    previous = os.environ.get("RF_HOME")
    os.environ["RF_HOME"] = str(path_dir)
    try:
        yield path_dir
    finally:
        if previous is None:
            del os.environ["RF_HOME"]
        else:
            os.environ["RF_HOME"] = previous
