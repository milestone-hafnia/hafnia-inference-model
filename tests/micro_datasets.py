"""Micro datasets: tiny copies of Hafnia platform datasets, stored with the source code in 'tests/data'.

They let tests and examples run offline and fast on real images and annotations. Each micro dataset is
created from a platform dataset by drawing a fixed number of samples with a fixed seed, the same approach
as the micro datasets in the hafnia repo. Creating one needs platform access (a hafnia profile with a
valid API key). Using one does not.

Create missing micro datasets, or recreate all of them with '--force':

    uv run python tests/micro_datasets.py
    uv run python tests/micro_datasets.py --force
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List, Optional

from cyclopts import App, Parameter
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.log import user_logger

PATH_MICRO_DATASETS = Path(__file__).parent / "data"
METADATA_FILES = ("dataset_info.json", "annotations.parquet")


@dataclass(frozen=True)
class MicroDatasetSpec:
    dataset_name: str  # Platform dataset the samples are drawn from
    version: str
    n_samples: int
    seed: int = 0


MICRO_DATASETS = {
    # 8 of the 20 samples in 'coco-2017-tiny' (all splits)
    "micro-coco-2017": MicroDatasetSpec(dataset_name="coco-2017-tiny", version="1.0.0", n_samples=8),
}


def path_micro_dataset(name: str) -> Path:
    if name not in MICRO_DATASETS:
        raise ValueError(f"Unknown micro dataset '{name}'. Options: {list(MICRO_DATASETS)}")
    return PATH_MICRO_DATASETS / name


def has_micro_dataset(name: str) -> bool:
    path_dataset = path_micro_dataset(name)
    return all((path_dataset / filename).exists() for filename in METADATA_FILES)


def create_micro_dataset(name: str) -> Path:
    """Download the source dataset from the Hafnia platform, draw the samples and write the micro dataset.

    An existing micro dataset is replaced, so no images of an earlier version are left behind.
    """
    spec = MICRO_DATASETS[name]
    path_dataset = path_micro_dataset(name)
    dataset = HafniaDataset.from_name(spec.dataset_name, version=spec.version)
    dataset = dataset.select_samples(n_samples=spec.n_samples, seed=spec.seed, raise_if_too_few_samples=True)

    if path_dataset.exists():
        shutil.rmtree(path_dataset)
    dataset.write(path_dataset)
    user_logger.info(f"Created micro dataset '{name}' with {len(dataset)} samples in '{path_dataset}'")
    return path_dataset


def get_micro_dataset(name: str) -> HafniaDataset:
    """Load a micro dataset, creating it first if it is missing."""
    if not has_micro_dataset(name):
        user_logger.info(f"Micro dataset '{name}' is missing. Creating it from the Hafnia platform.")
        create_micro_dataset(name)
    return HafniaDataset.from_path(path_micro_dataset(name))


app = App(name="micro_datasets", help="Create the micro datasets in 'tests/data' from the Hafnia platform")


@app.default
def main(
    names: Annotated[
        Optional[List[str]], Parameter(help=f"Micro datasets. Default: all {list(MICRO_DATASETS)}")
    ] = None,
    force: Annotated[bool, Parameter(help="Recreate micro datasets that already exist")] = False,
):
    for name in names or list(MICRO_DATASETS):
        if has_micro_dataset(name) and not force:
            user_logger.info(f"Micro dataset '{name}' exists. Use '--force' to recreate it.")
            continue
        create_micro_dataset(name)


if __name__ == "__main__":
    app()
