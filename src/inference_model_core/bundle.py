"""Reading and writing model bundles.

A model bundle is either a zip file or a directory with a 'model_info.json' at its root and the model
files listed in 'model_info.runtime' ('file_path' and 'extra_files'), e.g.

    model.zip
    ├── model_info.json
    ├── model.xml
    └── model.bin

A zip bundle is a single file, which is how models are passed to the trainer package on the platform
(as a checkpoint). The directory form is useful for local development and serving.
"""

import contextlib
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator, List, Mapping, Union

from inference_model_core.model_info import ModelBundleInfo, check_relative_bundle_path

MODEL_INFO_FILENAME = "model_info.json"

PathLike = Union[str, Path]
FileSource = Union[Path, bytes]  # Path to a local file or the file content


def is_zip_bundle(path_bundle: PathLike) -> bool:
    path_bundle = Path(path_bundle)
    return path_bundle.is_file() and zipfile.is_zipfile(path_bundle)


def read_bundle_file(path_bundle: PathLike, relative_path: str) -> bytes:
    """Read a single file from a zip or directory bundle without extracting the bundle."""
    path_bundle = Path(path_bundle)
    relative_path = check_relative_bundle_path(relative_path)
    if is_zip_bundle(path_bundle):
        with zipfile.ZipFile(path_bundle, "r") as archive:
            if relative_path not in archive.namelist():
                raise FileNotFoundError(f"File '{relative_path}' not found in bundle '{path_bundle}'.")
            return archive.read(relative_path)

    if path_bundle.is_dir():
        path_file = path_bundle / relative_path
        if not path_file.is_file():
            raise FileNotFoundError(f"File '{relative_path}' not found in bundle '{path_bundle}'.")
        return path_file.read_bytes()

    raise FileNotFoundError(f"Model bundle '{path_bundle}' is neither a zip file nor a directory.")


def read_model_info(path_bundle: PathLike) -> ModelBundleInfo:
    """Read only the model info of a bundle. Model files are not touched."""
    return ModelBundleInfo.model_validate_json(read_bundle_file(path_bundle, MODEL_INFO_FILENAME))


def missing_bundle_files(path_bundle: PathLike, model_info: ModelBundleInfo) -> List[str]:
    path_bundle = Path(path_bundle)
    if is_zip_bundle(path_bundle):
        with zipfile.ZipFile(path_bundle, "r") as archive:
            available = set(archive.namelist())
        return [path for path in model_info.runtime.all_files() if path not in available]
    return [path for path in model_info.runtime.all_files() if not (path_bundle / path).is_file()]


def validate_bundle(path_bundle: PathLike) -> ModelBundleInfo:
    """Read the model info and check that all model files it references are present."""
    model_info = read_model_info(path_bundle)
    missing = missing_bundle_files(path_bundle, model_info)
    if missing:
        raise FileNotFoundError(f"Model bundle '{path_bundle}' is missing files referenced in model info: {missing}")
    return model_info


@contextlib.contextmanager
def open_bundle(path_bundle: PathLike) -> Iterator[Path]:
    """Yield a directory holding the bundle files.

    Directory bundles are yielded as-is. Zip bundles are extracted to a temporary directory that is
    deleted when the context exits, so load the model into memory inside the context.
    """
    path_bundle = Path(path_bundle)
    model_info = validate_bundle(path_bundle)
    if path_bundle.is_dir():
        yield path_bundle
        return

    with tempfile.TemporaryDirectory(prefix="model_bundle_") as extract_dir:
        with zipfile.ZipFile(path_bundle, "r") as archive:
            # Only extract the files referenced by the (validated) model info
            for member in [MODEL_INFO_FILENAME, *model_info.runtime.all_files()]:
                archive.extract(member, path=extract_dir)
        yield Path(extract_dir)


def write_bundle(
    path_bundle: PathLike,
    model_info: ModelBundleInfo,
    files: Mapping[str, FileSource],
    overwrite: bool = False,
) -> Path:
    """Write a model bundle. A path ending with '.zip' creates a zip bundle, otherwise a directory.

    'files' maps bundle-relative paths to file sources and must contain exactly the files referenced
    by 'model_info.runtime' - a bundle never holds files its model info doesn't describe.
    """
    path_bundle = Path(path_bundle)
    expected_files = set(model_info.runtime.all_files())
    provided_files = {check_relative_bundle_path(path) for path in files}
    if provided_files != expected_files:
        raise ValueError(
            "Files must match the files referenced in model info. "
            f"Missing: {sorted(expected_files - provided_files)}, "
            f"not referenced: {sorted(provided_files - expected_files)}."
        )

    if path_bundle.exists():
        if not overwrite:
            raise FileExistsError(f"Model bundle '{path_bundle}' already exists. Use 'overwrite=True' to replace it.")
        if path_bundle.is_dir():
            shutil.rmtree(path_bundle)
        else:
            path_bundle.unlink()

    model_info_json = model_info.model_dump_json(indent=4)
    if path_bundle.suffix.lower() == ".zip":
        path_bundle.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path_bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MODEL_INFO_FILENAME, model_info_json)
            for relative_path, source in files.items():
                if isinstance(source, bytes):
                    archive.writestr(relative_path, source)
                else:
                    archive.write(source, arcname=relative_path)
        return path_bundle

    path_bundle.mkdir(parents=True)
    (path_bundle / MODEL_INFO_FILENAME).write_text(model_info_json)
    for relative_path, source in files.items():
        path_file = path_bundle / relative_path
        path_file.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(source, bytes):
            path_file.write_bytes(source)
        else:
            shutil.copy2(source, path_file)
    return path_bundle
