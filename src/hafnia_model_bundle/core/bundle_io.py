"""Reading and writing the files of a model bundle. Use 'ModelBundle' ('hafnia_model_bundle.core.model_bundle').

A model bundle is either a zip file or a folder with a 'model_info.json' at its root and the files of the
graph's components, each in the folder of its component, e.g.

    model.zip
    ├── model_info.json
    └── network
        ├── model.xml
        └── model.bin

A zip bundle is a single file, which is how models are passed to the trainer package on the platform
(as a checkpoint). The folder form is useful for local development and serving.

The functions here only know files: what the model info holds is up to 'ModelBundle'.
"""

import contextlib
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterator, List, Mapping, Sequence, Union

MODEL_INFO_FILENAME = "model_info.json"

PathLike = Union[str, Path]
FileSource = Union[Path, bytes]  # Path to a local file or the file content


def check_relative_bundle_path(path: str) -> str:
    """Bundle paths must be relative POSIX paths that stay inside the bundle root."""
    posix_path = PurePosixPath(path)
    if path == "" or "\\" in path or posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"Bundle file paths must be relative paths inside the bundle, got '{path}'.")
    return posix_path.as_posix()


def is_zip_bundle(path_bundle: PathLike) -> bool:
    path_bundle = Path(path_bundle)
    return path_bundle.is_file() and zipfile.is_zipfile(path_bundle)


def has_zip_suffix(path_bundle: PathLike) -> bool:
    return Path(path_bundle).suffix.lower() == ".zip"


def read_bundle_file(path_bundle: PathLike, relative_path: str) -> bytes:
    """Read a single file from a zip or folder bundle without extracting the bundle."""
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

    raise FileNotFoundError(f"Model bundle '{path_bundle}' is neither a zip file nor a folder.")


def missing_bundle_files(path_bundle: PathLike, files: Sequence[str]) -> List[str]:
    path_bundle = Path(path_bundle)
    if is_zip_bundle(path_bundle):
        with zipfile.ZipFile(path_bundle, "r") as archive:
            available = set(archive.namelist())
        return [path for path in files if path not in available]
    return [path for path in files if not (path_bundle / path).is_file()]


@contextlib.contextmanager
def open_bundle_dir(path_bundle: PathLike, files: Sequence[str]) -> Iterator[Path]:
    """Yield a folder holding the model info and 'files' of a bundle.

    Folder bundles are yielded as-is. Zip bundles are extracted to a temporary folder that is deleted when
    the context exits, so load the model into memory inside the context.
    """
    path_bundle = Path(path_bundle)
    if path_bundle.is_dir():
        yield path_bundle
        return

    with tempfile.TemporaryDirectory(prefix="model_bundle_") as extract_dir:
        with zipfile.ZipFile(path_bundle, "r") as archive:
            # Only the files the (validated) model info lists are extracted
            for member in [MODEL_INFO_FILENAME, *files]:
                archive.extract(member, path=extract_dir)
        yield Path(extract_dir)


def write_bundle_files(
    path_bundle: PathLike,
    model_info_json: str,
    files: Mapping[str, FileSource],
    as_zip: bool,
    overwrite: bool = False,
) -> Path:
    """Write the model info and the files of a bundle as a zip or a folder.

    'files' maps bundle-relative paths to file sources.
    """
    path_bundle = Path(path_bundle)
    if path_bundle.exists():
        if not overwrite:
            raise FileExistsError(f"Model bundle '{path_bundle}' already exists. Use 'overwrite=True' to replace it.")
        if path_bundle.is_dir():
            shutil.rmtree(path_bundle)
        else:
            path_bundle.unlink()

    if as_zip:
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
