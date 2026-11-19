import zipfile

import pytest
from helpers import make_model_info

from inference_model_core import (
    MODEL_INFO_FILENAME,
    open_bundle,
    read_bundle_file,
    read_model_info,
    validate_bundle,
    write_bundle,
)


@pytest.mark.parametrize("bundle_name", ["model.zip", "model_dir"])
def test_write_and_read_bundle(tmp_path, bundle_name):
    model_info = make_model_info()
    path_bundle = write_bundle(tmp_path / bundle_name, model_info, files={"model.json": b"weights"})

    assert read_model_info(path_bundle) == model_info
    assert validate_bundle(path_bundle) == model_info
    assert read_bundle_file(path_bundle, "model.json") == b"weights"
    with open_bundle(path_bundle) as path_dir:
        assert (path_dir / "model.json").read_bytes() == b"weights"
        assert (path_dir / MODEL_INFO_FILENAME).is_file()


def test_write_bundle_from_local_file(tmp_path):
    path_source = tmp_path / "source.bin"
    path_source.write_bytes(b"from file")
    path_bundle = write_bundle(tmp_path / "model.zip", make_model_info(), files={"model.json": path_source})
    assert read_bundle_file(path_bundle, "model.json") == b"from file"


def test_zip_bundle_is_extracted_to_temporary_dir(tmp_path):
    path_bundle = write_bundle(tmp_path / "model.zip", make_model_info(), files={"model.json": b"weights"})
    with open_bundle(path_bundle) as path_dir:
        assert path_dir != tmp_path
    assert not path_dir.exists()


@pytest.mark.parametrize(
    ("files", "match"),
    [({}, r"Missing: \[.model.json.\]"), ({"model.json": b"", "other.bin": b""}, r"not referenced: \[.other.bin.\]")],
)
def test_files_must_match_model_info(tmp_path, files, match):
    with pytest.raises(ValueError, match=match):
        write_bundle(tmp_path / "model.zip", make_model_info(), files=files)


def test_overwrite(tmp_path):
    path_bundle = tmp_path / "model.zip"
    write_bundle(path_bundle, make_model_info(), files={"model.json": b"v1"})
    with pytest.raises(FileExistsError):
        write_bundle(path_bundle, make_model_info(), files={"model.json": b"v2"})

    write_bundle(path_bundle, make_model_info(), files={"model.json": b"v2"}, overwrite=True)
    assert read_bundle_file(path_bundle, "model.json") == b"v2"


def test_missing_model_file_is_detected(tmp_path):
    path_bundle = tmp_path / "broken.zip"
    with zipfile.ZipFile(path_bundle, "w") as archive:
        archive.writestr(MODEL_INFO_FILENAME, make_model_info().model_dump_json())

    with pytest.raises(FileNotFoundError, match=r"missing files referenced in model info: \[.model.json.\]"):
        validate_bundle(path_bundle)


def test_read_bundle_file_rejects_paths_outside_bundle(tmp_path):
    path_bundle = write_bundle(tmp_path / "model_dir", make_model_info(), files={"model.json": b""})
    with pytest.raises(ValueError, match="inside the bundle"):
        read_bundle_file(path_bundle, "../secret.txt")


def test_not_a_bundle(tmp_path):
    path_file = tmp_path / "not_a_zip.zip"
    path_file.write_text("plain text")
    with pytest.raises(FileNotFoundError, match="neither a zip file nor a directory"):
        read_model_info(path_file)
