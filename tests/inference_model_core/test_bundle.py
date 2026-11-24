"""Reading and writing the files of a 'ModelBundle': folders and zips."""

import zipfile

import pytest
from helpers import DUMMY_NETWORK_FILE as MODEL_FILE
from helpers import make_bundle

from inference_model_core import MODEL_INFO_FILENAME, ModelBundle


def test_write_and_read_folder(tmp_path):
    bundle = make_bundle()
    assert bundle.path is None  # Only in memory

    written = bundle.write(tmp_path / "model_dir", files={MODEL_FILE: b"weights"})
    assert written.path == tmp_path / "model_dir"
    assert written.path.is_dir()
    assert bundle.path is None  # The bundle that was written is unchanged

    loaded = ModelBundle.from_path(written.path)
    assert loaded == bundle  # Bundles are compared by their description, not by where their files are
    assert loaded.path == written.path
    assert loaded.read_file(MODEL_FILE) == b"weights"
    with loaded.open_dir() as path_dir:
        assert path_dir == written.path
        assert (path_dir / MODEL_INFO_FILENAME).is_file()


def test_write_and_read_zip(tmp_path):
    bundle = make_bundle()
    written = bundle.write_zip(tmp_path / "model.zip", files={MODEL_FILE: b"weights"})
    assert zipfile.is_zipfile(written.path)

    for loaded in [ModelBundle.from_zip(written.path), ModelBundle.from_path(written.path)]:
        assert loaded == bundle
        assert loaded.read_file(MODEL_FILE) == b"weights"


def test_zip_bundle_is_extracted_to_temporary_dir(tmp_path):
    bundle = make_bundle().write_zip(tmp_path / "model.zip", files={MODEL_FILE: b"weights"})
    with bundle.open_dir() as path_dir:
        assert path_dir != tmp_path
        assert (path_dir / MODEL_FILE).read_bytes() == b"weights"
    assert not path_dir.exists()


def test_from_zip_only_takes_a_zip(tmp_path):
    bundle = make_bundle().write(tmp_path / "model_dir", files={MODEL_FILE: b"weights"})
    with pytest.raises(ValueError, match="is not a zip file"):
        ModelBundle.from_zip(bundle.path)


def test_write_creates_a_folder_and_write_zip_a_zip(tmp_path):
    with pytest.raises(ValueError, match="Use 'write_zip'"):
        make_bundle().write(tmp_path / "model.zip", files={MODEL_FILE: b"weights"})

    written = make_bundle().write_zip(tmp_path / "model_without_suffix", files={MODEL_FILE: b"weights"})
    assert zipfile.is_zipfile(written.path)


def test_write_from_local_file(tmp_path):
    path_source = tmp_path / "source.bin"
    path_source.write_bytes(b"from file")
    bundle = make_bundle().write_zip(tmp_path / "model.zip", files={MODEL_FILE: path_source})
    assert bundle.read_file(MODEL_FILE) == b"from file"


@pytest.mark.parametrize(("source_name", "target_name"), [("a.zip", "b_dir"), ("a_dir", "b.zip"), ("a.zip", "b.zip")])
def test_a_read_bundle_is_written_with_its_files(tmp_path, source_name, target_name):
    def write(bundle, name, **kwargs):
        writer = bundle.write_zip if name.endswith(".zip") else bundle.write
        return writer(tmp_path / name, **kwargs)

    source = ModelBundle.from_path(write(make_bundle(), source_name, files={MODEL_FILE: b"weights"}).path)
    copy = write(source, target_name)  # No 'files': they are taken from where the bundle was read

    assert copy.path == tmp_path / target_name
    assert ModelBundle.from_path(copy.path).read_file(MODEL_FILE) == b"weights"
    assert copy == source


def test_a_bundle_in_memory_needs_its_files(tmp_path):
    bundle = make_bundle()
    with pytest.raises(ValueError, match="is only in memory and has no files on disk"):
        bundle.write(tmp_path / "model_dir")
    with pytest.raises(ValueError, match="is only in memory and has no files on disk"):
        bundle.read_file(MODEL_FILE)
    with pytest.raises(ValueError, match="Write it before converting"):
        bundle.convert("openvino", tmp_path / "converted.zip")


def test_a_bundle_is_not_written_onto_itself(tmp_path):
    bundle = make_bundle().write_zip(tmp_path / "model.zip", files={MODEL_FILE: b"weights"})
    with pytest.raises(ValueError, match="onto itself"):
        bundle.write_zip(tmp_path / "model.zip", overwrite=True)
    assert bundle.read_file(MODEL_FILE) == b"weights"


@pytest.mark.parametrize(
    ("files", "match"),
    [
        ({}, r"Missing: \[.network/model.json.\]"),
        ({MODEL_FILE: b"", "other.bin": b""}, r"not listed: \[.other.bin.\]"),
    ],
)
def test_files_must_match_model_info(tmp_path, files, match):
    with pytest.raises(ValueError, match=match):
        make_bundle().write_zip(tmp_path / "model.zip", files=files)


def test_overwrite(tmp_path):
    path_bundle = tmp_path / "model.zip"
    make_bundle().write_zip(path_bundle, files={MODEL_FILE: b"v1"})
    with pytest.raises(FileExistsError):
        make_bundle().write_zip(path_bundle, files={MODEL_FILE: b"v2"})

    bundle = make_bundle().write_zip(path_bundle, files={MODEL_FILE: b"v2"}, overwrite=True)
    assert bundle.read_file(MODEL_FILE) == b"v2"


def test_missing_model_file_is_detected(tmp_path):
    path_bundle = tmp_path / "broken.zip"
    with zipfile.ZipFile(path_bundle, "w") as archive:
        archive.writestr(MODEL_INFO_FILENAME, make_bundle().model_dump_json())

    with pytest.raises(FileNotFoundError, match=r"missing files of its model info: \[.network/model.json.\]"):
        ModelBundle.from_path(path_bundle)


def test_read_file_rejects_paths_outside_bundle(tmp_path):
    bundle = make_bundle().write(tmp_path / "model_dir", files={MODEL_FILE: b""})
    with pytest.raises(ValueError, match="inside the bundle"):
        bundle.read_file("../secret.txt")


def test_not_a_bundle(tmp_path):
    path_file = tmp_path / "not_a_zip.zip"
    path_file.write_text("plain text")
    with pytest.raises(FileNotFoundError, match="neither a zip file nor a folder"):
        ModelBundle.from_path(path_file)
