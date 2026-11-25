import pytest

from hafnia_inference_model.exporters.class_names import detection_task, placeholder_class_name


def test_placeholders_make_class_names_unique():
    names = [placeholder_class_name(0), "person", placeholder_class_name(2)]
    assert names == ["NotDefined_000", "person", "NotDefined_002"]
    assert detection_task(names, "object_detection").get_class_names() == names
    with pytest.raises(ValueError, match="unique"):
        detection_task(["N/A", "person", "N/A"], "object_detection")
