import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # Make 'helpers' importable from test modules

from helpers import DummyDetector

from inference_model_core import register_flavor
from inference_model_core.registry import unregister_flavor


@pytest.fixture
def dummy_flavor():
    register_flavor(DummyDetector)
    yield DummyDetector
    unregister_flavor(DummyDetector.flavor_key())
