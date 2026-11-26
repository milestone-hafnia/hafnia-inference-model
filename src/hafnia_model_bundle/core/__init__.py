"""Core of the generic inference model concept: the definition of model bundles. The model bundle and its
files, component definitions, bundle conversion and the interface of a loaded model graph.

Nothing here computes anything: the torch graph engine ('hafnia_model_bundle.torch') implements the component
definitions and runs the graph. It is imported when a graph is loaded, so describing, reading and writing a
bundle needs no tensor library (see docs/design.md)."""

from hafnia_model_bundle.core.benchmark import AlignPredictionClasses
from hafnia_model_bundle.core.bundle_io import MODEL_INFO_FILENAME
from hafnia_model_bundle.core.convert import ConversionRequest, list_converters, register_converter
from hafnia_model_bundle.core.definition import ComponentDefinition, Port, Ports
from hafnia_model_bundle.core.graph import LoadSettings, ModelGraph, load_graph
from hafnia_model_bundle.core.inference_model import INPUT_KEY, OUTPUT_KEY, GraphInferenceModel
from hafnia_model_bundle.core.model_bundle import (
    ComponentInfo,
    ConversionInfo,
    GraphInfo,
    ModelBundle,
    NetworkParams,
    TensorInfo,
)
from hafnia_model_bundle.core.transform import ImageTransform

__all__ = [
    "INPUT_KEY",
    "MODEL_INFO_FILENAME",
    "OUTPUT_KEY",
    "AlignPredictionClasses",
    "ComponentDefinition",
    "ComponentInfo",
    "ConversionInfo",
    "ConversionRequest",
    "GraphInferenceModel",
    "GraphInfo",
    "ImageTransform",
    "LoadSettings",
    "ModelBundle",
    "ModelGraph",
    "NetworkParams",
    "Port",
    "Ports",
    "TensorInfo",
    "list_converters",
    "load_graph",
    "register_converter",
]
