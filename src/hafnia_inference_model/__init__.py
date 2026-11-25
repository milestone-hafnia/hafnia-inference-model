"""Model bundles: write, convert and run any model described as a graph of components.

The user interface is two classes: 'ModelBundle' (a bundle on disk) and 'GraphInferenceModel' (a loaded bundle).
Importing this package imports only 'hafnia_inference_model.core', which needs no torch and no inference engine.
The subpackages:

- 'core': the bundle, component definitions, conversion and the interface of a loaded model graph
- 'torch': the torch graph engine that runs every bundle, and conversion of torch networks to ONNX
- 'onnxruntime' and 'openvino': the inference engines of ONNX and OpenVINO networks, and the OpenVINO converters
- 'exporters': model families, which write bundles from pretrained models and trainer archives
"""

from hafnia_inference_model.core import GraphInferenceModel, ModelBundle

__all__ = ["GraphInferenceModel", "ModelBundle"]
