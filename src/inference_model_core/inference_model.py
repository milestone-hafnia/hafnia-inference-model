"""Base class for all inference model flavors.

A flavor is an 'InferenceModelBase' subclass for one (inference engine, framework, decoder) combination,
e.g. 'OpenVINOTorchvisionSSD'. Flavors are independent classes per engine (see docs/design.md): each
implements storage, raw inference and the model family's postprocessing. Preprocessing has one shared
implementation driven by the model info ('preprocess_model_input'), which flavors only override for a
stated reason.

'predict' and 'predict_batch' are fixed pipelines on top of those parts:

    image -> preprocess -> predict_raw / predict_raw_batch -> postprocess -> List[Primitive]

The class extends hafnia's 'InferenceModel', so every flavor works with 'hafnia.dataset.benchmark'.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from hafnia.dataset.benchmark.inference_model import ImageType, InferenceModel
from hafnia.dataset.primitives import Primitive
from pydantic import BaseModel

from inference_model_core.model_info import FlavorKey, ModelBundleInfo
from inference_model_core.preprocessing import PreprocessMeta, preprocess_model_input

RawTensors = Dict[str, np.ndarray]  # Tensors keyed by graph input/output name
PathLike = Union[str, Path]
T = TypeVar("T", bound="InferenceModelBase")


class InferenceModelBase(InferenceModel):
    # The flavor key a subclass handles. Used by 'register_flavor'.
    inference_engine: ClassVar[str]
    framework: ClassVar[str]
    decoder: ClassVar[str]
    # Bundle formats the engine can load, e.g. ('onnx', 'openvino') for OpenVINO
    supported_formats: ClassVar[Tuple[str, ...]]
    # Typed decoder parameters. Validated from the model info defaults merged with load-time overrides.
    params_model: ClassVar[Type[BaseModel]]

    def __init__(self, model_info: ModelBundleInfo, postprocessing_params: Optional[Dict[str, Any]] = None):
        """
        Args:
            model_info: Model info of the bundle.
            postprocessing_params: Overrides of the decoder parameters in the model info, e.g.
                '{"score_threshold": 0.5}'. Overrides only apply to this instance, not to written bundles.
        """
        cls = type(self)
        if (model_info.framework, model_info.runtime.postprocessing.decoder) != (cls.framework, cls.decoder):
            raise ValueError(
                f"'{cls.__name__}' handles framework '{cls.framework}' with decoder '{cls.decoder}', but the model "
                f"info has framework '{model_info.framework}' with decoder "
                f"'{model_info.runtime.postprocessing.decoder}'."
            )
        if model_info.runtime.model_format not in cls.supported_formats:
            raise ValueError(
                f"'{cls.__name__}' can not load model format '{model_info.runtime.model_format}'. "
                f"Supported formats: {list(cls.supported_formats)}."
            )
        self.model_info = model_info
        self.params = cls.params_model.model_validate(
            {**model_info.runtime.postprocessing.params, **(postprocessing_params or {})}
        )

    @classmethod
    def flavor_key(cls) -> FlavorKey:
        return FlavorKey(inference_engine=cls.inference_engine, framework=cls.framework, decoder=cls.decoder)

    # --- storage -------------------------------------------------------------------------------------
    @classmethod
    @abstractmethod
    def load_model(
        cls: Type[T],
        path_bundle: PathLike,
        postprocessing_params: Optional[Dict[str, Any]] = None,
        **engine_options: Any,
    ) -> T:
        """Load a model bundle (zip or directory).

        'postprocessing_params' override decoder parameters (see '__init__'). 'engine_options' are engine
        specific load-time settings such as 'device', which are not part of the model info.
        """

    @abstractmethod
    def write_model(self, path_bundle: PathLike, overwrite: bool = False) -> Path:
        """Write the model as a bundle. A path ending with '.zip' creates a zip bundle, otherwise a directory."""

    # --- engine part ---------------------------------------------------------------------------------
    @abstractmethod
    def predict_raw(self, inputs: RawTensors) -> RawTensors:
        """Run the model on inputs for a single image. No pre- or post-processing."""

    def predict_raw_batch(self, inputs: RawTensors) -> RawTensors:
        """Run the model on batched inputs (batch on axis 0). Only called when 'supports_batching()'.

        The default passes the batch straight to 'predict_raw', which works for graphs with a dynamic
        batch dimension. Override when the engine needs special handling of batches.
        """
        return self.predict_raw(inputs)

    def supports_batching(self) -> bool:
        """True when the first input tensor has a dynamic batch dimension (axis 0 of a rank 4 tensor)."""
        shape = self.model_info.runtime.input.tensors[0].shape
        return len(shape) == 4 and shape[0] is None

    # --- model family part ---------------------------------------------------------------------------
    def preprocess(self, image: ImageType) -> Tuple[RawTensors, PreprocessMeta]:
        """Convert one image into model inputs for a single image, plus the metadata needed to map
        outputs back to the original image.

        Uses the shared, model info driven implementation. Only override for a stated reason: exporters
        calibrate quantization with 'preprocess_model_input', so an override must produce the same inputs.
        """
        return preprocess_model_input(image, self.model_info.runtime)

    @abstractmethod
    def postprocess(self, outputs: RawTensors, meta: PreprocessMeta) -> List[Primitive]:
        """Convert the raw outputs for a single image into hafnia primitives (ground_truth=False)."""

    def collate_raw_inputs(self, inputs: List[RawTensors]) -> RawTensors:
        """Combine single-image inputs into one batch by concatenating on axis 0."""
        return {name: np.concatenate([single[name] for single in inputs], axis=0) for name in inputs[0]}

    def split_raw_outputs(self, outputs: RawTensors, batch_size: int) -> List[RawTensors]:
        """Split batched outputs into single-image outputs, keeping the batch axis (size 1) so each item
        looks like an output of 'predict_raw'. Override when outputs are not batch-major."""
        for name, tensor in outputs.items():
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Output '{name}' has shape {tensor.shape}, expected batch size {batch_size} on axis 0. "
                    f"Override 'split_raw_outputs' in '{type(self).__name__}' for outputs that are not batch-major."
                )
        return [{name: tensor[i : i + 1] for name, tensor in outputs.items()} for i in range(batch_size)]

    # --- fixed pipeline ------------------------------------------------------------------------------
    def predict(self, image: ImageType, sample_dict: Optional[dict] = None) -> List[Primitive]:
        inputs, meta = self.preprocess(image)
        return self.postprocess(self.predict_raw(inputs), meta)

    def predict_batch(
        self,
        images: List[ImageType],
        sample_dicts: Optional[List[dict]] = None,
    ) -> List[List[Primitive]]:
        if sample_dicts is not None and len(sample_dicts) != len(images):
            raise ValueError(
                f"Length of 'sample_dicts' ({len(sample_dicts)}) must match length of 'images' ({len(images)})."
            )
        preprocessed = [self.preprocess(image) for image in images]
        inputs = [single_inputs for single_inputs, _ in preprocessed]
        metas = [meta for _, meta in preprocessed]
        outputs = self.predict_raw_list(inputs)
        return [self.postprocess(single_outputs, meta) for single_outputs, meta in zip(outputs, metas, strict=True)]

    def predict_raw_list(self, inputs: List[RawTensors]) -> List[RawTensors]:
        """The engine part of 'predict_batch': single-image inputs in, single-image outputs out.

        Runs one batched engine call when the model supports batching, otherwise one call per image.
        Separate from 'predict_batch' so latency measurements can time the stages on their own.
        """
        if self.supports_batching() and len(inputs) > 1:
            batch_outputs = self.predict_raw_batch(self.collate_raw_inputs(inputs))
            return self.split_raw_outputs(batch_outputs, batch_size=len(inputs))
        return [self.predict_raw(single_inputs) for single_inputs in inputs]

    def get_model_info(self) -> ModelBundleInfo:
        return self.model_info
