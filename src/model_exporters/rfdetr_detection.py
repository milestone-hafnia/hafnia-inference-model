"""Model bundles of RF-DETR detection models.

Sources: a model archive of trainer-object-detection (e.g. a platform checkpoint) or pretrained COCO weights by
name. The bundle is a graph of components:

    load_image -> resize -> to_tensor -> network -> topk -> to_bboxes

The network is a torch network ('rfdetr_network', defined in 'model_exporters.definitions'): input
'[N, 3, R, R]', outputs one box per query ('dets', normalized center x, center y, width, height) and the class
logits per query ('labels'). It converts to ONNX with
RF-DETR's own export, and from there to OpenVINO (see 'inference_model_core.convert'). Sigmoid, top-k over
(query, class) and the box conversion run after the network ('query_topk'), as RF-DETR's 'PostProcess'.
DETR-style models need no NMS.

Preprocessing follows RF-DETR's 'predict': resize to R x R (torchvision 'resize', antialiased on tensors), scale
to [0, 1], the model's mean/std.

Logit column i is class i of the task, also for COCO ids without a category (placeholder classes named
'NotDefined_*', as in the trainer's archives). A model trained on N classes has one more column ("no object"),
which is not a task class: 'detections_to_bboxes' drops labels beyond the task's classes.
"""

import argparse
import importlib.metadata
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox
from rfdetr import detr
from rfdetr.util.coco_classes import COCO_CLASSES

from inference_model_core import ConversionInfo, GraphInfo, ModelBundleInfo, NetworkParams, TensorInfo, write_bundle
from inference_model_core.convert import TARGET_ONNX, TARGET_OPENVINO, TARGET_OPENVINO_INT8
from inference_model_core.definitions import DetectionsToBboxes, ImageToTensor, LoadImage, QueryTopK, ResizeImage
from inference_model_torch import register_component
from inference_model_torch.network import OnnxExport, TorchNetwork, load_state_dict_file
from inference_model_torch.onnx_export import DEFAULT_OPSET_VERSION, onnx_options
from model_exporters.class_names import detection_task, placeholder_class_name
from model_exporters.definitions import RFDETRNetwork
from model_exporters.trainer_archive import open_trainer_archive

FRAMEWORK = "rfdetr"
WEIGHTS_FILE = "network/weights.pth"
TASK_NAME = "object_detection"
# Score threshold of trainer-object-detection's inference ('InferenceConfig.threshold')
DEFAULT_SCORE_THRESHOLD = 0.05

# Model name (as in trainer-object-detection's 'model_config.json') -> RF-DETR class. Detection models only.
MODEL_CLASSES: Dict[str, Type[detr.RFDETR]] = {
    "RFDETRNano": detr.RFDETRNano,
    "RFDETRSmall": detr.RFDETRSmall,
    "RFDETRMedium": detr.RFDETRMedium,
    "RFDETRLarge": detr.RFDETRLarge,
}


@register_component
class RFDETRModule(TorchNetwork[RFDETRNetwork]):
    model: detr.RFDETR

    def build_model(self, path_weights: Path, device: torch.device) -> detr.RFDETR:
        # RF-DETR unpickles its checkpoints. Bundles hold checkpoints of tensors and plain values only (see
        # 'write_checkpoint'): reading them here without unpickling objects rejects anything else.
        load_state_dict_file(path_weights)
        model = _model_class(self.params.model_name)(pretrain_weights=str(path_weights))
        model.model.model = model.model.model.to(device).eval()
        model.model.device = device
        return model

    def forward(self, input: torch.Tensor) -> Dict[str, Any]:  # 'input' is the tensor name of RF-DETR's export
        outputs = self.model.model.model(input)
        return {"dets": outputs["pred_boxes"], "labels": outputs["pred_logits"]}

    def export_onnx(self, path_onnx: Path, options: Dict[str, Any]) -> OnnxExport:
        options = onnx_options(options, defaults={"opset_version": DEFAULT_OPSET_VERSION, "dynamic_batch": True})
        model, dynamic_batch = self.model, options["dynamic_batch"]
        with tempfile.TemporaryDirectory(prefix="rfdetr_export_") as tmp_dir:
            model.export(
                output_dir=tmp_dir, dynamic_batch=dynamic_batch, opset_version=options["opset_version"], verbose=False
            )
            (path_exported,) = Path(tmp_dir).glob("*.onnx")
            path_exported.replace(path_onnx)

        batch, resolution = None if dynamic_batch else 1, model.model.resolution
        n_queries, n_columns = model.model_config.num_queries, model.model_config.num_classes + 1
        params = NetworkParams(
            input_tensors=[TensorInfo(name="input", shape=[batch, 3, resolution, resolution])],
            output_tensors=[
                TensorInfo(name="dets", shape=[batch, n_queries, 4]),
                TensorInfo(name="labels", shape=[batch, n_queries, n_columns]),
            ],
        )
        return OnnxExport(params=params, batched=True, provenance={"rfdetr": importlib.metadata.version("rfdetr")})


def load_pretrained_model(model_name: str) -> Tuple[detr.RFDETR, List[str]]:
    """Pretrained COCO model (weights are downloaded on first use). Returns (model, category name per logit
    column), with placeholders for COCO ids without a category."""
    model = _model_class(model_name)()
    n_columns = model.model_config.num_classes + 1
    categories = [COCO_CLASSES.get(column, placeholder_class_name(column)) for column in range(n_columns)]
    return model, categories


def create_pretrained_rfdetr_detection_bundle(
    model_name: str, path_bundle: Union[str, Path], overwrite: bool = False
) -> Path:
    model, categories = load_pretrained_model(model_name)
    path_weights = Path(model.model_config.pretrain_weights or "")
    return create_rfdetr_detection_bundle(
        model,
        model_name=model_name,
        path_weights=path_weights,
        categories=categories,
        path_bundle=path_bundle,
        name=model_name,
        variant=model_name,
        overwrite=overwrite,
        provenance={"weights": path_weights.name},
    )


def create_trainer_archive_rfdetr_bundle(
    path_archive: Union[str, Path], path_bundle: Union[str, Path], overwrite: bool = False
) -> Path:
    """Bundle of a trainer-object-detection archive, with the task of the archive."""
    with open_trainer_archive(path_archive) as (config, path_weights):
        model_name = config["name"]
        task = TaskInfo.model_validate(config["task"])
        if task.primitive is not Bbox:
            raise ValueError(f"Only detection (Bbox) models are supported, '{model_name}' is '{task.primitive}'.")
        model = _model_class(model_name)(pretrain_weights=str(path_weights))
        return create_rfdetr_detection_bundle(
            model,
            model_name=model_name,
            path_weights=path_weights,
            categories=list(task.get_class_names() or []),
            path_bundle=path_bundle,
            name=model_name,
            variant=model_name,
            overwrite=overwrite,
            provenance={"trainer_archive": Path(path_archive).name},
        )


def create_rfdetr_detection_bundle(
    model: detr.RFDETR,
    model_name: str,
    path_weights: Union[str, Path],
    categories: List[str],
    path_bundle: Union[str, Path],
    name: str,
    variant: Optional[str] = None,
    overwrite: bool = False,
    provenance: Optional[dict] = None,
) -> Path:
    """Write an RF-DETR detection model as a model bundle with a torch network.

    Args:
        model: RF-DETR model ('rfdetr.detr.RFDETR') built from 'path_weights'. Its settings (resolution,
            normalization, number of detections) go into the bundle.
        model_name: RF-DETR model of 'model' (see 'MODEL_CLASSES'). The bundle stores the weights and rebuilds
            the model from this name.
        path_weights: RF-DETR checkpoint of 'model'. It is unpickled, so it has to come from a trusted source.
            The bundle gets a copy with only the weights and plain settings.
        categories: Class name per logit column (column i = class i), unique. Columns beyond the list (the
            "no object" column) are not task classes.
        path_bundle: Output bundle, a '.zip' path or a directory.
        name: Model name stored in the model info.
        variant: Informational variant stored in the model info.
        overwrite: Replace an existing bundle.
        provenance: Extra provenance information stored with the network component.
    """
    model_config = model.model_config
    if model_config.segmentation_head or getattr(model_config, "use_grouppose_keypoints", False):
        raise ValueError("Only RF-DETR detection models are supported (no segmentation or keypoint heads).")
    _model_class(model_name)
    resolution = model.model.resolution
    n_columns = model_config.num_classes + 1
    if len(categories) > n_columns:
        raise ValueError(f"Got {len(categories)} categories for a model with {n_columns} logit columns.")
    task = detection_task(categories, TASK_NAME)

    components = [
        LoadImage().connect(name="load_image", inputs={"image": "image"}, outputs={"image": "image_rgb"}),
        ResizeImage(
            mode="stretch", height=resolution, width=resolution, interpolation="linear", antialias=True
        ).connect(
            name="resize",
            inputs={"image": "image_rgb"},
            outputs={"image": "image_resized", "transform": "image_transform"},
        ),
        ImageToTensor(scale=1 / 255, mean=list(model.means), std=list(model.stds)).connect(
            name="to_tensor", inputs={"image": "image_resized"}, outputs={"tensor": "input_tensor"}
        ),
        RFDETRNetwork(model_name=model_name).connect(
            name="network",
            inputs={"input": "input_tensor"},
            outputs={"dets": "query_boxes", "labels": "query_logits"},
            files=[WEIGHTS_FILE],
            batched=True,
            conversions={
                TARGET_ONNX: ConversionInfo(options={"opset_version": DEFAULT_OPSET_VERSION, "dynamic_batch": True}),
                TARGET_OPENVINO: ConversionInfo(),
                # NNCF's default quantized the transformer as a CNN and dropped mAP from 0.66 to 0.0 on the micro
                # dataset. Its transformer mode keeps softmax, layer norm and similar operations in floating point.
                TARGET_OPENVINO_INT8: ConversionInfo(options={"model_type": "transformer"}),
            },
            provenance={
                "created_by": __name__,
                "rfdetr": importlib.metadata.version("rfdetr"),
                "torch": torch.__version__,
                **(provenance or {}),
            },
        ),
        QueryTopK(num_select=model.model.postprocess.num_select).connect(
            name="topk",
            inputs={"boxes": "query_boxes", "logits": "query_logits", "transform": "image_transform"},
            outputs={"boxes": "boxes", "scores": "scores", "labels": "labels"},
        ),
        DetectionsToBboxes(task_name=TASK_NAME, score_threshold=DEFAULT_SCORE_THRESHOLD).connect(
            name="to_bboxes",
            inputs={"boxes": "boxes", "scores": "scores", "labels": "labels", "transform": "image_transform"},
            outputs={"predictions": "predictions"},
        ),
    ]
    model_info = ModelBundleInfo(
        name=name,
        framework=FRAMEWORK,
        variant=variant,
        tasks=[task],
        graph=GraphInfo(inputs=["image"], outputs=["predictions"], components=components),
    )
    with tempfile.TemporaryDirectory(prefix="rfdetr_bundle_") as tmp_dir:
        path_checkpoint = write_checkpoint(path_weights, Path(tmp_dir) / "weights.pth")
        return write_bundle(path_bundle, model_info, files={WEIGHTS_FILE: path_checkpoint}, overwrite=overwrite)


def write_checkpoint(path_source: Union[str, Path], path_checkpoint: Path) -> Path:
    """Copy an RF-DETR checkpoint with only what inference needs: the weights and the plain settings.

    Training state (optimizer, scheduler) is dropped, which shrinks the file to a third, and the result holds
    tensors and plain values only, so it can be read without unpickling objects.
    """
    checkpoint = torch.load(path_source, map_location="cpu", weights_only=False)
    # 'model' + 'args' in RF-DETR's own checkpoints, 'state_dict' + 'hyper_parameters' in PyTorch Lightning's
    weights = {key: dict(checkpoint[key]) for key in ("model", "state_dict") if key in checkpoint}
    if not weights:
        raise ValueError(f"RF-DETR checkpoint '{path_source}' has neither 'model' nor 'state_dict' weights.")
    settings = {}
    for key in ("args", "hyper_parameters"):
        values = checkpoint.get(key)
        if isinstance(values, argparse.Namespace):
            values = vars(values)
        if isinstance(values, dict):
            settings[key] = {name: value for name, value in values.items() if _is_plain(value)}
    torch.save({**weights, **settings}, path_checkpoint)
    return path_checkpoint


def _is_plain(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return all(_is_plain(item) for item in value)
    return value is None or isinstance(value, (str, int, float, bool))


def _model_class(model_name: str) -> Type[detr.RFDETR]:
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported RF-DETR model '{model_name}'. Options: {list(MODEL_CLASSES)}")
    return MODEL_CLASSES[model_name]
