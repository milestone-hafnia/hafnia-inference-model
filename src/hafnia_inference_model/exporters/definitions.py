"""Definitions of the torch networks of the supported model families.

A torch network is stored by reference: the bundle holds the weights and these parameters name the code that
builds the module. The implementations are in the family modules ('torchvision_detection',
'torchvision_classification', 'rfdetr_detection'), which need torch and the model's library. This module needs
neither.
"""

from pydantic import Field

from hafnia_inference_model.core import ComponentDefinition, Port

_DETECTION_OUTPUTS = (
    Port("boxes", "(K, 4) xyxy boxes in pixels of the image"),
    Port("scores", "(K,) scores"),
    Port("labels", "(K,) model labels"),
)


class TorchvisionClassifier(ComponentDefinition):
    """A torchvision classification network."""

    type = "torchvision_classifier"
    input_ports = (Port("images", "(N, 3, S, S) normalized images"),)
    output_ports = (Port("logits", "(N, classes) class logits"),)

    architecture: str = Field(description="torchvision architecture, e.g. 'resnet18'")
    num_classes: int = Field(gt=0)
    image_size: int = Field(gt=0, description="Square input size the model was trained on")


class TorchvisionSSDNetwork(ComponentDefinition):
    """SSD backbone and head with box decoding, for the fixed input size of the architecture."""

    type = "torchvision_ssd_network"
    input_ports = (Port("images", "(N, 3, H, W) normalized images"),)
    output_ports = (
        Port("boxes", "(N, A, 4) decoded xyxy boxes per anchor, in input pixels"),
        Port("scores", "(N, A, labels) class scores per anchor"),
    )

    architecture: str = Field(description="torchvision architecture, e.g. 'ssdlite320_mobilenet_v3_large'")
    num_classes: int = Field(gt=0, description="Number of model labels, including the background label")


class TorchvisionFasterRCNN(ComponentDefinition):
    """The full Faster R-CNN model for one image. Resize, normalization and NMS are part of the network."""

    type = "torchvision_faster_rcnn"
    input_ports = (Port("image", "(3, H, W) image in [0, 1]"),)
    output_ports = _DETECTION_OUTPUTS

    architecture: str = Field(description="torchvision architecture, e.g. 'fasterrcnn_mobilenet_v3_large_320_fpn'")
    num_classes: int = Field(gt=0, description="Number of model labels, including the background label")
    min_score: float = Field(description="Score threshold in the network. The runtime threshold filters on top")
    nms_iou_threshold: float
    max_detections: int = Field(gt=0, description="Highest scoring detections the network returns")


class RFDETRNetwork(ComponentDefinition):
    """The RF-DETR network. The port names are the tensor names of RF-DETR's ONNX export."""

    type = "rfdetr_network"
    input_ports = (Port("input", "(N, 3, R, R) normalized images"),)
    output_ports = (
        Port("dets", "(N, Q, 4) boxes per query as normalized (center x, center y, width, height)"),
        Port("labels", "(N, Q, C) class logits per query"),
    )

    model_name: str = Field(description="RF-DETR model, e.g. 'RFDETRNano'")
