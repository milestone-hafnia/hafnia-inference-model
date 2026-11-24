"""Component definitions: what a component of a model graph is, without how it is computed.

A definition is a pydantic model for one component type. Its fields are the configuration stored as 'params' in
the model info, and it declares the ports: the named inputs the component reads and the named outputs it writes.

    class ResizeImage(ComponentDefinition):
        type = "resize_image"
        input_ports = (Port("image", "(H, W, 3) uint8 RGB image"),)
        output_ports = (Port("image", "Resized image"), Port("transform", "How it maps to the original"))

        height: int
        width: int

A definition holds no implementation and depends on no tensor library, so bundles can be described, written and
checked without one. The torch graph engine ('inference_model_torch') has an implementation per definition and
runs the graph. 'connect' turns a definition into a step of a graph:

    ResizeImage(height=320, width=320).connect(
        "resize", inputs={"image": "image_rgb"}, outputs={"image": "image_resized", "transform": "transform"}
    )
"""

from typing import Any, ClassVar, Dict, List, Mapping, NamedTuple, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ConfigDict

from inference_model_core.model_bundle import ComponentInfo, ConversionInfo


class Port(NamedTuple):
    """A named input or output of a component."""

    name: str
    description: str = ""
    optional: bool = False  # Inputs only: the component also works when the port is not connected


class Ports(NamedTuple):
    inputs: Tuple[str, ...]
    optional_inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]


D = TypeVar("D", bound="ComponentDefinition")


class ComponentDefinition(BaseModel):
    """Base class of all component definitions.

    Subclasses set 'type', 'input_ports' and 'output_ports' and declare their configuration as fields.
    Definitions whose ports depend on their configuration (e.g. the tensor names of a network) override 'ports'.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ClassVar[str]
    input_ports: ClassVar[Tuple[Port, ...]] = ()
    output_ports: ClassVar[Tuple[Port, ...]] = ()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        for kind, ports in (("input", cls.input_ports), ("output", cls.output_ports)):
            names = [port.name for port in ports]
            if len(set(names)) != len(names):
                raise TypeError(f"'{cls.__name__}' declares {kind} ports {names}: port names must be unique.")
        optional_outputs = [port.name for port in cls.output_ports if port.optional]
        if optional_outputs:
            raise TypeError(f"'{cls.__name__}' declares optional outputs {optional_outputs}: only inputs can be.")

    def ports(self) -> Ports:
        """The port names of this component."""
        cls = type(self)
        return Ports(
            inputs=tuple(port.name for port in cls.input_ports if not port.optional),
            optional_inputs=tuple(port.name for port in cls.input_ports if port.optional),
            outputs=tuple(port.name for port in cls.output_ports),
        )

    def connect(
        self,
        name: str,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        files: Optional[List[str]] = None,
        batched: bool = False,
        conversions: Optional[Dict[str, ConversionInfo]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ComponentInfo:
        """The component as a step of a graph: its component info for the model info.

        Args:
            name: Unique name in the graph.
            inputs: Input port -> data dictionary key it reads.
            outputs: Output port -> data dictionary key it writes.
            files: Files of the component, relative to the bundle.
            batched: Call the component once per batch instead of once per sample.
            conversions: Conversion target -> support and export options.
            provenance: Where the component comes from.
        """
        info = ComponentInfo(
            name=name,
            type=type(self).type,
            inputs=inputs,
            outputs=outputs,
            params=self.model_dump(mode="json"),
            files=files or [],
            batched=batched,
            conversions=conversions or {},
            provenance=provenance,
        )
        self.check_connections(info)
        return info

    @classmethod
    def from_info(cls: Type[D], info: ComponentInfo, params: Optional[Mapping[str, Any]] = None) -> D:
        """The validated definition of a component of a model info.

        Args:
            info: Component info of the model info.
            params: Overrides of the parameters in 'info'.
        """
        if info.type != cls.type:
            raise ValueError(
                f"'{cls.__name__}' defines type '{cls.type}', but component '{info.name}' is '{info.type}'."
            )
        definition = cls.model_validate({**info.params, **(params or {})})
        definition.check_connections(info)
        return definition

    def check_connections(self, info: ComponentInfo) -> None:
        """The component info must connect all required inputs and only known ports."""
        ports = self.ports()
        missing = sorted(set(ports.inputs) - set(info.inputs))
        unknown_inputs = sorted(set(info.inputs) - set(ports.inputs) - set(ports.optional_inputs))
        unknown_outputs = sorted(set(info.outputs) - set(ports.outputs))
        if missing or unknown_inputs or unknown_outputs:
            raise ValueError(
                f"Component '{info.name}' ('{type(self).type}') has inputs {list(ports.inputs)}, optional inputs "
                f"{list(ports.optional_inputs)} and outputs {list(ports.outputs)}. Missing inputs: {missing}, "
                f"unknown inputs: {unknown_inputs}, unknown outputs: {unknown_outputs}."
            )
