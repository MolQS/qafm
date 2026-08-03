# src/qafm/parameters.py

from __future__ import annotations
import ast
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

class ParameterSet:
    """
    Dynamic container for parameter dataclasses.

    Parameters are stored by their class name, for example:
    - SensorParameters
    - VdwParameters
    - LennardJonesParameters
    """

    def __init__(
            self,
            parameters: Any | Iterable[Any] | Mapping[str, Any] | None = None):
        self._params: dict[str, Any] = {}

        if parameters is None:
            return

        if isinstance(parameters, Mapping):
            for value in parameters.values():
                self.add(value)
            return

        if self._is_iterable_but_not_parameter(parameters):
            for parameter in parameters:
                self.add(parameter)
        else:
            self.add(parameters)

    def add(self, parameter: Any) -> None:
        key = parameter.__class__.__name__

        if key in self._params:
            raise ValueError(f"ParameterSet already contains\
                                parameters for {key!r}.")

        self._params[key] = parameter

    def get(self, parameter_type: type[T], default: T | None = None) -> T:
        key = parameter_type.__name__

        if key not in self._params:
            if default is not None:
                return default

            raise KeyError(f"ParameterSet does not contain {key!r}.")

        value = self._params[key]

        if not isinstance(value, parameter_type):
            raise TypeError(
                f"Stored value for {key!r} must be {parameter_type.__name__}, "
                f"got {type(value).__name__}."
            )

        return value

    def __getitem__(self, parameter_type: type[T] | str) -> Any:
        if isinstance(parameter_type, str):
            return self._params[parameter_type]

        return self.get(parameter_type)

    def __contains__(self, parameter_type: type[Any] | str) -> bool:
        if isinstance(parameter_type, str):
            return parameter_type in self._params

        return parameter_type.__name__ in self._params

    def __repr__(self) -> str:
        keys = ", ".join(self._params.keys())
        return f"ParameterSet({keys})"

    @staticmethod
    def _is_iterable_but_not_parameter(value: Any) -> bool:
        if isinstance(value, (str, bytes, Mapping)):
            return False

        try:
            iter(value)
        except TypeError:
            return False

        return True


def resolve_params(
    model_par: T | ParameterSet | Mapping[str, Any] | None,
    expected_type: type[T],
    default: T,
) -> T:
    """
    Resolve parameters from:
    - the specific parameter dataclass
    - a dynamic ParameterSet
    - a dict using class names as keys
    - None, in which case default is used
    """

    if model_par is None:
        return default

    if isinstance(model_par, expected_type):
        return model_par

    if isinstance(model_par, ParameterSet):
        return model_par.get(expected_type, default=default)

    if isinstance(model_par, Mapping):
        key = expected_type.__name__

        if key not in model_par:
            return default

        value = model_par[key]

        if not isinstance(value, expected_type):
            raise TypeError(
                f"model_par[{key!r}] must be {expected_type.__name__}, "
                f"got {type(value).__name__}."
            )

        return value

    raise TypeError(
        f"Expected {expected_type.__name__}, ParameterSet, dict, or None. "
        f"Got {type(model_par).__name__}."
    )


def load_parameter_values(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    values: dict[str, Any] = {}

    tree = ast.parse(path.read_text())

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise ValueError(
                "Only simple assignments like 'f0 = 300000.0' are allowed."
            )

        key = node.targets[0].id

        try:
            values[key] = ast.literal_eval(node.value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid value for parameter {key!r} in {path}. "
                "Only literal values are allowed," \
                "e.g. 2.5e-17, None, 1.0, 'text'. " \
                "Expressions like '5e-9 * 5e-9' are not supported."
            ) from exc

    return values


def build_dataclass_from_values(cls: type[T], values: Mapping[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass.")

    kwargs = {
        field.name: values[field.name]
        for field in fields(cls)
        if field.name in values
    }

    return cls(**kwargs)


def parameter_set_from_file(
    path: str | Path,
    parameter_types: Iterable[type[Any]] | None = None,
) -> ParameterSet:
    values = load_parameter_values(path)

    if parameter_types is None:
        from qafm.oscillator import OscillatorParameters

        from qafm.interactions.vdw import VdwParameters
        from qafm.interactions.chemical import (
            MorseParameters, LennardJonesParameters
        )
        from qafm.interactions.electrostatic import (
            ElectrostaticParameters,
            SpherePlaneElectrostaticParameters,
        )

        from qafm.ToDoElectrostatics.electrostatic import (
            ElectrostaticModelParameters,
            ElectrostaticOscillationParameters,
        )

        parameter_types = [
            OscillatorParameters,
            VdwParameters,
            MorseParameters,
            LennardJonesParameters,
            ElectrostaticParameters,
            SpherePlaneElectrostaticParameters,
            ElectrostaticModelParameters,
            ElectrostaticOscillationParameters,
        ]

    return ParameterSet([
        build_dataclass_from_values(parameter_type, values)
        for parameter_type in parameter_types
    ])

def default_parameter_set() -> ParameterSet:
    path = files("qafm.config").joinpath("defaultParameterSet.py")
    return parameter_set_from_file(path)