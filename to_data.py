from dataclasses import is_dataclass, fields
from typing import Type, get_args, get_origin, Any, TypeVar

T = TypeVar("T")


def to_data(type_definition: Type[T], data: T | dict[str, Any]) -> T:  # data: dict if T is dataclass
    origin = get_origin(type_definition)  # remove generic (the args)
    if origin is None:  # or use as is
        origin = type_definition

    if is_dataclass(origin):
        if not isinstance(data, dict):
            raise TypeError((f"Can only convert from dict to dataclass, "
                             f"but trying to convert to {type_definition} from {type(data)} (from {data})"))
        field_values: dict[str, Any] = data  # needed for if there are extra arguments
        for field in fields(type_definition):
            field_name = field.name
            field_type = field.type
            field_default = field.default
            field_value = data.get(field_name, field_default)

            # Recursively process fields
            field_values[field_name] = to_data(field_type, field_value)

        return type_definition(**field_values)  # instance the dataclass

    # type conversion end
    if not isinstance(data, origin):
        raise TypeError(f"Expected type {origin} (from {type_definition}) but got type {type(data)} (from {data})")
    # type checked start

    if origin is list:
        if len(data) == 0:
            return []
        args = get_args(type_definition)
        if len(args) == 0:  # untyped list
            return data
        inside_type = args[0]
        return [to_data(inside_type, e) for e in data]

    if origin is tuple:  # to not do: handle variable length; use a list
        if len(data) == 0:
            return ()
        args = get_args(type_definition)
        if len(args) == 0:
            return data
        return tuple([to_data(args[i], e) for i, e in enumerate(data)])

    if origin is dict:
        if len(data) == 0:
            return dict()
        args = get_args(type_definition)
        if len(args) == 0:
            return data
        return dict([(to_data(args[0], e[0]), to_data(args[1], e[1])) for e in data.items()])

    return data
