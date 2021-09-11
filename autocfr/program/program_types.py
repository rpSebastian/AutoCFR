import numpy as np
from typing import Any, Dict, List

nptype = np.float64


class Type:
    def __init__(self):
        pass


supertypes: Dict[Type, Type] = {}
subtypes: Dict[Type, List[Type]] = {}


def register_supertype(supertype: Any):
    def _register_supertype(program_type: Type):
        assert (
            program_type not in supertypes
        ), f"{program_type} already has a supertype!"
        supertypes[program_type] = supertype

        if supertype not in subtypes:
            subtypes[supertype] = []
        subtypes[supertype].append(program_type)

        return program_type

    return _register_supertype


def type_and_supertypes(program_type: Type):
    return [program_type] + all_supertypes(program_type)


def all_supertypes(program_type: Type):
    if program_type not in supertypes:
        return [Type]
    else:
        supertype = supertypes[program_type]
        return [supertype] + all_supertypes(supertype)


def equal_or_supertype(program_type: Type, potential_supertype: Type):
    return (
        potential_supertype == Type
        or program_type == potential_supertype
        or potential_supertype in all_supertypes(program_type)
    )


class Vector(Type):
    value_class = np.ndarray

    @classmethod
    def is_valid_value(cls, value):
        f1 = not np.isnan(value).any()
        f2 = not np.isinf(value).any()
        f3 = value.dtype == nptype
        return f1 and f2 and f3


@register_supertype(Vector)
class Scalar(Vector):
    value_class = nptype

    @classmethod
    def is_valid_value(cls, value):
        f1 = not np.isnan(value).any()
        f2 = not np.isinf(value).any()
        f3 = value.dtype == nptype
        return f1 and f2


class DataStructure(Type):
    def create_empty(self):
        raise NotImplementedError()


@register_supertype(Scalar)
class Constant(DataStructure):
    value_class = nptype

    def __init__(self, constant_value):
        super().__init__()
        self.constant_value = constant_value

    def create_empty(self):
        return nptype(self.constant_value)
