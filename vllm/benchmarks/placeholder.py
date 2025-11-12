import importlib
import re
from typing import Never


def get_vllm_optional_dependencies():
    metadata = importlib.metadata.metadata("vllm")
    requirements = metadata.get_all("Requires-Dist", [])
    extras = metadata.get_all("Provides-Extra", [])

    return {
        extra: [
            re.split(r";|>=|<=|==", req)[0]
            for req in requirements
            if req.endswith(f'extra == "{extra}"')
        ]
        for extra in extras
    }


class _PlaceholderBase:
    """
    Disallows downstream usage of placeholder modules.

    We need to explicitly override each dunder method because
    [`__getattr__`][vllm.utils.import_utils._PlaceholderBase.__getattr__]
    is not called when they are accessed.

    Info:
        [Special method lookup](https://docs.python.org/3/reference/datamodel.html#special-lookup)
    """

    def __getattr__(self, key: str) -> Never:
        """
        The main class should implement this to throw an error
        for attribute accesses representing downstream usage.
        """
        raise NotImplementedError

    # [Basic customization]

    def __lt__(self, other: object):
        return self.__getattr__("__lt__")

    def __le__(self, other: object):
        return self.__getattr__("__le__")

    def __eq__(self, other: object):
        return self.__getattr__("__eq__")

    def __ne__(self, other: object):
        return self.__getattr__("__ne__")

    def __gt__(self, other: object):
        return self.__getattr__("__gt__")

    def __ge__(self, other: object):
        return self.__getattr__("__ge__")

    def __hash__(self):
        return self.__getattr__("__hash__")

    def __bool__(self):
        return self.__getattr__("__bool__")

    # [Callable objects]

    def __call__(self, *args: object, **kwargs: object):
        return self.__getattr__("__call__")

    # [Container types]

    def __len__(self):
        return self.__getattr__("__len__")

    def __getitem__(self, key: object):
        return self.__getattr__("__getitem__")

    def __setitem__(self, key: object, value: object):
        return self.__getattr__("__setitem__")

    def __delitem__(self, key: object):
        return self.__getattr__("__delitem__")

    # __missing__ is optional according to __getitem__ specification,
    # so it is skipped

    # __iter__ and __reversed__ have a default implementation
    # based on __len__ and __getitem__, so they are skipped.

    # [Numeric Types]

    def __add__(self, other: object):
        return self.__getattr__("__add__")

    def __sub__(self, other: object):
        return self.__getattr__("__sub__")

    def __mul__(self, other: object):
        return self.__getattr__("__mul__")

    def __matmul__(self, other: object):
        return self.__getattr__("__matmul__")

    def __truediv__(self, other: object):
        return self.__getattr__("__truediv__")

    def __floordiv__(self, other: object):
        return self.__getattr__("__floordiv__")

    def __mod__(self, other: object):
        return self.__getattr__("__mod__")

    def __divmod__(self, other: object):
        return self.__getattr__("__divmod__")

    def __pow__(self, other: object, modulo: object = ...):
        return self.__getattr__("__pow__")

    def __lshift__(self, other: object):
        return self.__getattr__("__lshift__")

    def __rshift__(self, other: object):
        return self.__getattr__("__rshift__")

    def __and__(self, other: object):
        return self.__getattr__("__and__")

    def __xor__(self, other: object):
        return self.__getattr__("__xor__")

    def __or__(self, other: object):
        return self.__getattr__("__or__")

    # r* and i* methods have lower priority than
    # the methods for left operand so they are skipped

    def __neg__(self):
        return self.__getattr__("__neg__")

    def __pos__(self):
        return self.__getattr__("__pos__")

    def __abs__(self):
        return self.__getattr__("__abs__")

    def __invert__(self):
        return self.__getattr__("__invert__")

    # __complex__, __int__ and __float__ have a default implementation
    # based on __index__, so they are skipped.

    def __index__(self):
        return self.__getattr__("__index__")

    def __round__(self, ndigits: object = ...):
        return self.__getattr__("__round__")

    def __trunc__(self):
        return self.__getattr__("__trunc__")

    def __floor__(self):
        return self.__getattr__("__floor__")

    def __ceil__(self):
        return self.__getattr__("__ceil__")

    # [Context managers]

    def __enter__(self):
        return self.__getattr__("__enter__")

    def __exit__(self, *args: object, **kwargs: object):
        return self.__getattr__("__exit__")


class PlaceholderModule(_PlaceholderBase):
    """
    A placeholder object to use when a module does not exist.

    This enables more informative errors when trying to access attributes
    of a module that does not exist.
    """

    def __init__(self, name: str) -> None:
        super().__init__()

        # Apply name mangling to avoid conflicting with module attributes
        self.__name = name

    def placeholder_attr(self, attr_path: str):
        return _PlaceholderModuleAttr(self, attr_path)

    def __getattr__(self, key: str) -> Never:
        name = self.__name

        try:
            importlib.import_module(name)
        except ImportError as exc:
            for extra, names in get_vllm_optional_dependencies().items():
                if name in names:
                    msg = f"Please install vllm[{extra}] for {extra} support"
                    raise ImportError(msg) from exc

            raise exc

        raise AssertionError(
            "PlaceholderModule should not be used "
            "when the original module can be imported"
        )


class _PlaceholderModuleAttr(_PlaceholderBase):
    def __init__(self, module: PlaceholderModule, attr_path: str) -> None:
        super().__init__()

        # Apply name mangling to avoid conflicting with module attributes
        self.__module = module
        self.__attr_path = attr_path

    def placeholder_attr(self, attr_path: str):
        return _PlaceholderModuleAttr(self.__module, f"{self.__attr_path}.{attr_path}")

    def __getattr__(self, key: str) -> Never:
        getattr(self.__module, f"{self.__attr_path}.{key}")

        raise AssertionError(
            "PlaceholderModule should not be used "
            "when the original module can be imported"
        )