import ast
import datetime
import warnings
from abc import abstractmethod, ABC
from typing import TypeVar, Hashable, Callable, Sequence, Iterator, Iterable, Union, Generic, Any

from xcs.configurable import Configurable, python_type_to_config, datetime_to_config, is_simple_literal

_Self = TypeVar('_Self')
_Value = TypeVar('_Value', bound=Hashable)
_Value2 = TypeVar('_Value2', bound=Hashable)


# (value_type, sparse): vector_class
DEFAULT_IMPLEMENTATIONS: dict[tuple[type[_Value], bool], 'type[Vector[_Value]]'] = {}


def find_common_type(values: Iterable[_Value], preferred: type[_Value] = None) -> type[_Value]:
    types = set()
    preferred_ok = preferred is not None
    for value in values:
        value_type = type(value)
        if value_type in types or any(issubclass(value_type, candidate) for candidate in types):
            continue
        types = {candidate for candidate in types if not issubclass(candidate, value_type)}
        types.add(value_type)
        if preferred_ok and not issubclass(value_type, preferred):
            preferred_ok = False
    if preferred_ok:
        return preferred
    if len(types) != 1:
        return object
    return types.pop()


def vectorize(values: Sequence[_Value], value_type: type[_Value] = None,
              sparse: bool = False) -> 'Vector[_Value]':
    value_type = find_common_type(values, preferred=value_type)
    for sparse_flag in (sparse, not sparse):
        vector_type = DEFAULT_IMPLEMENTATIONS.get((value_type, sparse_flag), None)
        if vector_type is not None:
            break
    if vector_type is None:
        vector_type = Vector
    return vector_type(values)


def vector_from_str(text: str, value_type: type[_Value] = None, sparse: bool = False) -> 'Vector[_Value]':
    values = ast.literal_eval(text)
    return vectorize(values, value_type, sparse)


class Vector(Sequence[_Value], Configurable):
    _storage_type: type[Sequence[_Value]] = tuple
    _storage_type_cast: Callable[[Sequence[_Value]], Sequence[_Value]] = None

    @classmethod
    def from_str(cls, text: str) -> 'Vector[_Value]':
        return cls(ast.literal_eval(text))

    def __init__(self, values: Sequence[_Value], hash_value: int = None):
        assert hash_value is None or isinstance(hash_value, int)
        if not isinstance(values, self._storage_type):
            values = (self._storage_type_cast or self._storage_type)(values)
        assert isinstance(values, self._storage_type)
        self._values = values
        self._hash = hash_value

    def __bool__(self) -> bool:
        return any(bool(value) for value in self._values)

    def __str__(self) -> str:
        return repr(tuple(self._values))

    def __repr__(self) -> str:
        return type(self).__name__ + '(' + repr(tuple(self._values)) + ')'

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[_Value]:
        yield from self._values

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._values[index]
        else:
            return type(self)(self._values[index])

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(tuple(self._values))
        return self._hash

    def __eq__(self, other: Sequence[_Value]) -> bool:
        if not isinstance(other, Vector):
            other = type(self)(other)
        if len(self) != len(other):
            return NotImplemented
        return bool(self.eq(other))

    def __ne__(self, other: Sequence[_Value]) -> bool:
        if not isinstance(other, Vector):
            other = type(self)(other)
        if len(self) != len(other):
            return NotImplemented
        return not bool(self.eq(other))

    def map(self, f: Union[Callable[[_Value], _Value2], Callable[[_Value, ...], _Value2]],
            *args: 'Vector[_Value]') -> 'Vector[_Value2]':
        if args:
            for arg in args:
                if len(self) != len(arg):
                    raise ValueError("Vector length mismatch.")
            return vectorize([f(*row) for row in zip(self._values, *args)])
        else:
            return vectorize([f(v) for v in self._values])

    def reduce(self, f: Callable[[_Value, _Value], _Value], start: _Value) -> _Value:
        result = start
        for value in self._values:
            result = f(result, value)
        return result

    def concat(self, other: 'Vector[_Value]') -> 'Vector[_Value]':
        return vectorize(tuple(self._values) + tuple(other))

    def nonzero(self) -> 'Vector[bool]':
        return self.map(bool)

    def eq(self, other: 'Vector[_Value]') -> 'Vector[bool]':
        return self.map(lambda a, b: a == b, other)

    def get_configuration(self) -> dict[str, Any]:
        config = super().get_configuration()
        assert 'values' not in config
        values_config = []
        for value in self:
            if isinstance(value, Configurable):
                values_config.append(value.get_configuration())
            elif isinstance(value, type):
                values_config.append(python_type_to_config(value))
            elif is_simple_literal(value):
                values_config.append(value)
            elif isinstance(value, datetime.datetime):
                values_config.append(datetime_to_config(value))
            else:
                raise TypeError(type(value))
        config['values'] = values_config
        return config

    @classmethod
    def build(cls: type[_Self], config: dict[str, Any]) -> '_Self':
        values = config['values']
        result = cls(values)
        result.configure(config)
        return result

    def configure(self, config: dict[str, Any]) -> None:
        super().configure(config)
        assert tuple(self._values) == tuple(config['values'])


def default_vector_type(type_: type[_Value], *, sparse: bool = False):
    assert isinstance(type_, type)
    assert isinstance(sparse, bool)
    # noinspection PyTypeChecker
    key: tuple[type[_Value], bool] = (type_, sparse)

    def set_default_vector_type(vector_type: type[Vector[_Value]]):
        assert issubclass(vector_type, Vector)
        if key in DEFAULT_IMPLEMENTATIONS:
            warnings.warn(f"Default {'sparse' if sparse else 'dense'} vector type for value type {type_} "
                          f"is being overwritten.")
        DEFAULT_IMPLEMENTATIONS[key] = vector_type
        return vector_type

    return set_default_vector_type


class VectorCondition(Generic[_Value], Configurable, ABC):

    @classmethod
    @abstractmethod
    def cover(cls: type[_Self], values: Vector[_Value], wildcard_prob: float) -> _Self:
        raise NotImplementedError()

    @abstractmethod
    def mutate(self: _Self, situation: Union[Sequence[_Value], _Self], mutation_probability: float) -> _Self:
        raise NotImplementedError()

    @abstractmethod
    def crossover_with(self: _Self, other: _Self) -> 'tuple[_Self, _Self]':
        raise NotImplementedError()

    @abstractmethod
    def specificity(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self: _Self, situation: Union[Sequence[_Value], _Self]) -> bool:
        raise NotImplementedError()
