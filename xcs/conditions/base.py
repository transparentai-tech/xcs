import math
import random
from abc import abstractmethod, ABC
from typing import TypeVar, Hashable, Generic, Mapping, Iterator, Callable
import operator

from vectorface import Vector, BoolVector
from xcs.conditions import sparse_ops
from xcs.probabilities import Probability

__all__ = [
    'Condition',
    'BitCondition',
    'IntervalCondition',
]


_Self = TypeVar('_Self')
_Value = TypeVar('_Value', bound=Hashable)
_Value2 = TypeVar('_Value2', bound=Hashable)
_Situation = TypeVar('_Situation', bound=Hashable)


class Condition(Generic[_Situation], ABC):

    @classmethod
    @abstractmethod
    def cover(cls: type[_Self], situation: _Situation, **kwargs) -> _Self:
        """Create a new condition that matches against the given situation."""
        raise NotImplementedError()

    @abstractmethod
    def mutate(self: _Self, *, situation: _Situation = None, condition: _Self = None, **kwargs) -> _Self:
        """Create a new condition by mutating this one.

        If a situation is provided, the created condition will have a match degree for
        the situation which is equal to or greater than the match degree that this
        condition has for it. That is, the following assertion will always succeed:

            assert self.matches(situation) <= self.mutate(situation).matches(situation)

        If a condition is provided, the created condition will be closer to generalizing
        the provided condition than this condition is. That is, a loop of the following
        form will terminate with probability 1, with all assertions passing:

            mutated = self
            while not mutated.generalizes(other):
                mutated = mutated.mutate(condition=other)
            assert mutated.generalizes(other)
            assert mutated.mutate(condition=other).generalizes(other)
        """
        raise NotImplementedError()

    @abstractmethod
    def cross_with(self: _Self, other: _Self, **kwargs) -> 'tuple[_Self, _Self]':
        """Create a new condition by crossing this condition over with another condition."""
        raise NotImplementedError()

    @abstractmethod
    def expected_value(self) -> None | Probability:
        """If a well-defined maximum entropy distribution (i.e. a uniform distribution)
        exists for the situation space this condition accepts, return the expected value
        of the matches() method for a sample drawn from that distribution. Otherwise,
        return None."""
        raise NotImplementedError()

    @abstractmethod
    def generalizes(self, other: 'Condition[_Value]') -> None | bool:
        """Returns True iff self(situation) >= other(situation) for all possible situation
        vectors in the situation space. If it cannot be determined whether this is the
        case due to incompatible types, returns None."""
        raise NotImplementedError()

    @abstractmethod
    def matches(self, situation: _Situation) -> bool | Probability:
        """Return a boolean or probability indicating whether/to what degree the situation
        vector is matched by this condition."""
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def equivalent(self, other: 'Condition[_Value]') -> bool:
        return self.generalizes(other) and other.generalizes(self)

    def __eq__(self, other):
        if not isinstance(other, Condition):
            return NotImplemented
        # NOTE: We have to add the hash comparison here due to Python semantics,
        # since allowing hashes to be unequal while values compare as equal can
        # break hashing data structures like dicts and sets. Not checking the
        # hash value here would mean that every class would have to understand
        # every other class's hash algorithm if they can ever compare as equivalent.
        # If you need to check whether the two conditions are truly equivalent, use
        # the equivalent() method directly instead of the == operator.
        return self.equivalent(self) and (type(self) is type(other) or hash(self) == hash(other))

    def __ne__(self, other):
        return not (self == other)

    def __le__(self, other):
        if not isinstance(other, Condition):
            return NotImplemented
        return other.generalizes(self)

    def __ge__(self, other):
        if not isinstance(other, Condition):
            return NotImplemented
        return self.generalizes(other)

    def __lt__(self, other):
        if not isinstance(other, Condition):
            return NotImplemented
        return other.generalizes(self) and not self == other

    def __gt__(self, other):
        if not isinstance(other, Condition):
            return NotImplemented
        return self.generalizes(other) and not self == other


class BitCondition(Condition[BoolVector]):
    """A pair of bit strings, one indicating the bit values, and the other
    indicating the bit mask, which together act as a matching template for
    bit strings. Like bit strings, bit conditions are hashable and
    immutable. Think of BitConditions as patterns which can match against
    BitStrings of the same length. At each index, we can have a 1, a 0, or
    a # (wildcard). If the value is 1 or 0, the BitString must have the
    same value at that index. If the value is #, the BitString can have any
    value at that index.

    BitConditions are matched against BitStrings in one of two ways:
        Method 1:
            result = condition // bitstring
            # result now contains a new BitString which contains a 1 for
            # each position that violated the pattern, and a 0 for each
            # position that did not. This tells us exactly where the
            # condition and the bitstring disagree
        Method 2:
            result = condition(bitstring)
            # result now contains a single Boolean value which is True if
            # the bitstring fully satisfies the pattern specified by the
            # condition, or False if the bitstring disagrees with the
            # condition at one or more indices

    BitConditions can also match against other BitConditions in the same
    way that they are matched against BitStrings, with the sole exception
    that if the condition being used as the pattern specifies a 1 or 0 at a
    particular index, and the condition being used as the substrate
    contains an # at that point, the match fails. This means that if
    you have two conditions, condition1 and condition2, where condition1
    matches a bitstring and condition2 matches condition1, then condition2
    is guaranteed to match the bitstring, as well.

    Usage:
        # A few ways to create a BitCondition instance
        condition1 = BitCondition('001###01#1')
        condition2 = BitCondition(BitString('0010010111'),
                                  BitString('1110001101'))
        assert condition1 == condition2
        condition3 = BitCondition.cover('0010010111', .25)
        assert condition3(BitString('0010010111'))  # It matches

        # They print up nicely
        assert str(condition1) == '001###01#1'
        print(condition1)  # Prints: 001###01#1
        print(repr(condition1))  # Prints: BitCondition('001###01#1')

        # Indexing is from left to right, like an ordinary string.
        # (Wildcards are represented as the value None at the given index.)
        assert condition1[0] == 0
        assert condition1[-1] == 1
        assert condition1[4] is None

        # They are immutable
        condition1[3] = 0  # This will raise a TypeError

        # Slicing works
        assert condition1[3:-3] == BitCondition('###0')

        # You can iterate over them
        for bit in condition1:
            if bit is None:
                print("Found a wildcard!")

        # Unlike bitstrings, they cannot be cast as ints
        as_int = int(condition1)  # This will raise a TypeError

        # They can be used in hash-based containers
        s = {condition1, condition3}
        d = {condition1: "a", condition3: "b"}

        # Unlike bitstrings, they do not support the `any()` method
        condition1.any()  # This will raise an AttributeError

        # Unlike bitstrings, BitCondition.count() returns the number of
        # bits that are not wildcards, rather than the number of bits that
        # have a value of 1.
        assert condition1.count() == condition1.mask.count() == 6

        # The bitwise operators for BitConditions work differently from
        # those of BitStrings; provided the bits of each condition are
        # compatible, i.e. there is no point where their bits disagree
        # and neither of them is a wildcard, then &, |, and ~ actually
        # represent set operations over the BitStrings that the conditions
        # will match.
        assert condition1 & condition1 == condition1
        assert condition1 | condition1 == condition1
        assert (condition1 | ~condition1)(BitString.random(10))
        assert condition1(condition1 & condition3)  # They are compatible
        assert condition3(condition1 & condition3)  # They are compatible
        assert (condition1 | condition3)(condition1)  # They are compatible
        assert (condition1 | condition3)(condition3)  # They are compatible

        # BitConditions can also be concatenated together like strings
        concatenation = condition1 + condition3
        assert len(concatenation) == 10 * 2

        # They support the Genetic Algorithm's crossover operator directly
        child1, child2 = condition1.cross_with(condition3)

    Init Arguments:
        bits: If mask is provided, a sequence from which the bits of the
            condition can be determined. If mask is omitted, a sequence
            from which the bits and mask of the condition can be
            determined.
        mask: None, or a sequence from which the mask can be determined,
            having the same length as the sequence provided for bits.
    """
    _hash_value: int = None

    @classmethod
    def cover(cls, situation: BoolVector, *, wildcard_prob: Probability = None, **kwargs) -> 'BitCondition':
        elements = sparse_ops.cover(situation, wildcard_prob)
        return cls(len(situation), elements)

    @classmethod
    def from_str(cls, bits: str) -> 'BitCondition':
        elements = {}
        for index, char in enumerate(bits):
            if char == '1':
                elements[index] = True
            elif char == '0':
                elements[index] = False
        return cls(len(bits), elements)

    def __init__(self, length: int, elements: Mapping[int, bool]):
        self._length = length
        self._elements = dict(elements)
        assert all(0 <= index < length for index in self._elements)

    def mutate(self, *, situation: BoolVector = None, condition: 'BitCondition' = None,
               mutation_probability: Probability = None, **kwargs) -> 'BitCondition':
        if condition is None:
            other_elements = None
        else:
            other_elements = condition._elements
        elements = sparse_ops.mutate(self._length, self._elements, situation=situation, parent2=other_elements,
                                     mutation_probability=mutation_probability)
        return type(self)(self._length, elements)

    def cross_with(self, other: 'BitCondition', *, points: int = 2, **kwargs) -> 'tuple[BitCondition, BitCondition]':
        assert self._length == other._length
        child1, child2 = sparse_ops.cross_with(self._length, self._elements, other._elements, points=points)
        return type(self)(self._length, child1), type(self)(self._length, child2)

    def expected_value(self) -> None | Probability:
        return Probability(2 ** -len(self._elements))

    def generalizes(self, other: 'Condition[BoolVector]') -> None | bool:
        if not isinstance(other, BitCondition):
            return None
        return sparse_ops.generalizes(self._elements, other._elements, operator.eq)

    def matches(self, situation: BoolVector) -> bool | Probability:
        return sparse_ops.matches(self._elements, situation, operator.eq)

    def concat(self, *others: 'BitCondition') -> 'BitCondition':
        elements = self._elements.copy()
        length = self._length
        for other in others:
            for index, element in other._elements.items():
                elements[length + index] = element
            length += other._length
        return type(self)(length, elements)

    def equivalent(self, other: 'Condition[bool]') -> bool:
        if type(other) is BitCondition:
            other: BitCondition
            return self._elements == other._elements
        else:
            return super().equivalent(other)

    def __hash__(self) -> int:
        if self._hash_value is None:
            self._hash_value = hash((self._length, frozenset(self._elements.items())))
        return self._hash_value

    def __str__(self) -> str:
        return ''.join('1' if bit else ('#' if bit is None else '0') for bit in self)

    def __repr__(self) -> str:
        return type(self).__name__ + '.from_str(' + repr(str(self)) + ')'

    def __iter__(self) -> Iterator[None | bool]:
        for index in range(self._length):
            yield self._elements.get(index, None)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._elements.get(item, None)
        else:
            assert isinstance(item, slice)
            indices = range(*item.indices(self._length))
            elements = {index - indices.start: value for index, value in self._elements.items() if index in indices}
            return type(self)(len(indices), elements)

    def __and__(self, other: 'BitCondition') -> 'BitCondition':
        if not isinstance(other, BitCondition):
            return NotImplemented
        elements = self._elements.copy()
        for index, value in other._elements.items():
            if index not in elements:
                elements[index] = value
            elif elements[index] != value:
                del elements[index]
        return type(self)(self._length, elements)

    def __or__(self, other: 'BitCondition') -> 'BitCondition':
        if not isinstance(other, BitCondition):
            return NotImplemented
        elements = {}
        for index, value in self._elements.items():
            if other._elements.get(index, None) == value:
                elements[index] = value
        return type(self)(self._length, elements)

    def __invert__(self) -> 'BitCondition':
        elements = {index: not value for index, value in self._elements.items()}
        return type(self)(self._length, elements)


class IntervalCondition(Condition[_Value]):
    _hash_value: int = None

    @classmethod
    def default_interval_radius_dist(cls, situation: Vector[_Value]) -> Vector[_Value]:
        return Vector.new(situation.value_type, [0] * len(situation))
        # mean = situation.mean()
        # stddev = situation.stddev(mean)
        # return Vector.sample_gaussian(float, len(situation)).cast(situation.value_type) * stddev + mean

    @classmethod
    def cover(cls, situation: Vector[_Value], *, wildcard_prob: Probability = None,
              interval_radius_dist: Callable[[Vector[_Value]], Vector[_Value]] = None,
              **kwargs) -> 'IntervalCondition[_Value]':
        if wildcard_prob is None:
            wildcard_prob = 1 - 1 / len(situation)
        wildcard_prob = Probability(wildcard_prob)
        # p(wildcard) = p(top wild and bottom wild) = p(top wild) p(bottom wild) = prob(half wild) ** 2
        # p(half wild) = p(wildcard) ** 0.5
        half_wild_prob = wildcard_prob ** 0.5
        tops = sparse_ops.cover(situation, half_wild_prob)
        bottoms = sparse_ops.cover(situation, half_wild_prob)
        bound_indices = tops.keys() | bottoms.keys()
        interval_radius_dist = interval_radius_dist or cls.default_interval_radius_dist
        interval_radii = abs(interval_radius_dist(situation))
        for index, interval_radius in zip(bound_indices, interval_radii):
            if index in tops:
                tops[index] += interval_radius
            if index in bottoms:
                bottoms[index] -= interval_radius
        return cls(len(situation), bottoms, tops)

    def __init__(self, length: int, bottoms: Mapping[int, _Value], tops: Mapping[int, _Value]):
        self._length = length
        self._bottoms = dict(bottoms)
        self._tops = dict(tops)
        assert all(0 <= index < length for index in self._bottoms)
        assert all(0 <= index < length for index in self._tops)

    def mutate(self, *, situation: Vector[_Value] = None, condition: 'IntervalCondition' = None,
               mutation_probability: Probability = None, **kwargs) -> 'IntervalCondition':
        if mutation_probability is None:
            mutation_probability = 1 / self._length
        mutation_probability = Probability(mutation_probability)
        # p(mutation) = p(top mutation or bottom mutation) = 1 - p(~top mutation) p(~bottom mutation)
        #   = 1 - p(~half mutation) ** 2 = 1 - (1 - p(half mutation)) ** 2
        # p(half mutation) = 1 - (1 - p(mutation)) ** 0.5
        half_mutation_probability = 1 - (1 - mutation_probability) ** 0.5
        tops = sparse_ops.mutate(self._length, self._tops, situation=situation,
                                 parent2=None if condition is None else condition._tops,
                                 mutation_probability=half_mutation_probability)
        bottoms = sparse_ops.mutate(self._length, self._bottoms, situation=situation,
                                    parent2=None if condition is None else condition._bottoms,
                                    mutation_probability=half_mutation_probability)
        return type(self)(self._length, bottoms, tops)

    def cross_with(self, other: 'IntervalCondition[_Value]', *,
                   points: int = 2, **kwargs) -> 'tuple[IntervalCondition[_Value], IntervalCondition[_Value]]':
        linear = kwargs.get('linear', False)
        scatter = kwargs.get('scatter', False)
        interval = kwargs.get('interval', False)
        blend = kwargs.get('blend', False)
        swap = kwargs.get('swap', False)
        sort = kwargs.get('sort', False)
        if not (linear or scatter or interval or blend or swap or sort):
            linear = scatter = interval = blend = swap = sort = 1
        styles = []
        for index, style in enumerate([linear, scatter, interval, blend, swap, sort]):
            if style:
                styles.append(index)
        style = random.choice(styles) if len(styles) > 1 else styles[0]
        if style == 0:  # linear
            joint1 = {index: (self._tops.get(index, None), self._bottoms.get(index, None))
                      for index in self._tops.keys() | self._bottoms.keys()}
            joint2 = {index: (other._tops.get(index, None), other._bottoms.get(index, None))
                      for index in other._tops.keys() | other._bottoms.keys()}
            child_joints = sparse_ops.cross_with(self._length, joint1, joint2, points=points)
        elif style == 1:  # scatter
            joint1 = {}
            joint2 = {}
            proportion = random.random()
            for index in self._tops.keys() | self._bottoms.keys() | other._tops.keys() | other._bottoms.keys():
                pair1 = (self._tops.get(index, None), self._bottoms.get(index, None))
                pair2 = (other._tops.get(index, None), other._bottoms.get(index, None))
                if random.random() <= proportion:
                    joint1[index] = pair1
                    joint2[index] = pair2
                else:
                    joint1[index] = pair2
                    joint2[index] = pair1
            child_joints = (joint1, joint2)
        elif style == 2:  # interval
            joint1 = {}
            joint2 = {}
            for index in self._tops.keys() | self._bottoms.keys() | other._tops.keys() | other._bottoms.keys():
                inner_top = self._tops.get(index, None)
                outer_top = other._tops.get(index, None)
                if inner_top is None or (outer_top is not None and inner_top > outer_top):
                    inner_top, outer_top = outer_top, inner_top
                inner_bottom = self._bottoms.get(index, None)
                outer_bottom = other._bottoms.get(index, None)
                if inner_bottom is None or (outer_bottom is not None and inner_bottom < outer_bottom):
                    inner_bottom, outer_bottom = outer_bottom, inner_bottom
                if inner_bottom is not None and inner_top is not None and inner_bottom > inner_top:
                    if random.randrange(2):
                        inner_top, outer_top = outer_top, inner_top
                    else:
                        inner_bottom, outer_bottom = outer_bottom, inner_top
                assert outer_bottom is None or outer_top is None or outer_bottom <= outer_top
                assert inner_bottom is None or inner_top is None or inner_bottom <= inner_top
                joint1[index] = (inner_top, inner_bottom)
                joint2[index] = (outer_top, outer_bottom)
            child_joints = (joint1, joint2)
        elif style == 3:  # blend
            joint1 = {}
            joint2 = {}
            for index in self._tops.keys() | self._bottoms.keys() | other._tops.keys() | other._bottoms.keys():
                bottom1 = self._bottoms.get(index, None)
                top1 = self._tops.get(index, None)
                bottom2 = other._bottoms.get(index, None)
                top2 = other._tops.get(index, None)
                if bottom1 is not None and top1 is not None and random.randrange(2):
                    bottom1, top1 = top1, bottom1
                if bottom2 is not None and top2 is not None and random.randrange(2):
                    bottom2, top2 = top2, bottom2
                if random.randrange(2):
                    bottom1, bottom2 = bottom2, bottom1
                if random.randrange(2):
                    top1, top2 = top2, top1
                joint1[index] = (top1, bottom1)
                joint2[index] = (top2, bottom2)
            child_joints = (joint1, joint2)
        elif style == 4:  # swap
            joint1 = {}
            joint2 = {}
            for index in self._tops.keys() | self._bottoms.keys() | other._tops.keys() | other._bottoms.keys():
                top1 = self._tops.get(index, None)
                bottom1 = self._bottoms.get(index, None)
                top2 = other._tops.get(index, None)
                bottom2 = other._bottoms.get(index, None)
                joint1[index] = (top1, bottom2)
                joint2[index] = (top2, bottom1)
            child_joints = (joint1, joint2)
        elif style == 5:  # sort
            joint1 = {}
            joint2 = {}
            for index in self._tops.keys() | self._bottoms.keys() | other._tops.keys() | other._bottoms.keys():
                top1 = self._tops.get(index, math.inf)
                bottom1 = self._bottoms.get(index, -math.inf)
                top2 = other._tops.get(index, math.inf)
                bottom2 = other._bottoms.get(index, -math.inf)
                values = [top1, bottom1, top2, bottom2]
                values.sort()
                bottom1, top1, bottom2, top2 = values
                if top1 == -math.inf or bottom2 == math.inf:
                    bottom2, top1 = top1, bottom2
                assert bottom1 <= top1
                assert bottom2 <= top2
                if not math.isfinite(bottom1):
                    bottom1 = None
                if not math.isfinite(bottom2):
                    bottom2 = None
                if not math.isfinite(top1):
                    top1 = None
                if not math.isfinite(top2):
                    top2 = None
                joint1[index] = (top1, bottom1)
                joint2[index] = (top2, bottom2)
            child_joints = (joint1, joint2)
        else:
            assert False, "Unknown crossover style"
        children = []
        for child_joint in child_joints:
            tops = {}
            bottoms = {}
            for index, (top, bottom) in child_joint.items():
                if top is not None:
                    if bottom is not None:
                        if bottom <= top:
                            tops[index] = top
                            bottoms[index] = bottom
                        else:
                            tops[index] = bottom
                            bottoms[index] = top
                    else:
                        tops[index] = top
                elif bottom is not None:
                    bottoms[index] = bottom
            children.append(type(self)(self._length, bottoms, tops))
        child1, child2 = children
        return child1, child2

    def expected_value(self) -> None | Probability:
        return None

    def generalizes(self, other: 'Condition[_Value]') -> None | bool:
        if not isinstance(other, IntervalCondition):
            return None
        return (sparse_ops.generalizes(self._tops, other._tops, operator.ge) and
                sparse_ops.generalizes(self._bottoms, other._bottoms, operator.le))

    def matches(self, situation: BoolVector) -> bool | Probability:
        return (sparse_ops.matches(self._tops, situation, operator.ge) and
                sparse_ops.matches(self._bottoms, situation, operator.le))

    def concat(self, *others: 'IntervalCondition[_Value]') -> 'IntervalCondition[_Value]':
        tops = self._tops.copy()
        bottoms = self._bottoms.copy()
        length = self._length
        for other in others:
            for index, element in other._tops.items():
                tops[length + index] = element
            for index, element in other._bottoms.items():
                bottoms[length + index] = element
            length += other._length
        return type(self)(length, bottoms, tops)

    def equivalent(self, other: 'Condition[_Value]') -> bool:
        if type(self) is type(other):
            other: IntervalCondition
            return self._bottoms == other._bottoms and self._tops == other._tops
        return super().equivalent(other)

    def __hash__(self) -> int:
        if self._hash_value is None:
            self._hash_value = hash((self._length, tuple(sorted(self._tops.items())),
                                     tuple(sorted(self._bottoms.items()))))
        return self._hash_value

    def __str__(self) -> str:
        indices = self._tops.keys() | self._bottoms.keys()
        if not indices:
            return '*'
        pieces = []
        for index in sorted(indices):
            if index in self._bottoms:
                lower = self._bottoms[index]
                if index in self._tops:
                    upper = self._tops[index]
                    piece = f"{lower} <= x[{index}] <= {upper}"
                else:
                    piece = f"{lower} <= x[{index}]"
            else:
                upper = self._tops[index]
                piece = f"x[{index}] <= {upper}"
            pieces.append(piece)
        return ' and '.join(pieces)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._length}, {self._bottoms:r}, {self._tops:r})"

    def __iter__(self) -> Iterator[tuple[_Value | None, _Value | None]]:
        for index in range(self._length):
            yield self._bottoms.get(index, None), self._tops.get(index, None)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._bottoms.get(item, None), self._tops.get(item, None)
        else:
            assert isinstance(item, slice)
            indices = range(*item.indices(self._length))
            bottoms = {index - indices.start: value for index, value in self._bottoms.items() if index in indices}
            tops = {index - indices.start: value for index, value in self._tops.items() if index in indices}
            return type(self)(len(indices), bottoms, tops)

    def __and__(self, other: 'IntervalCondition[_Value]') -> 'IntervalCondition[_Value]':
        if not isinstance(other, IntervalCondition):
            return NotImplemented
        bottoms = self._bottoms.copy()
        for index, value in other._bottoms.items():
            if index in bottoms:
                bottoms[index] = max(bottoms[index], value)
            else:
                bottoms[index] = value
        tops = self._tops.copy()
        for index, value in other._tops.items():
            if index in tops:
                tops[index] = min(tops[index], value)
            else:
                tops[index] = value
        for index in bottoms.keys() & tops.keys():
            if bottoms[index] > tops[index]:
                # Make sure the condition is satisfiable
                del bottoms[index]
                del tops[index]
        return type(self)(self._length, bottoms, tops)

    def __or__(self, other: 'IntervalCondition') -> 'IntervalCondition':
        if not isinstance(other, IntervalCondition):
            return NotImplemented
        bottoms = {}
        for index in self._bottoms.keys() & other._bottoms.keys():
            bottoms[index] = min(self._bottoms[index], other._bottoms[index])
        tops = {}
        for index in self._tops.keys() & other._tops.keys():
            tops[index] = max(self._tops[index], other._tops[index])
        return type(self)(self._length, bottoms, tops)
