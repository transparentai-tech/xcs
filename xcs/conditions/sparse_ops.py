import random
from typing import TypeVar, Hashable, Callable, Sequence

from vectorface import Vector
from xcs.probabilities import Probability
from xcs.conditions import set_ops

_Value = TypeVar('_Value', bound=Hashable)


def cover(situation: Sequence[_Value], wildcard_prob: Probability = None) -> dict[int, _Value]:
    if wildcard_prob is None:
        wildcard_prob = 1 - 1 / len(situation)
    wildcard_prob = Probability(wildcard_prob)
    inclusion_count = random.binomialvariate(len(situation), 1 - wildcard_prob)
    return {index: situation[index] for index in random.sample(range(len(situation)), inclusion_count)}


def mutate(length: int, parent1: dict[int, _Value], *, situation: Sequence[_Value] = None,
           parent2: dict[int, _Value] = None, mutation_probability: Probability = None) -> dict[int, _Value]:
    if mutation_probability is None:
        mutation_probability = 1 / length
    mutation_probability = Probability(mutation_probability)
    mutation_count = random.binomialvariate(length, mutation_probability)
    if not mutation_count:
        return parent1

    child = parent1.copy()
    options = [None]
    if parent2 is not None:
        options.append(parent2)
    if situation is not None:
        options.append(situation)
    for index in random.sample(range(length), mutation_count):
        selected = random.choice(options)
        if selected is None or (selected is parent2 and index not in parent2):
            if index in child:
                del child[index]
        else:
            other_value = selected[index]
            if index in child and random.randrange(2):
                current_value = child[index]
                center = (current_value + other_value) / 2
                radius = abs(current_value - other_value) / 2
                child[index] = center + random.randint(-2, 2) * radius
            else:
                child[index] = other_value

    return child


def cross_with(length: int, parent1: dict[int, _Value], parent2: dict[int, _Value], *,
               points: int = 2) -> tuple[dict[int, _Value], dict[int, _Value]]:
    points = sorted(set(random.sample(range(length + 1), points)) | {0, length})
    indices1 = sorted(parent1)
    next1 = 0
    indices2 = sorted(parent2)
    next2 = 0
    child1 = {}
    child2 = {}
    for end in points:
        while next1 < len(indices1):
            index = indices1[next1]
            if index < end:
                child1[index] = parent1[index]
                next1 += 1
            else:
                break
        while next2 < len(indices2):
            index = indices2[next2]
            if index < end:
                child2[index] = parent2[index]
                next2 += 1
            else:
                break
        child1, child2 = child2, child1
    # assert child1.keys() & child2.keys() == parent1.keys() & parent2.keys()
    # assert child1.keys() | child2.keys() == parent1.keys() | parent2.keys()
    return child1, child2


def generalizes(general: dict[int, _Value], specific: dict[int, _Value],
                constraint: Callable[[_Value, _Value], bool]) -> bool:
    for index, value in general.items():
        if index not in specific or not constraint(value, specific[index]):
            return False
    return True


def matches(elements: dict[int, _Value], situation: Sequence[_Value],
            constraint: Callable[[_Value, _Value], bool]) -> bool:
    for index, value in elements.items():
        if not constraint(value, situation[index]):
            return False
    return True


def delta(general: dict[int, _Value], specific: dict[int, _Value]) -> dict[int, _Value]:
    result = {}
    for index, value in specific.items():
        if general.get(index, None) != value:
            result[index] = value
    return result
