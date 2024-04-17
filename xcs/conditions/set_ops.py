import random
from typing import AbstractSet

from vectorface import Vector
from xcs.probabilities import Probability


def cover(length: int, wildcard_prob: Probability = None) -> set[int]:
    if wildcard_prob is None:
        # Choose wildcard_prob so that E[|result|] = 1
        wildcard_prob = 1 - 1 / length
    wildcard_prob = Probability(wildcard_prob)
    count = random.binomialvariate(length, 1 - wildcard_prob)
    return set(random.sample(range(length), count))


def mutate(length: int, parent1: AbstractSet[int], *, situation: bool = False, parent2: AbstractSet[int] = None,
           mutation_probability: Probability = None) -> set[int]:
    if mutation_probability is None:
        mutation_probability = 1 / length
    mutation_probability = Probability(mutation_probability)
    flags = Vector.sample_categorical(bool, length, weights={False: 1 - mutation_probability,
                                                             True: mutation_probability})
    child = set(parent1)
    if not flags.any():
        return child
    if parent2 is None:
        for index, flag in enumerate(flags):
            if flag:
                child.discard(index)
    else:
        for index, flag in enumerate(flags):
            if not flag:
                continue
            can_remove = index in child
            can_add = situation or index in parent2
            if can_remove and (not can_add or random.randrange(2)):
                child.discard(index)
            elif can_add:
                child.add(index)
    return child


def cross_with(length: int, parent1: AbstractSet[int], parent2: AbstractSet[int], *,
               points: int = 2) -> tuple[set[int], set[int]]:
    points = sorted(set(random.sample(range(length + 1), points)) | {0, length})
    range_map = []
    which = random.randrange(2)
    for start, end in zip(points[:-1], points[1:]):
        range_map.append((range(start, end), which))
        which = not which
    child1 = set()
    child2 = set()
    children = (child1, child2)
    for index in parent1:
        for r, w in range_map:
            if index in r:
                children[w].add(index)
    children = (child2, child1)
    for index in parent2:
        for r, w in range_map:
            if index in r:
                children[w].add(index)
    assert child1 | child2 == parent1 | parent2
    assert child1 & child2 == parent2 & parent2
    return child1, child2
