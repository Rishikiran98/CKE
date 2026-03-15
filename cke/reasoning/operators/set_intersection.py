"""Deterministic set intersection helper."""


def set_intersection(left: set, right: set) -> set:
    return set(left).intersection(set(right))
