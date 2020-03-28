#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_parameters.py

"""Test the ``Params`` class."""

# pylint: disable=missing-docstring,invalid-name,redefined-outer-name
# pylint: disable=not-an-iterable

import os

import yaml
import pytest

from nets.parameters import Params


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_KEY = "params"


@pytest.fixture
def x():
    return {
        DATA_KEY: {"p1": 0, 0: 0, 1: 0},
        "c1": {DATA_KEY: {0: 1, 1: 1}},
        "c2": {DATA_KEY: {0: 1, 1: 1}, 0: {}, "cc2": {0: {DATA_KEY: {}}}},
        "c3": {0: {}},
    }


@pytest.fixture
def t(x):
    return Params(x)


def test_init(x):
    Params(x)


def test_eq(t):
    # Identity
    assert t == t
    # No data, no children
    assert Params() == Params()
    # No data, same children
    assert Params({0: {}, 1: {}}) == Params({0: {}, 1: {}})
    # No data, different children
    assert Params({0: {}, 1: {}}) != Params({1: {}, 2: {}})
    # Same data, no children
    assert Params({DATA_KEY: {0: 0}}) == Params({DATA_KEY: {0: 0}})
    # Same data, same children
    assert Params({DATA_KEY: {0: 0}, 0: {}, 1: {}}) == Params(
        {DATA_KEY: {0: 0}, 0: {}, 1: {}}
    )
    # Same data, different children
    assert Params({DATA_KEY: {0: 0}, 0: {}}) != Params({DATA_KEY: {0: 0}, 0: {}, 1: {}})
    # Different data, no children
    assert Params({DATA_KEY: {0: 0}}) != Params({DATA_KEY: {0: 1}})
    # Different data, same children
    assert Params({DATA_KEY: {0: 0}, 0: {}, 1: {}}) != Params(
        {DATA_KEY: {0: 1}, 0: {}, 1: {}}
    )
    # Different data, different children
    assert Params({DATA_KEY: {0: 0}, 0: {}, 1: {}}) != Params(
        {DATA_KEY: {0: 1}, 1: {}, 2: {}}
    )


def test_iter(x, t):
    assert set(t) == set(x[DATA_KEY])
    assert set(t.c["c2"].c["cc2"].c[0]) == set.union(
        *[set(x[DATA_KEY]), set(x["c2"][DATA_KEY]), set(x["c2"]["cc2"][0]),]
    ) - set([DATA_KEY])


def test_missing(t):
    child = t.c["c2"].c["cc2"].c[0]
    assert child["p1"] == 0


def test_contains(t):
    assert "p1" in t
    assert "p1" in t.c["c2"].c["cc2"]


def test_get_node(t):
    # Tuple
    assert t.get_node("c2", "cc2") == t.c["c2"].c["cc2"]
    # Name
    assert t.get_node("c2") == t.c["c2"]


def test_getitem(t):
    # Tuple
    assert t[("c2", "cc2", 0)] == Params({})
    # Key
    assert t["p1"] == 0


def test_ancestors(t):
    leaf = t.c["c2"].c["cc2"].c[0]
    assert list(leaf.ancestors()) == [
        leaf,
        leaf.p,
        leaf.p.p,
        leaf.p.p.p,
    ]


def test_keys(t):
    assert set(t.keys()) == set(["p1", 0, 1])


def test_num_children(t):
    assert t.num_children == 3


def test_leaves(t):
    assert list(t.leaves()) == [
        Params({DATA_KEY: {0: 1, 1: 1}}),
        Params({}),
        Params({}),
        Params({}),
    ]


def test_named_leaves(t):
    assert list(t.named_leaves()) == [
        ("c1", Params({DATA_KEY: {0: 1, 1: 1}})),
        (0, Params({})),
        (0, Params({})),
        (0, Params({})),
    ]


def test_load(x, t):
    path = os.path.join(HERE, "tree.yml")
    with open(path, "wt") as f:
        yaml.dump(x, f)
    assert Params.load(path) == t


@pytest.fixture
def trees():
    return [
        Params({DATA_KEY: {0: 0}}),
        Params({DATA_KEY: {0: 1, 1: 1}, 0: {DATA_KEY: {0: 0}}}),
        Params({DATA_KEY: {0: 0}, 0: {DATA_KEY: {0: 1}}, 1: {}}),
        Params({DATA_KEY: {0: 0, 2: 2}, 1: {DATA_KEY: {0: 0}}, 2: {}}),
        Params({DATA_KEY: {0: 0, 1: 2}, "hi": {}}),
    ]


@pytest.fixture
def inheritance_trees():
    return [
        Params({"intermediate1": {DATA_KEY: {"key": "intermediate"}}}),
        Params(
            {
                DATA_KEY: {"key": "root"},
                "intermediate1": {"leaf1": {DATA_KEY: {"key": "leaf"}}, "leaf2": {}},
                "intermediate2": {"leaf1": {}},
            }
        ),
    ]


@pytest.fixture
def merged(trees):
    return Params.merge(*trees)


def test_merge(trees):
    merged = Params.merge(*trees)
    # Check that it's a new object
    for tree in trees:
        assert merged != tree
    key = "NEW_KEY"
    merged[key] = 0
    for tree in trees:
        assert key not in trees[0]
    # Check that data was merged
    for key in [0, 1, 2]:
        assert key in merged
    for name in [0, 1, 2, "hi"]:
        assert name in merged.c
    assert 0 in merged.c[1]


def test_merge_inheritance(inheritance_trees):
    merged = Params.merge(*inheritance_trees)
    assert merged.c["intermediate1"].c["leaf1"]["key"] == "leaf"
    assert merged.c["intermediate1"].c["leaf2"]["key"] == "intermediate"
    assert merged.c["intermediate2"].c["leaf1"]["key"] == "root"


def test_merge_precedence(merged):
    assert merged[0] == 0
    assert merged[1] == 1
    assert merged.c[0][0] == 0


def test_merge_access_traversal(merged):
    assert merged.c[0][2] == 2


def test_merge_contains(merged):
    assert 2 in merged.c[0]
