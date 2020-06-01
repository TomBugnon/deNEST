#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_parameters.py

"""Test the ``Params`` class."""

# pylint: disable=missing-docstring,invalid-name,redefined-outer-name
# pylint: disable=not-an-iterable

import tempfile
from pathlib import Path

import pytest

from denest.parameters import ParamsTree

assert len(ParamsTree.DATA_KEYS) == 2
DATA_KEYS = ParamsTree.DATA_KEYS
DK1, DK2 = DATA_KEYS


@pytest.fixture
def x():
    return {
        DK1: {"c0_1": "1", "a": "c0_a1", "b": "c0_b1"},
        DK2: {"c0_2": "2", "a": "c0_a2", "b": "c0_b2"},
        "c1": {DK1: {"a": "c1_a1", "b": "c1_b1"}, DK2: {"a": "c1_a2", "b": "c1_b2"},},
        "c2": {
            DK1: {"a": "c2_a1", "b": "c2_b1"},
            # DK2: {'a': 'c2_a2', 'b': 'c2_b2'},
            "cc2": {"ccc2": {DK1: {"a": "ccc2_a1"}, DK2: {"b": "ccc2_b2"},}},
        },
        "c3": {"cc3": {}},
    }


@pytest.fixture
def t(x):
    return ParamsTree(x)


def test_init(x):
    ParamsTree(x)


def test_eq(t):
    # Identity
    assert t == t
    # No data, no children
    assert ParamsTree() == ParamsTree()
    # No data, same children
    assert ParamsTree({0: {}, 1: {}}) == ParamsTree({0: {}, 1: {}})
    # No data, different children
    assert ParamsTree({0: {}, 1: {}}) != ParamsTree({1: {}, 2: {}})
    # Same data, no children
    assert ParamsTree({DK1: {0: 0}}) == ParamsTree({DK1: {0: 0}})
    assert ParamsTree({DK2: {0: 0}}) == ParamsTree({DK2: {0: 0}})
    assert ParamsTree({DK1: {0: 0}, DK2: {1: 1}}) == ParamsTree(
        {DK1: {0: 0}, DK2: {1: 1}}
    )
    # Same data, same children
    assert ParamsTree({DK1: {0: 0}, 0: {}, 1: {}}) == ParamsTree(
        {DK1: {0: 0}, 0: {}, 1: {}}
    )
    assert ParamsTree({DK2: {0: 0}, 0: {}, 1: {}}) == ParamsTree(
        {DK2: {0: 0}, 0: {}, 1: {}}
    )
    # Same data, different children
    assert ParamsTree({DK1: {0: 0}, 0: {}}) != ParamsTree({DK1: {0: 0}, 0: {}, 1: {}})
    # Different data, no children
    assert ParamsTree({DK1: {0: 0}}) != ParamsTree({DK1: {0: 1}})
    # Different data, same children
    assert ParamsTree({DK1: {0: 0}, 0: {}, 1: {}}) != ParamsTree(
        {DK1: {0: 1}, 0: {}, 1: {}}
    )
    # Different data, different children
    assert ParamsTree({DK1: {0: 0}, 0: {}, 1: {}}) != ParamsTree(
        {DK1: {0: 1}, 1: {}, 2: {}}
    )


def test_access(t):
    # Access data as attributes, from dict(self) or from self.data
    for k in DATA_KEYS:
        # For root
        assert t[k] == getattr(t, k)
        assert t.data[k] == t[k]
        # For leaf
        leaf = t.children["c2"].children["cc2"].children["ccc2"]
        assert leaf[k] == getattr(leaf, k)
        assert leaf.data[k] == leaf[k]


def test_missing(t):
    # Empty leaf and intermediate nodes. Inherits root
    child = t.children["c3"].children["cc3"]
    assert child[DK1]["c0_1"] == "1"
    assert child[DK1]["a"] == "c0_a1"
    assert child[DK1]["b"] == "c0_b1"
    assert child[DK2]["c0_2"] == "2"


def test_contains(t):
    assert "c0_1" in t[DK1]
    assert "c0_1" in t.children["c2"][DK1]
    assert "c0_1" in t.children["c2"].children["cc2"][DK1]
    assert "c0_1" in t.children["c2"].children["cc2"].children["ccc2"][DK1]


def test_ancestors(t):
    leaf = t.children["c2"].children["cc2"].children["ccc2"]
    # List of ancestors does NOT include self
    assert list(leaf.ancestors()) == [
        leaf.parent,
        leaf.parent.parent,
        leaf.parent.parent.parent,
    ]
    assert list(leaf.ancestors()) == [
        t.children["c2"].children["cc2"],
        t.children["c2"],
        t,
    ]


def test_keys(t):
    assert set(t[DK1].keys()) == set(["c0_1", "a", "b"])
    assert set(t[DK2].keys()) == set(["c0_2", "a", "b"])


def test_leaves(t):
    assert t.leaves() == [
        ParamsTree(
            {DK1: {"a": "c1_a1", "b": "c1_b1"}, DK2: {"a": "c1_a2", "b": "c1_b2"},}
        ),  # c1
        ParamsTree({DK1: {"a": "ccc2_a1"}, DK2: {"b": "ccc2_b2"},}),  # ccc2
        ParamsTree({}),  # cc3
    ]


def test_named_leaves(t):
    assert t.named_leaves() == [
        (
            "c1",
            ParamsTree(
                {DK1: {"a": "c1_a1", "b": "c1_b1"}, DK2: {"a": "c1_a2", "b": "c1_b2"},}
            ),
        ),  # c1
        ("ccc2", ParamsTree({DK1: {"a": "ccc2_a1"}, DK2: {"b": "ccc2_b2"},})),  # ccc2
        ("cc3", ParamsTree({})),  # cc3
    ]


def test_read_write(x, t):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tree.yml"
        t.write(path)
        assert ParamsTree.read(path) == t


def test_asdict(t):
    assert t == ParamsTree(t.asdict())


@pytest.fixture
def trees():
    return [
        ParamsTree({DK1: {0: 0}, 0: {DK1: {}},}),
        ParamsTree(
            {DK1: {0: 1, 1: 1}, DK2: {0: 1, 1: 1}, 0: {DK1: {"test_key": "test_value"}}}
        ),
        ParamsTree(
            {DK1: {0: 0}, DK2: {0: 0}, 0: {DK1: {"test_key": "overriden_value"}}, 1: {}}
        ),
        ParamsTree(
            {DK1: {0: 0, 2: 2, 3: 3}, DK2: {0: 0, 2: 2, 3: 3}, 1: {DK1: {0: 0}}, 2: {}}
        ),
        ParamsTree({DK1: {0: 0, 1: 2, "hello": 0}, DK2: {"hello": 0}, "hi": {}}),
    ]


@pytest.fixture
def inheritance_trees():
    return [
        ParamsTree(
            {"intermediate1": {DK1: {"key": "intermediate", "NEW_KEY": "intermediate"}}}
        ),
        ParamsTree(
            {
                DK1: {"key": "root"},
                "intermediate1": {
                    DK1: {"key": "overriden"},
                    "leaf1": {DK1: {"key": "leaf"}},
                    "leaf2": {},
                },
                "intermediate2": {"leaf1": {DK1: {"other": "other"}}},
            }
        ),
    ]


@pytest.fixture
def merged(trees):
    return ParamsTree.merge(*trees)


def test_merge(trees):
    merged = ParamsTree.merge(*trees)
    # Check that it's a new object
    for tree in trees:
        assert merged != tree
    key = "NEW_KEY"
    merged[DK1][key] = 0
    merged[DK2][key] = 0
    print(merged)
    print(trees[0])
    for tree in trees:
        assert key not in tree[DK1]
        assert key not in tree[DK2]
    # Check that data was merged
    for k in DATA_KEYS:
        for key in [0, 1, 2, "hello"]:
            assert key in merged[k]
    for name in [0, 1, 2, "hi"]:
        assert name in merged.children
    assert "test_key" in merged.children[0][DK1]
    assert merged.children[0][DK1]["test_key"] == "test_value"


def test_merge_inheritance(inheritance_trees):
    merged = ParamsTree.merge(*inheritance_trees)
    assert merged.children["intermediate1"].children["leaf1"][DK1]["key"] == "leaf"
    assert (
        merged.children["intermediate1"].children["leaf2"][DK1]["key"] == "intermediate"
    )
    assert (
        merged.children["intermediate1"].children["leaf1"][DK1]["NEW_KEY"]
        == "intermediate"
    )
    assert merged.children["intermediate2"].children["leaf1"][DK1]["key"] == "root"


def test_merge_precedence(merged):
    assert merged[DK1][0] == 0
    assert merged[DK1][1] == 1
    assert merged[DK2][0] == 1
    assert merged.children[0][DK1]["test_key"] == "test_value"


def test_merge_access_traversal(merged):
    assert merged[DK2][3] == 3
    assert merged[DK2]["hello"] == 0
    assert merged.children[1][DK1][0] == 0


def test_merge_children(merged):
    assert set(merged.children.keys()) == set(["hi", 0, 1, 2])

