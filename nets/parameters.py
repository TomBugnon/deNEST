#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# parameters.py

"""Provide the ``ParamsTree`` class."""

from collections import ChainMap, UserDict
from collections.abc import Mapping
from pprint import pformat

import yaml

_MAX_LINES = 30


class InvalidTreeError(ValueError):
    """Raised when a mapping is not a valid ``ParamsTree``."""

    pass


class DeepChainMap(ChainMap):
    """Variant of ChainMap that allows direct updates to inner scopes."""

    def __setitem__(self, key, value):
        for mapping in self.maps:
            if key in mapping:
                mapping[key] = value
                return
        self.maps[0][key] = value

    def __delitem__(self, key):
        for mapping in self.maps:
            if key in mapping:
                del mapping[key]
                return
        raise KeyError(key)


class ParamsTree(UserDict):
    """A tree of nodes that inherit and override ancestors' data.

    A tree is created from a tree-like mapping. The key-value pairs
    containing the node's data are defined in ``self.DATA_KEYS`` (``params``
    and ``nest_params``). Data is inherited from ancestors for each of those
    keys.
    Note that the order of traversals is undefined.

    Keyword Args:
        mapping (Mapping): A dictionary-like object that maps names to
            children, but with special key-value pairs containing the node's
            data. Defaults to an empty dictionary.
        parent (ParamsTree): The parent tree. Data from each of the data keys is
            inherited from ancestors.

    Attributes:
        name (str | None): Name of the node.
        children (dict): A dictionary of named children.
        parent (type(self)): The parent of this node.
        data (dict): Dictionary containing the data accessible from this node
            (i.e., the node's data and that inherited from its parents). Keys
            are values of ``self.DATA_KEYS`` (``'params'`` and
            ``'nest_params'``).
        params, nest_params (dict-like): Contain the node data. Syntactic sugar
            to allow access to the node's data.
    """

    DATA_KEYS = ["params", "nest_params"]

    def __init__(self, mapping=None, parent=None, name=None, validate=True):
        # Parent
        self._parent = parent
        # Name
        self._name = name
        # Validate mapping
        if mapping is None:
            # No data & no children
            mapping = {}
        if validate:
            mapping = self.validate(mapping)
        # Data internal to this node. Keys are those specified by DATA_KEYS.
        # Each data key contains an empty dictionary by default.
        self._data = {key: mapping.get(key, {}) for key in self.DATA_KEYS}
        # Accessible data (inherits from parents)
        super().__init__(self._data)
        self.data = {
            key: DeepChainMap(
                self.data[key],
                *(ancestor.node_data[key] for ancestor in self.ancestors()),
            )
            for key in self.DATA_KEYS
        }
        # Children
        self._children = {
            key: ParamsTree(value, parent=self, name=key)
            for key, value in mapping.items()
            if key not in self.DATA_KEYS
        }
        # Syntactic sugar to allow data keys to be accessed as attributes
        for data_key, value in self.data.items():
            setattr(self, data_key, value)

    @property
    def node_data(self):
        """The data associated with this node (not inherited from parents)."""
        return self._data

    def __eq__(self, other):
        """Nodes are equal when their (own) data and children are equal.

        Note that parents can differ. We compare node's own data rather than
        inherited data.
        """
        return self.node_data == other.node_data and self.children == other.children

    @property
    def parent(self):
        """This node's parent. ``None`` if node is the root."""
        return self._parent

    @property
    def name(self):
        """This name of this node."""
        return self._name

    @property
    def children(self):
        """A dictionary of this node's children"""
        return self._children

    def ancestors(self):
        """Return a list of ancestors of this node.

        Doesn't include self. More recent ancestors appear earlier in the list.
        """
        ancestors = []
        node = self
        while True:
            if node.parent is None:
                break
            ancestors.append(node.parent)
            node = node.parent
        return ancestors

    def leaves(self, root=True):
        """Return a list of leaf nodes of the tree.

        Traversal order is not defined. If root is False and there is not
        children, returns an empty list
        """
        leaves = []
        # Recursive case: not a leaf
        for child in self.children.values():
            leaves.extend(child.leaves(root=True))
        # Base case: leaf
        if not leaves and root:
            leaves.append(self)
        return leaves

    def named_leaves(self, root=True):
        """Return list of ``(<name>, <node>)`` tuples for all the leaves.

        Traversal order is not defined. If root is False and there is not
        children, returns an empty list
        """
        return [(leaf.name, leaf) for leaf in self.leaves(root=root)]

    @classmethod
    def merge(cls, *trees, parent=None, name=None):
        """Merge trees into a new tree. Earlier trees take precedence.

        If a node exists at the same location in more than one tree and these
        nodes have duplicate keys, the values from the nodes in the trees that
        appear earlier in the argument list will take precedence over those
        from later trees.

        Equivalent nodes' data is merged horizontally before hierarchical
        inheritance.
        """
        # Merge node's own data
        data = {
            key: dict(ChainMap(*(tree.node_data[key] for tree in trees)))
            for key in cls.DATA_KEYS
        }
        # Initialize tree with node data and no children
        merged = cls(mapping=data, parent=parent, name=name)
        # Merge children recursively, passing parent so that children inherit
        # merged node's data
        children = set.union(*(set(tree.children) for tree in trees))
        merged._children = {
            name: cls.merge(
                *(tree.children[name] for tree in trees if name in tree.children),
                parent=merged,
                name=name,
            )
            for name in children
        }
        return merged

    def copy(self):
        """Copy this ``ParamsTree``."""
        return ParamsTree(self.asdict())

    def asdict(self):
        """Convert this ``ParamsTree`` to a nested dictionary."""
        return {
            **{key: dict(value) for key, value in self.node_data.items()},
            **{name: child.asdict() for name, child in self.children.items()},
        }

    def __str__(self):
        return yaml.dump(self.asdict(), sort_keys=False)

    def __repr__(self):
        lines = str(self).split("\n")
        n = len(lines)
        if n >= _MAX_LINES:
            lines = (
                lines[: (_MAX_LINES // 2)]
                + [f"\n  ... [{n - _MAX_LINES} lines] ...\n"]
                + lines[-(_MAX_LINES // 2) :]
            )
        parent_str = f"'{self.parent.name}'" if self.parent is not None else None
        return f"ParamsTree(name='{self.name}', parent={parent_str})\n" + "\n".join(
            ["  " + line for line in lines]
        )

    @classmethod
    def read(cls, path):
        """Load a YAML representation of a tree from disk."""
        with open(path, "rt") as f:
            return cls(yaml.load(f, Loader=yaml.SafeLoader))

    def write(self, path):
        """Write a YAML representation of a tree to disk."""
        with open(path, "wt") as f:
            yaml.dump(self.asdict(), f, default_flow_style=False, sort_keys=False)
        return path

    def validate(self, mapping, path=None):
        """Check that a mapping is a valid ``ParamsTree``."""
        if path is None:
            path = list()
        if mapping:
            self._validate(mapping, path)
            for name, child in mapping.items():
                # Validate data
                if name in self.DATA_KEYS:
                    self._validate(child, path)
                # Validate children
                else:
                    self.validate(child, path + [name])
        return mapping

    @staticmethod
    def _validate(mapping, path):
        if not isinstance(mapping, Mapping):
            raise InvalidTreeError(
                "Invalid tree at {node}:\n{mapping}".format(
                    node=f"node {path}" if path else "root node",
                    mapping=pformat(mapping, indent=2),
                )
            )
