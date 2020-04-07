#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# parameters.py

"""Provide the ``Tree`` class."""

from collections import ChainMap, UserDict
from collections.abc import Mapping
from pprint import pformat

import yaml


class InvalidTreeError(ValueError):
    """Raised when a mapping is not a valid ``Tree``."""
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


class Tree(UserDict):

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
        # Data internal to this node. Keys are keys in DATA_KEYS. data keys
        # contain empty dictionaries by default.
        self._data = {
            key: mapping.get(key, {})
            for key in self.DATA_KEYS
        }
        # Accessible data (inherits from parents)
        super().__init__(self._data)
        # if name == 'warmup':
        #     import ipdb; ipdb.set_trace()
        self.data = {
            key: DeepChainMap(
                self.data[key],
                *(ancestor._data[key] for ancestor in self.ancestors()),
            )
            for key in self.DATA_KEYS
        }
        # Children
        self._children = {
            key: Tree(value, parent=self, name=key)
            for key, value in mapping.items()
            if key not in self.DATA_KEYS
        }
        # Syntactic sugar to allow data keys to be accessed as attributes
        for data_key, value in self.data.items():
            setattr(self, data_key, value)

    def __repr__(self):
        return (
            f'{type(self).__name__}[{len(self.children)}]'
            f'(`{self._name}`, {dict(self._data)})'
        )

    def __str__(self):
        return repr(self)

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

        More recent ancestors appear earlier in the list.
        """
        ancestors = []
        node = self
        while True:
            if node.parent is None:
                break
            ancestors.append(node.parent)
            node = node.parent
        return ancestors

    def leaves(self):
        """Return a list of leaf nodes of the tree.

        Traversal order is not defined.
        """
        leaves = []
        # Recursive case: not a leaf
        for child in self.children.values():
            leaves.extend(child.leaves())
        # Base case: leaf
        if not leaves:
            leaves.append(self)
        return leaves

    def named_leaves(self):
        """Return list of `(<name>, <node>)` tuples for all the leaves.

        Traversal order is not defined."""
        return [(leave.name, leave) for leave in self.leaves()]

    @classmethod
    def merge(cls, *trees, parent=None, name=None):
        """Merge trees into a new tree. Earlier trees take precedence.

        If a node exists at the same location in more than one tree and these
        nodes have duplicate keys, the values from the nodes in the trees that
        appear earlier in the argument list will take precedence over those
        from later trees.
        """
        # Merge node's own data
        node_data = {
            key: ChainMap(*(tree._data[key] for tree in trees))
            for key in cls.DATA_KEYS
        }
        # Initialize tree with node data and no children.
        merged = cls(mapping=node_data, parent=parent, name=name)
        # Merge children recursively. Pass parent so that children inherit
        # merged node's data
        all_names = set.union(*[set(tree.children.keys()) for tree in trees])
        merged._children = {
            name: cls.merge(
                *[
                    tree.children[name] for tree in trees
                    if name in tree.children
                ],
                parent=merged,
                name=name
            )
            for name in all_names
        }
        return merged

    @classmethod
    def read(cls, path):
        with open(path, 'rt') as f:
            return cls(yaml.load(f, Loader=yaml.SafeLoader))

    def write(self, path):
        # TODO save (in yaml)
        # with open(path, 'wt') as f:
        pass

    def print(self):
        # TODO print tree
        pass

    def validate(self, mapping, path=None):
        """Check that a mapping is a valid ``Tree``."""
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
            raise InvalidTreeError('Invalid tree at {node}:\n{mapping}'.format(
                node=f'node {path}' if path else 'root node',
                mapping=pformat(mapping, indent=2)
            ))
