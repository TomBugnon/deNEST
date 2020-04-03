#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# parameters.py

"""Provide the ``Params`` class."""

from collections import ChainMap, Mapping, UserDict


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


class RecursiveChainMap(ChainMap):
    """Variant of ChainMap that also chains nested mappings together."""

    def __getitem__(self, key):
        values = []
        for mapping in self.maps:
            try:
                values.append(mapping[key])
            except KeyError:
                pass
        if not values:
            return self.__missing__(key)
        is_mapping = [isinstance(value, Mapping) for value in values]
        if any(is_mapping):
            # Validate values: if one value is a mapping, all must be
            if not all(is_mapping):
                raise KeyError(
                    f"inconsistent structure for key '{key}': either all"
                    " or none of the values must be mappings"
                )
            # All are mappings
            return RecursiveChainMap(*values)
        # Return first value (earlier maps have precedence)
        return values[0]

    def __repr__(self):
        return str(dict(self))


class RecursiveDeepChainMap(RecursiveChainMap, DeepChainMap):
    pass


class Tree(UserDict):

    DATA_KEYS = ["params", "nest_params"]

    def __init__(self, mapping, parent=None, name=None):
        # Parent
        self._parent = parent
        # Name
        self._name = name
        # Data internal to this node
        self._data = {key: value for key, value in mapping.items() if key in self.DATA_KEYS}
        # Accessible data (inherits from parents)
        super().__init__(self._data)
        self.data = RecursiveDeepChainMap(
            self.data,
            *(ancestor._data for ancestor in self.ancestors()),
        )
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
        return f'{type(self).__name__}[{len(self.children)}]({dict(self.data)})'

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

    @classmethod
    def read(cls, path):
        with open(path, 'rt') as f:
            return cls(yaml.load(f, Loader=yaml.SafeLoader))

    def write(self, path):
        with open(path, 'wt') as f:
            # TODO save (in yaml)
            pass

    def print(self):
        pass
        # TODO print tree