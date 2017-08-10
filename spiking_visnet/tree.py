#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tree.py

"""Provide the ``Tree`` class."""

# pylint: disable=attribute-defined-outside-init,no-member
# pylint: disable=too-few-public-methods,too-many-ancestors

from collections import ChainMap, UserDict

import yaml


class Tree(UserDict):
    """A tree of dictionary-like nodes.

    Note that the order of traversals is undefined.

    Keyword Args:
        mapping (Mapping): A dictionary-like object that maps names to
            children, but with a special key-value pair containing the node's
            data. Defaults to an empty dictionary.
        data_key (Hashable) type: The special key of ``mapping`` that maps to
            the node's data. Defaults to ``DEFAULT_DATA_KEY``.

    Attributes:
        c (dict): A dictionary of named children.
        p (type(self)): The parent of this node.
    """

    DEFAULT_DATA_KEY = 'data'

    def __init__(self, mapping=None, data_key=None):
        if mapping is None:
            mapping = dict()
        if data_key is None:
            data_key = self.DEFAULT_DATA_KEY
        # Put data into self.
        super().__init__(mapping.get(data_key, dict()))
        # Default parent is an empty dictionary so dictionary-related errors
        # are raised when traversing upwards.
        self.p = dict()  # pylint: disable=invalid-name
        # Create named child nodes, if any.
        self.c = {  # pylint: disable=invalid-name
            name: type(self)(child, data_key=data_key)
            for name, child in mapping.items()
            if name != data_key
        }
        # Set the parent references on children.
        for child in self.c.values():
            child.p = self

    def __eq__(self, other):
        """Trees are equal when their data and children are equal.

        Note that parents can differ.
        """
        return self.data == other.data and self.c == other.c

    def ancestors(self):
        """Generate the ancestors of this node.

        Includes self as the first element.
        """
        try:
            yield self
            yield from self.p.ancestors()
        except AttributeError:
            return

    def keys(self):
        yield from iter(self)

    def children(self):
        """Generate the child nodes (in undefined order)."""
        yield from self.c.values()

    @property
    def num_children(self):
        """The number of children."""
        return len(self.c)

    def leaves(self):
        """Generate the leaf nodes (in undefined order)."""
        # Base case: leaf
        if not self.c:
            yield self
            return
        # Recursive case: not a leaf
        for child in self.children():
            yield from child.leaves()

    @classmethod
    def merge(cls, *trees):
        """Merge trees into a new tree. Earlier trees take precedence.

        If a node exists at the same location in more than one tree and these
        nodes have duplicate keys, the values from the nodes in the trees that
        appear earlier in the argument list will take precedence over those
        from later trees.
        """
        merged = cls()
        # Merge data.
        merged.data = dict(ChainMap(*trees))
        # Merge children recursively.
        all_names = set.union(*[set(tree.c.keys()) for tree in trees])
        merged.c = {
            name: cls.merge(*[tree.c.get(name, cls()) for tree in trees])
            for name in all_names
        }
        # Update parent references.
        for child in merged.children():
            child.p = merged
        return merged

    @classmethod
    def load(cls, path):
        """Load a YAML representation of a tree."""
        with open(path, 'rt') as tree:
            return cls(yaml.load(tree))


class Params(Tree):
    """A tree of dict-like nodes that inherit and override ancestors' data."""

    DEFAULT_DATA_KEY = 'params'

    def __missing__(self, key):
        """Traverse the tree upwards to find the value."""
        return self.p[key]

    def __contains__(self, key):
        """Return whether key is in self or any ancestor."""
        return super().__contains__(key) or key in self.p

    def __repr__(self):
        return 'Params[{num_children}]({data})'.format(
            num_children=len(self.c), data=self._all_data())

    def _all_data(self):
        # Access underlying data dictionary to avoid infinite recursion
        return dict(ChainMap(*[a.data for a in self.ancestors()]))

    def __iter__(self):
        """Iterate over keys, including inherited ones.

        Deeper values override shallower values.
        """
        yield from self._all_data().keys()


# Complete the Params docstring with all but the first line of the Tree
# docstring
Params.__doc__ = '\n'.join([Params.__doc__] + Tree.__doc__.split('\n')[1:])
