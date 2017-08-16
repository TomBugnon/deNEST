#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# parameters.py

"""Provide the ``Tree``, ``Scope``, and ``Params`` class."""

# pylint: disable=attribute-defined-outside-init,no-member
# pylint: disable=too-few-public-methods,too-many-ancestors

from collections import ChainMap, UserDict
from collections.abc import Mapping
import os
from pprint import pformat

import yaml


class InvalidTreeError(ValueError):
    """Raised when a mapping is not a valid ``Tree``."""

    pass


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

    def __init__(self, mapping=None, data_key=None, validate=True):
        self.data_key = data_key or self.DEFAULT_DATA_KEY
        # Validate mapping.
        if mapping is None:
            mapping = dict()
        if validate:
            mapping = self.validate(mapping)
        # Put data into self.
        super().__init__(mapping.get(self.data_key, dict()))
        # Default parent is an empty dictionary so dictionary-related errors
        # are raised when traversing upwards.
        self.p = dict()  # pylint: disable=invalid-name
        # Create named child nodes, if any.
        self.c = {  # pylint: disable=invalid-name
            name: type(self)(child, data_key=self.data_key, validate=False)
            for name, child in mapping.items()
            if name != self.data_key
        }
        # Set the parent references on children.
        for child in self.c.values():
            child.p = self

    def __repr__(self, data=None):
        if data is None:
            data = self.data
        return '{cls}[{num_children}]({data})'.format(
            cls=self.__class__.__name__, num_children=len(self.c), data=data)

    def __eq__(self, other):
        """Trees are equal when their data and children are equal.

        Note that parents can differ.
        """
        return self.data == other.data and self.c == other.c

    def get_node(self, name):
        """Traverse the tree downward to get a node.

        If ``name`` is not a tuple, returns just the child node of that name.
        """
        if isinstance(name, tuple) and name:
            name, descendants = name[0], name[1:]
            if descendants:
                return self.get_node(name).get_node(descendants)
        try:
            return self.c[name]
        except KeyError:
            raise KeyError(f'no child named `{name}`')

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

    def named_children(self):
        """Generate the (name, node) pairs of children (in undefined order)."""
        yield from self.c.items()

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

    def named_leaves(self, _name=None):
        """Generate the named and nodes of the leaves (in undefined order)."""
        # Base case: leaf
        if not self.c:
            yield (_name, self)
            return
        # Recursive case: not a leaf
        for name, child in self.named_children():
            yield from child.named_leaves(_name=name)

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
    def load(cls, *path):
        """Load a YAML representation of a tree."""
        with open(os.path.join(*path), 'rt') as tree:
            return cls(yaml.load(tree))

    def validate(self, mapping, path=None):
        """Check that a mapping is a valid ``Tree``."""
        if path is None:
            path = list()
        if mapping:
            self._validate(mapping, path)
            for name, child in mapping.items():
                # Validate data
                if name == self.data_key:
                    self._validate(child, path)
                # Validate children
                else:
                    self.validate(child, path + [name])
        return mapping

    @staticmethod
    def _validate(mapping, path):
        if not isinstance(mapping, Mapping):
            raise InvalidTreeError('invalid tree at {node}:\n{mapping}'.format(
                node=f'node {path}' if path else 'root node',
                mapping=pformat(mapping, indent=2)))


class Scope(Tree):
    """A tree of dict-like nodes that inherit and override ancestors' data."""
    # Append Tree docstring
    __doc__ += '\n' + '\n'.join(Tree.__doc__.split('\n')[1:])

    def _all_data(self):
        # Access underlying data dictionary to avoid infinite recursion
        return dict(ChainMap(*[a.data for a in self.ancestors()]))

    def __repr__(self):  # pylint: disable=signature-differs
        return super().__repr__(data=self._all_data())

    def __missing__(self, key):
        """Traverse the tree upwards to find the value."""
        return self.p[key]

    def __contains__(self, key):
        """Return whether key is in self or any ancestor."""
        return super().__contains__(key) or key in self.p

    def __iter__(self):
        """Iterate over keys, including inherited ones.

        Deeper values override shallower values.
        """
        yield from self._all_data().keys()


class Params(Scope):
    """A tree of parameters.

    Supports traversal by access with tuples.
    """
    # Insert Tree docstring
    __doc__ += '\n'.join(Tree.__doc__.split('\n')[1:])
    # Append examples
    __doc__ += """
    Examples:
        Accessing with a tuple traverses the tree:

        >>> parameters = Params(
        ...     {'c1': {'c2': {'params': {'last_key': 'value'}}}}
        ... )
        >>> parameters[('c1', 'c2', 'last_key')]
        'param_value'

        Values can be set similarly:

        >>> parameters = Params(
        ...     {'c1': {'c2': {'params': {'last_key': 'value'}}}}
        ... )
        >>> parameters[('c1', 'c2', 'last_key')] = 'new_value'
        >>> parameters[('c1', 'c2', 'last_key')]
        'new_value'
    """

    DEFAULT_DATA_KEY = 'params'

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.get_node(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self[key[:-1]][key[-1]] = value
        else:
            super().__setitem__(key, value)
