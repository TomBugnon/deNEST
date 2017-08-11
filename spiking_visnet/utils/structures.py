#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils/structures.py


"""Tree, dictionary and list utility functions."""


import copy as cp
from collections import ChainMap
from collections.abc import Mapping


def chaintree(tree_list, children_key='children'):
    """Recursively combine the trees in <tree_list> using ChainMaps.

    If there is only one tree in tree_list, return it.
    Otherwise, return a tree of which each node with path <path> has:
    - for keys the set of all keys of the node at <path> in the tree_list trees
    - for value under <key> either:
        if <key> is not children_key:
            - the ChainMap of values under <key> at <path> of all the trees if
                all there is more than one value and they are all dictionaries
            - the value under <key> at <path> of the first tree in treelist for
                which this <key> at <path> exists if one of the values is not a
                dictionary, or if there is only one value.
        if <key> is children_key:
            - the recursively chained subtrees.
    """
    chained_tree = {}

    # Remove empty stuff from the list
    tree_list = [tree for tree in tree_list if bool(tree)]
    if len(tree_list) == 1:
        return tree_list[0]
    if len(tree_list) == 0:
        return {}

    # Combine horizontally the values in all trees under each key except
    # <children_key>
    key_value_tups = all_key_values(tree_list, omit_keys=[children_key])
    chained_tree.update({key: combine_values(values)
                         for (key, values) in key_value_tups})

    # Recursively combine children if there are any.
    # <children_tups> is a list of (<child_key>, <list_of_child_subtrees>
    children_tups = all_key_values([tree[children_key]
                                    for tree in tree_list
                                    if (children_key in tree
                                        and tree[children_key])])
    if len(children_tups) > 0:
        combined_subtrees = {child_key: chaintree(child_subtrees_list,
                                                  children_key='children')
                             for (child_key, child_subtrees_list)
                             in children_tups}
        chained_tree[children_key] = combined_subtrees

    return chained_tree


def combine_values(values):
    """Combine a list of elements.

    - First remove empty stuff from the list.
    - Return an empty dict is the list is empty.
    - Return either the ChainMap of the list <values> if all its elements are
    mappings, or the first element of the list otherwise.

    """
    values = [v for v in values if v]
    if len(values) == 0:
        return {}
    elif len(values) > 1 and all([isinstance(x, Mapping) for x in values]):
        return ChainMap(*values)
    return(values[0])


def all_key_values(dict_list, omit_keys=[]):
    """Combine entries across a list of dictionaries.

    Returns:
        list: list of tuples each of the form:
                (<key>, <value_list>)
            for each unique key in the dictionaries of dict_list, omitting the
            keys in <omit_keys>.
            <value_list> is the list of values under a given key in each of the
            dictionaries, scanned from left to right.

    """
    all_keys = (set(flatten([list(d.keys())
                             for d in dict_list]))
                - set(omit_keys))
    return [(key, [d[key] for d in dict_list if key in d])
            for key in all_keys]


def invert_dict(d, inversion_key):
    """Invert a dictionary.

    Inverts a dictionary of the form {<name>:{<inversion_key>:<value>, ...}}
    into a dictionary of the form: {<value>:<name_list>} where <name_list> is
    a list of the keys <name> of which <value> is the inversion_key value.

    Examples:
        >>> a = {'key1':{'inversion_key': value1,
                         'other_key1': x1},
                 'key2':{'inversion_key': value1,
                         'other_key1': x2}
                 'key3':{'inversion_key': value2,
                          'other_key2': x1}
        >>> invert_dict(a)
        {'value1':['key1', 'key2'],
         'value2':[key3]}

    """
    return {value: [key for (key, entry) in d.items()
                    if entry[inversion_key] == value]
            for value in set([entry[inversion_key]
                              for _, entry in d.items()])}


def deepcopy_dict(source_dict, diffs={}):
    """Return a deep copy of <source_dict> updated with <diffs>."""
    result = cp.deepcopy(source_dict)
    result.update(diffs)
    return result


def distribute_to_tuple(tuple_list, value, pos=0):
    """Insert <value> at the position <pos> of each tuple in <tuple_list>."""
    return [tuple(insert_in_list(list(tup), value, pos))
            for tup in tuple_list]


def insert_in_list(l, value, position=0):
    """Insert value at specific position of list and returns updated list."""
    l.insert(position, value)
    return l


def flatten(l):
    """Flatten a list of lists.

    If <l> is a list of lists, return the list of items in the sublists.
    If some elements of <l> are not lists, don't iterate on them. Therefore
    flatten(l) == l if l is eg a list of tuples.

    Examples:
        >>> flatten([[1,2],[3,4]])
        [1,2,3,4]
        >>> flatten(['notlist', ['li', 'st'])
        ['notlist', 'li', 'st']

    """
    gen = (x if isinstance(x, list) else [x] for x in l)
    return [item for sublist in gen for item in sublist]


def traverse(tree, data_key='params', children_key='children',
             name_key='name', accumulator=None):
    """Return the leaf nodes of a tree, accumulating data from ancestors.

    For each leaf, return the value of its ``name_key`` key and a ChainMap
    containing the ordered contents of the ``data_key`` field (if it exists) in
    each of the parent nodes.

    Args:
        data_key (str): Look up data with this key.
        accumulator (ChainMap): Append the newly accumulated parameters to
            this ChainMap. Used for recursion.
        children_key (str): The nodes children are a list of trees under this
            key.
        name_key (str): When reaching a leaf, its name is under the key
            name_key.

    Returns:
        list: list of tuples of the form ``(<leaf_name>, <data_chainmap>)``
        where ``leaf_name`` is the value of the ``name`` key of each leaf and
        ``data_chainmap`` is a chainmap of the accumulated data from leaf to
        root.
    """
    if accumulator is None:
        accumulator = list()
    # Get our own accumulator (we need a separate object for each leaf)
    accumulator = list(accumulator)
    # Get the current data if there is any
    if tree.get(data_key, False):
        accumulator.append(tree[data_key])
    # Base case: leaf
    if not tree.get(children_key, False):
        return (tree[name_key], ChainMap(*reversed(accumulator)))
    # Recursive case: not a leaf
    return flatten([traverse(child,
                             data_key=data_key,
                             children_key=children_key,
                             name_key=name_key,
                             accumulator=accumulator)
                    for child in tree[children_key].values()])


def dictify(obj):
    """Recursively convert generic mappings to dictionaries."""
    if isinstance(obj, Mapping):
        return {key: dictify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dictify(elt) for elt in obj]
    return obj
