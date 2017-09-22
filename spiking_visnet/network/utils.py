#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network/utils.py

import functools
import logging
import logging.config

log = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
        }
    },
    'handlers': {
        'stdout': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        }
    },
    'loggers': {
        'spiking_visnet': {
            'level': 'INFO',
            'handlers': ['stdout'],
        }
    }
})

def flatten(seq):
    """Flatten an iterable of iterables into a tuple."""
    return tuple(item for subseq in seq for item in subseq)


def indent(string, amount=2):
    """Indent a string by an amount."""
    return '\n'.join((' ' * amount) + line for line in string.split('\n'))


class NotCreatedError(AttributeError):
    """Raised when a ``NestObject`` needs to have been created, but wasn't."""
    pass


# pylint: disable=protected-access

def if_not_created(method):
    """Only call a method if the ``_created`` attribute isn't set.

    After calling, sets ``_created = True``.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):  # pylint: disable=missing-docstring
        if self._created:
            log.warning('Attempted to create object more than once:\n%s',
                        indent(str(self)))
            return
        try:
            self._created = True
            value = method(self, *args, **kwargs)
        except Exception as error:
            self._created = False
            raise error
        return value
    return wrapper


def if_created(method):
    """Raise an error if the `_created` attribute is not set."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):  # pylint: disable=missing-docstring
        if not self._created:
            raise NotCreatedError('Must call `create()` first:\n' +
                                  indent(str(self)))
        return method(self, *args, **kwargs)
    return wrapper


@functools.total_ordering
class NestObject:
    """Base class for a named NEST object.

    Args:
        name (str): The name of the object.
        params (Params): The object parameters.

    Objects are ordered and hashed by name.
    """

    def __init__(self, name, params):
        self.name = name
        # Flatten the parameters to a dictionary (and make a copy)
        self.params = dict(params)
        # Whether the object has been created in NEST
        self._created = False

    # pylint: disable=unused-argument,invalid-name
    def _repr_pretty_(self, p, cycle):
        opener = '{classname}({name}, '.format(
            classname=type(self).__name__, name=self.name)
        closer = ')'
        with p.group(p.indentation, opener, closer):
            p.breakable()
            p.pretty(self.params)
    # pylint: enable=unused-argument,invalid-name

    def __repr__(self):
        return '{classname}({name}, {params})'.format(
            classname=type(self).__name__,
            name=self.name,
            params=pformat(self.params))

    def __lt__(self, other):
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)

    def __getattr__(self, name):
        try:
            return self.params[name]
        except KeyError:
            return self.__getattribute__(name)
