# -*- coding: utf-8 -*-
"""
biosppy.utils
-------------

This module provides several frequently used functions and hacks.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import map, range, zip
import six

# built-in
import collections
import copy
import keyword
import os
import re

# 3rd party
import numpy as np


def normpath(path):
    """Normalize a path.

    Parameters
    ----------
    path : str
        The path to normalize.

    Returns
    -------
    npath : str
        The normalized path.

    """

    if "~" in path:
        out = os.path.abspath(os.path.expanduser(path))
    else:
        out = os.path.abspath(path)

    return out


def fileparts(path):
    """split a file path into its directory, name, and extension.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    dirname : str
        File directory.
    fname : str
        File name.
    ext : str
        File extension.

    Notes
    -----
    * Removes the dot ('.') from the extension.

    """

    dirname, fname = os.path.split(path)
    fname, ext = os.path.splitext(fname)
    ext = ext.replace(".", "")

    return dirname, fname, ext


def fullfile(*args):
    """Join one or more file path components, assuming the last is
    the extension.

    Parameters
    ----------
    ``*args`` : list, optional
        Components to concatenate.

    Returns
    -------
    fpath : str
        The concatenated file path.

    """

    nb = len(args)
    if nb == 0:
        return ""
    elif nb == 1:
        return args[0]
    elif nb == 2:
        return os.path.join(*args)

    fpath = os.path.join(*args[:-1]) + "." + args[-1]

    return fpath


def walktree(top=None, spec=None):
    """Iterator to recursively descend a directory and return all files
    matching the spec.

    Parameters
    ----------
    top : str, optional
        Starting directory; if None, defaults to the current working directoty.
    spec : str, optional
        Regular expression to match the desired files;
        if None, matches all files; typical patterns:
            * `r'\.txt$'` - matches files with '.txt' extension;
            * `r'^File_'` - matches files starting with 'File\_'
            * `r'^File_.+\.txt$'` - matches files starting with 'File\_' and ending with the '.txt' extension.

    Yields
    ------
    fpath : str
        Absolute file path.

    Notes
    -----
    * Partial matches are also selected.

    See Also
    --------
    * https://docs.python.org/3/library/re.html
    * https://regex101.com/

    """

    if top is None:
        top = os.getcwd()

    if spec is None:
        spec = r".*?"

    prog = re.compile(spec)

    for root, _, files in os.walk(top):
        for name in files:
            if prog.search(name):
                fname = os.path.join(root, name)
                yield fname


def remainderAllocator(votes, k, reverse=True, check=False):
    """Allocate k seats proportionally using the Remainder Method.

    Also known as Hare-Niemeyer Method. Uses the Hare quota.

    Parameters
    ----------
    votes : list
        Number of votes for each class/party/cardinal.
    k : int
        Total number o seats to allocate.
    reverse : bool, optional
        If True, allocates remaining seats largest quota first.
    check : bool, optional
        If True, limits the number of seats to the total number of votes.

    Returns
    -------
    seats : list
        Number of seats for each class/party/cardinal.

    """

    # check total number of votes
    tot = np.sum(votes)
    if check and k > tot:
        k = tot

    # frequencies
    length = len(votes)
    freqs = np.array(votes, dtype="float") / tot

    # assign items
    aux = k * freqs
    seats = aux.astype("int")

    # leftovers
    nb = k - seats.sum()
    if nb > 0:
        if reverse:
            ind = np.argsort(aux - seats)[::-1]
        else:
            ind = np.argsort(aux - seats)

        for i in range(nb):
            seats[ind[i % length]] += 1

    return seats.tolist()


def highestAveragesAllocator(votes, k, divisor="dHondt", check=False):
    """Allocate k seats proportionally using the Highest Averages Method.

    Parameters
    ----------
    votes : list
        Number of votes for each class/party/cardinal.
    k : int
        Total number o seats to allocate.
    divisor : str, optional
        Divisor method; one of 'dHondt', 'Huntington-Hill', 'Sainte-Lague',
        'Imperiali', or 'Danish'.
    check : bool, optional
        If True, limits the number of seats to the total number of votes.

    Returns
    -------
    seats : list
        Number of seats for each class/party/cardinal.

    """

    # check total number of cardinals
    tot = np.sum(votes)
    if check and k > tot:
        k = tot

    # select divisor
    if divisor == "dHondt":
        fcn = lambda i: float(i)
    elif divisor == "Huntington-Hill":
        fcn = lambda i: np.sqrt(i * (i + 1.0))
    elif divisor == "Sainte-Lague":
        fcn = lambda i: i - 0.5
    elif divisor == "Imperiali":
        fcn = lambda i: float(i + 1)
    elif divisor == "Danish":
        fcn = lambda i: 3.0 * (i - 1.0) + 1.0
    else:
        raise ValueError("Unknown divisor method.")

    # compute coefficients
    tab = []
    length = len(votes)
    D = [fcn(i) for i in range(1, k + 1)]
    for i in range(length):
        for j in range(k):
            tab.append((i, votes[i] / D[j]))

    # sort
    tab.sort(key=lambda item: item[1], reverse=True)
    tab = tab[:k]
    tab = np.array([item[0] for item in tab], dtype="int")

    seats = np.zeros(length, dtype="int")
    for i in range(length):
        seats[i] = np.sum(tab == i)

    return seats.tolist()


def random_fraction(indx, fraction, sort=True):
    """Select a random fraction of an input list of elements.

    Parameters
    ----------
    indx : list, array
        Elements to partition.
    fraction : int, float
        Fraction to select.
    sort : bool, optional
        If True, output lists will be sorted.

    Returns
    -------
    use : list, array
        Selected elements.
    unuse : list, array
        Remaining elements.

    """

    # number of elements to use
    fraction = float(fraction)
    nb = int(fraction * len(indx))

    # copy because shuffle works in place
    aux = copy.deepcopy(indx)

    # shuffle
    np.random.shuffle(aux)

    # select
    use = aux[:nb]
    unuse = aux[nb:]

    # sort
    if sort:
        use.sort()
        unuse.sort()

    return use, unuse


class ReturnTuple(tuple):
    """A named tuple to use as a hybrid tuple-dict return object.

    Parameters
    ----------
    values : iterable
        Return values.
    names : iterable, optional
        Names for return values.

    Raises
    ------
    ValueError
        If the number of values differs from the number of names.
    ValueError
        If any of the items in names:
        * contain non-alphanumeric characters;
        * are Python keywords;
        * start with a number;
        * are duplicates.

    """

    def __new__(cls, values, names=None):

        return tuple.__new__(cls, tuple(values))

    def __init__(self, values, names=None):

        nargs = len(values)

        if names is None:
            # create names
            names = ["_%d" % i for i in range(nargs)]
        else:
            # check length
            if len(names) != nargs:
                raise ValueError("Number of names and values mismatch.")

            # convert to str
            names = list(map(str, names))

            # check for keywords, alphanumeric, digits, repeats
            seen = set()
            for name in names:
                if not all(c.isalnum() or (c == "_") for c in name):
                    raise ValueError(
                        "Names can only contain alphanumeric \
                                      characters and underscores: %r."
                        % name
                    )

                if keyword.iskeyword(name):
                    raise ValueError("Names cannot be a keyword: %r." % name)

                if name[0].isdigit():
                    raise ValueError("Names cannot start with a number: %r." % name)

                if name in seen:
                    raise ValueError("Encountered duplicate name: %r." % name)

                seen.add(name)

        self._names = names

    def as_dict(self):
        """Convert to an ordered dictionary.

        Returns
        -------
        out : OrderedDict
            An OrderedDict representing the return values.

        """

        return collections.OrderedDict(zip(self._names, self))

    __dict__ = property(as_dict)

    def __getitem__(self, key):
        """Get item as an index or keyword.

        Returns
        -------
        out : object
            The object corresponding to the key, if it exists.

        Raises
        ------
        KeyError
            If the key is a string and it does not exist in the mapping.
        IndexError
            If the key is an int and it is out of range.

        """

        if isinstance(key, six.string_types):
            if key not in self._names:
                raise KeyError("Unknown key: %r." % key)

            key = self._names.index(key)

        return super(ReturnTuple, self).__getitem__(key)

    def __repr__(self):
        """Return representation string."""

        tpl = "%s=%r"

        rp = ", ".join(tpl % item for item in zip(self._names, self))

        return "ReturnTuple(%s)" % rp

    def __getnewargs__(self):
        """Return self as a plain tuple; used for copy and pickle."""

        return tuple(self)

    def keys(self):
        """Return the value names.

        Returns
        -------
        out : list
            The keys in the mapping.

        """

        return list(self._names)
