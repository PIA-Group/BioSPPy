# -*- coding: utf-8 -*-
"""
biosppy.storage
---------------

This module provides several data storage methods.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range
import six

# built-in
import datetime
import json
import os
import zipfile

# 3rd party
import h5py
import numpy as np
import shortuuid
import joblib

# local
from . import utils


def serialize(data, path, compress=3):
    """Serialize data and save to a file using sklearn's joblib.

    Parameters
    ----------
    data : object
        Object to serialize.
    path : str
        Destination path.
    compress : int, optional
        Compression level; from 0 to 9 (highest compression).

    """

    # normalize path
    utils.normpath(path)

    joblib.dump(data, path, compress=compress)


def deserialize(path):
    """Deserialize data from a file using sklearn's joblib.

    Parameters
    ----------
    path : str
        Source path.

    Returns
    -------
    data : object
        Deserialized object.

    """

    # normalize path
    path = utils.normpath(path)

    return joblib.load(path)


def dumpJSON(data, path):
    """Save JSON data to a file.

    Parameters
    ----------
    data : dict
        The JSON data to dump.
    path : str
        Destination path.

    """

    # normalize path
    path = utils.normpath(path)

    with open(path, 'w') as fid:
        json.dump(data, fid)


def loadJSON(path):
    """Load JSON data from a file.

    Parameters
    ----------
    path : str
        Source path.

    Returns
    -------
    data : dict
        The loaded JSON data.

    """

    # normalize path
    path = utils.normpath(path)

    with open(path, 'r') as fid:
        return json.load(fid)


def zip_write(fid, files, recursive=True, root=None):
    """Write files to zip archive.

    Parameters
    ----------
    fid : file-like object
        The zip file to write into.
    files : iterable
        List of files or directories to pack.
    recursive : bool, optional
        If True, sub-directories and sub-folders are also written to the
        archive.
    root : str, optional
        Relative folder path.

    Notes
    -----
    * Ignores non-existent files and directories.

    """

    if root is None:
        root = ''

    for item in files:
        fpath = utils.normpath(item)

        if not os.path.exists(fpath):
            continue

        # relative archive name
        arcname = os.path.join(root, os.path.split(fpath)[1])

        # write
        fid.write(fpath, arcname)

        # recur
        if recursive and os.path.isdir(fpath):
            rfiles = [os.path.join(fpath, subitem)
                      for subitem in os.listdir(fpath)]
            zip_write(fid, rfiles, recursive=recursive, root=arcname)


def pack_zip(files, path, recursive=True, forceExt=True):
    """Pack files into a zip archive.

    Parameters
    ----------
    files : iterable
        List of files or directories to pack.
    path : str
        Destination path.
    recursive : bool, optional
        If True, sub-directories and sub-folders are also written to the
        archive.
    forceExt : bool, optional
        Append default extension.

    Returns
    -------
    zip_path : str
        Full path to created zip archive.

    """

    # normalize destination path
    zip_path = utils.normpath(path)

    if forceExt:
        zip_path += '.zip'

    # allowZip64 is True to allow files > 2 GB
    with zipfile.ZipFile(zip_path, 'w', allowZip64=True) as fid:
        zip_write(fid, files, recursive=recursive)

    return zip_path


def unpack_zip(zip_path, path):
    """Unpack a zip archive.

    Parameters
    ----------
    zip_path : str
        Path to zip archive.
    path : str
        Destination path (directory).

    """

    # allowZip64 is True to allow files > 2 GB
    with zipfile.ZipFile(zip_path, 'r', allowZip64=True) as fid:
        fid.extractall(path)


def alloc_h5(path):
    """Prepare an HDF5 file.

    Parameters
    ----------
    path : str
        Path to file.

    """

    # normalize path
    path = utils.normpath(path)

    with h5py.File(path):
        pass


def store_h5(path, label, data):
    """Store data to HDF5 file.

    Parameters
    ----------
    path : str
        Path to file.
    label : hashable
        Data label.
    data : array
        Data to store.

    """

    # normalize path
    path = utils.normpath(path)

    with h5py.File(path) as fid:
        label = str(label)

        try:
            fid.create_dataset(label, data=data)
        except (RuntimeError, ValueError):
            # existing label, replace
            del fid[label]
            fid.create_dataset(label, data=data)


def load_h5(path, label):
    """Load data from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to file.
    label : hashable
        Data label.

    Returns
    -------
    data : array
        Loaded data.

    """

    # normalize path
    path = utils.normpath(path)

    with h5py.File(path) as fid:
        label = str(label)

        try:
            return fid[label][...]
        except KeyError:
            return None


def store_txt(path, data, sampling_rate=1000., resolution=None, date=None,
              labels=None, precision=6):
    """Store data to a simple text file.

    Parameters
    ----------
    path : str
        Path to file.
    data : array
        Data to store (up to 2 dimensions).
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    resolution : int, optional
        Sampling resolution.
    date : datetime, str, optional
        Datetime object, or an ISO 8601 formatted date-time string.
    labels : list, optional
        Labels for each column of `data`.
    precision : int, optional
        Precision for string conversion.

    Raises
    ------
    ValueError
        If the number of data dimensions is greater than 2.
    ValueError
        If the number of labels is inconsistent with the data.

    """

    # ensure numpy
    data = np.array(data)

    # check dimension
    if data.ndim > 2:
        raise ValueError("Number of data dimensions cannot be greater than 2.")

    # build header
    header = "Simple Text Format\n"
    header += "Sampling Rate (Hz):= %0.2f\n" % sampling_rate
    if resolution is not None:
        header += "Resolution:= %d\n" % resolution
    if date is not None:
        if isinstance(date, six.string_types):
            header += "Date:= %s\n" % date
        elif isinstance(date, datetime.datetime):
            header += "Date:= %s\n" % date.isoformat()
    else:
        ct = datetime.datetime.utcnow().isoformat()
        header += "Date:= %s\n" % ct

    # data type
    header += "Data Type:= %s\n" % data.dtype

    # labels
    if data.ndim == 1:
        ncols = 1
    elif data.ndim == 2:
        ncols = data.shape[1]

    if labels is None:
        labels = ['%d' % i for i in range(ncols)]
    elif len(labels) != ncols:
        raise ValueError("Inconsistent number of labels.")

    header += "Labels:= %s" % '\t'.join(labels)

    # normalize path
    path = utils.normpath(path)

    # data format
    p = '%d' % precision
    if np.issubdtype(data.dtype, np.integer):
        fmt = '%d'
    elif np.issubdtype(data.dtype, np.float):
        fmt = '%%.%sf' % p
    elif np.issubdtype(data.dtype, np.bool_):
        fmt = '%d'
    else:
        fmt = '%%.%se' % p

    # store
    np.savetxt(path, data, header=header, fmt=fmt, delimiter='\t')


def load_txt(path):
    """Load data from a text file.

    Parameters
    ----------
    path : str
        Path to file.

    Returns
    -------
    data : array
        Loaded data.
    mdata : dict
        Metadata.

    """

    # normalize path
    path = utils.normpath(path)

    with open(path, 'rb') as fid:
        lines = fid.readlines()

    # extract header
    mdata_tmp = {}
    fields = ['Sampling Rate', 'Resolution', 'Date', 'Data Type', 'Labels']
    values = []
    for item in lines:
        if b'#' in item:
            item = item.decode('utf-8')
            # parse comment
            for f in fields:
                if f in item:
                    mdata_tmp[f] = item.split(':= ')[1].strip()
                    fields.remove(f)
                    break
        else:
            values.append(item)

    # convert mdata
    mdata = {}
    df = '%Y-%m-%dT%H:%M:%S.%f'
    try:
        mdata['sampling_rate'] = float(mdata_tmp['Sampling Rate'])
    except KeyError:
        pass
    try:
        mdata['resolution'] = int(mdata_tmp['Resolution'])
    except KeyError:
        pass
    try:
        dtype = mdata_tmp['Data Type']
    except KeyError:
        dtype = None
    try:
        d = datetime.datetime.strptime(mdata_tmp['Date'], df)
        mdata['date'] = d
    except (KeyError, ValueError):
        pass
    try:
        labels = mdata_tmp['Labels'].split('\t')
        mdata['labels'] = labels
    except KeyError:
        pass

    # load array
    data = np.genfromtxt(values, dtype=dtype, delimiter=b'\t')

    return data, mdata


class HDF(object):
    """Wrapper class to operate on BioSPPy HDF5 files.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    mode : str, optional
        File mode; one of:

        * 'a': read/write, creates file if it does not exist;
        * 'r+': read/write, file must exist;
        * 'r': read only, file must exist;
        * 'w': create file, truncate if it already exists;
        * 'w-': create file, fails if it already esists.

    """

    def __init__(self, path=None, mode='a'):
        # normalize path
        path = utils.normpath(path)

        # open file
        self._file = h5py.File(path, mode)

        # check BioSPPy structures
        try:
            self._signals = self._file['signals']
        except KeyError:
            if mode == 'r':
                raise IOError(
                    "Unable to create 'signals' group with current file mode.")
            self._signals = self._file.create_group('signals')

        try:
            self._events = self._file['events']
        except KeyError:
            if mode == 'r':
                raise IOError(
                    "Unable to create 'events' group with current file mode.")
            self._events = self._file.create_group('events')

    def __enter__(self):
        """Method for with statement."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Method for with statement."""

        self.close()

    def _join_group(self, *args):
        """Join group elements.

        Parameters
        ----------
        ``*args`` : list
            Group elements to join.

        Returns
        -------
        weg : str
            Joined group path.

        """

        bits = []
        for item in args:
            bits.extend(item.split('/'))

        # filter out blanks, slashes, white space
        out = []
        for item in bits:
            item = item.strip()
            if item == '':
                continue
            elif item == '/':
                continue
            out.append(item)

        weg = '/' + '/'.join(out)

        return weg

    def add_header(self, header=None):
        """Add header metadata.

        Parameters
        ----------
        header : dict
            Header metadata.

        """

        # check inputs
        if header is None:
            raise TypeError("Please specify the header information.")

        self._file.attrs['json'] = json.dumps(header)

    def get_header(self):
        """Retrieve header metadata.

        Returns
        -------
        header : dict
            Header metadata.

        """

        try:
            header = json.loads(self._file.attrs['json'])
        except KeyError:
            header = {}

        return utils.ReturnTuple((header,), ('header',))

    def add_signal(self,
                   signal=None,
                   mdata=None,
                   group='',
                   name=None,
                   compress=False):
        """Add a signal to the file.

        Parameters
        ----------
        signal : array
            Signal to add.
        mdata : dict, optional
            Signal metadata.
        group : str, optional
            Destination signal group.
        name : str, optional
            Name of the dataset to create.
        compress : bool, optional
            If True, the signal will be compressed with gzip.

        Returns
        -------
        group : str
            Destination group.
        name : str
            Name of the created signal dataset.

        """

        # check inputs
        if signal is None:
            raise TypeError("Please specify an input signal.")

        if mdata is None:
            mdata = {}

        if name is None:
            name = shortuuid.uuid()

        # navigate to group
        weg = self._join_group(self._signals.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            # create group
            node = self._file.create_group(weg)

        # create dataset
        if compress:
            dset = node.create_dataset(name, data=signal, compression='gzip')
        else:
            dset = node.create_dataset(name, data=signal)

        # add metadata
        dset.attrs['json'] = json.dumps(mdata)

        # output
        grp = weg.replace('/signals', '')

        return utils.ReturnTuple((grp, name), ('group', 'name'))

    def _get_signal(self, group='', name=None):
        """Retrieve a signal dataset from the file.

        Parameters
        ----------
        group : str, optional
            Signal group.
        name : str
            Name of the signal dataset.

        Returns
        -------
        dataset : h5py.Dataset
            HDF5 dataset.

        """

        # check inputs
        if name is None:
            raise TypeError(
                "Please specify the name of the signal to retrieve.")

        # navigate to group
        weg = self._join_group(self._signals.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent signal group.")

        # get data
        try:
            dset = node[name]
        except KeyError:
            raise KeyError("Inexistent signal dataset.")

        return dset

    def get_signal(self, group='', name=None):
        """Retrieve a signal from the file.

        Parameters
        ----------
        group : str, optional
            Signal group.
        name : str
            Name of the signal dataset.

        Returns
        -------
        signal : array
            Retrieved signal.
        mdata : dict
            Signal metadata.

        Notes
        -----
        * Loads the entire signal data into memory.

        """

        dset = self._get_signal(group=group, name=name)
        signal = dset[...]

        try:
            mdata = json.loads(dset.attrs['json'])
        except KeyError:
            mdata = {}

        return utils.ReturnTuple((signal, mdata), ('signal', 'mdata'))

    def del_signal(self, group='', name=None):
        """Delete a signal from the file.

        Parameters
        ----------
        group : str, optional
            Signal group.
        name : str
            Name of the dataset.

        """

        dset = self._get_signal(group=group, name=name)

        try:
            del self._file[dset.name]
        except IOError:
            raise IOError("Unable to delete object with current file mode.")

    def del_signal_group(self, group=''):
        """Delete all signals in a file group.

        Parameters
        ----------
        group : str, optional
            Signal group.

        """

        # navigate to group
        weg = self._join_group(self._signals.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent signal group.")

        if node.name == '/signals':
            # delete all elements
            for _, item in six.iteritems(node):
                try:
                    del self._file[item.name]
                except IOError:
                    raise IOError(
                        "Unable to delete object with current file mode.")
        else:
            # delete single node
            try:
                del self._file[node.name]
            except IOError:
                raise IOError(
                    "Unable to delete object with current file mode.")

    def list_signals(self, group='', recursive=False):
        """List signals in the file.

        Parameters
        ----------
        group : str, optional
            Signal group.
        recursive : bool, optional
            If True, also lists signals in sub-groups.

        Returns
        -------
        signals : list
            List of (group, name) tuples of the found signals.

        """

        # navigate to group
        weg = self._join_group(self._signals.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent signal group.")

        out = []
        for name, item in six.iteritems(node):
            if isinstance(item, h5py.Dataset):
                out.append((group, name))
            elif recursive and isinstance(item, h5py.Group):
                aux = self._join_group(group, name)
                out.extend(self.list_signals(group=aux,
                                             recursive=True)['signals'])

        return utils.ReturnTuple((out,), ('signals',))

    def add_event(self,
                  ts=None,
                  values=None,
                  mdata=None,
                  group='',
                  name=None,
                  compress=False):
        """Add an event to the file.

        Parameters
        ----------
        ts : array
            Array of time stamps.
        values : array, optional
            Array with data for each time stamp.
        mdata : dict, optional
            Event metadata.
        group : str, optional
            Destination event group.
        name : str, optional
            Name of the dataset to create.
        compress : bool, optional
            If True, the data will be compressed with gzip.

        Returns
        -------
        group : str
            Destination group.
        name : str
            Name of the created event dataset.

        """

        # check inputs
        if ts is None:
            raise TypeError("Please specify an input array of time stamps.")

        if values is None:
            values = []

        if mdata is None:
            mdata = {}

        if name is None:
            name = shortuuid.uuid()

        # navigate to group
        weg = self._join_group(self._events.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            # create group
            node = self._file.create_group(weg)

        # create new event group
        evt_node = node.create_group(name)

        # create datasets
        if compress:
            _ = evt_node.create_dataset('ts', data=ts, compression='gzip')
            _ = evt_node.create_dataset('values',
                                        data=values,
                                        compression='gzip')
        else:
            _ = evt_node.create_dataset('ts', data=ts)
            _ = evt_node.create_dataset('values', data=values)

        # add metadata
        evt_node.attrs['json'] = json.dumps(mdata)

        # output
        grp = weg.replace('/events', '')

        return utils.ReturnTuple((grp, name), ('group', 'name'))

    def _get_event(self, group='', name=None):
        """Retrieve event datasets from the file.

        Parameters
        ----------
        group : str, optional
            Event group.
        name : str
            Name of the event dataset.

        Returns
        -------
        event : h5py.Group
            HDF5 event group.
        ts : h5py.Dataset
            HDF5 time stamps dataset.
        values : h5py.Dataset
            HDF5 values dataset.

        """

        # check inputs
        if name is None:
            raise TypeError(
                "Please specify the name of the signal to retrieve.")

        # navigate to group
        weg = self._join_group(self._events.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent event group.")

        # event group
        try:
            evt_group = node[name]
        except KeyError:
            raise KeyError("Inexistent event dataset.")

        # get data
        try:
            ts = evt_group['ts']
        except KeyError:
            raise KeyError("Could not find expected time stamps structure.")

        try:
            values = evt_group['values']
        except KeyError:
            raise KeyError("Could not find expected values structure.")

        return evt_group, ts, values

    def get_event(self, group='', name=None):
        """Retrieve an event from the file.

        Parameters
        ----------
        group : str, optional
            Event group.
        name : str
            Name of the event dataset.

        Returns
        -------
        ts : array
            Array of time stamps.
        values : array
            Array with data for each time stamp.
        mdata : dict
            Event metadata.

        Notes
        -----
        Loads the entire event data into memory.

        """

        evt_group, dset_ts, dset_values = self._get_event(group=group,
                                                          name=name)
        ts = dset_ts[...]
        values = dset_values[...]

        try:
            mdata = json.loads(evt_group.attrs['json'])
        except KeyError:
            mdata = {}

        return utils.ReturnTuple((ts, values, mdata),
                                 ('ts', 'values', 'mdata'))

    def del_event(self, group='', name=None):
        """Delete an event from the file.

        Parameters
        ----------
        group : str, optional
            Event group.
        name : str
            Name of the event dataset.

        """

        evt_group, _, _ = self._get_event(group=group, name=name)

        try:
            del self._file[evt_group.name]
        except IOError:
            raise IOError("Unable to delete object with current file mode.")

    def del_event_group(self, group=''):
        """Delete all events in a file group.

        Parameters
        ----------
        group  str, optional
            Event group.

        """

        # navigate to group
        weg = self._join_group(self._events.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent event group.")

        if node.name == '/events':
            # delete all elements
            for _, item in six.iteritems(node):
                try:
                    del self._file[item.name]
                except IOError:
                    raise IOError(
                        "Unable to delete object with current file mode.")
        else:
            # delete single node
            try:
                del self._file[node.name]
            except IOError:
                raise IOError(
                    "Unable to delete object with current file mode.")

    def list_events(self, group='', recursive=False):
        """List events in the file.

        Parameters
        ----------
        group : str, optional
            Event group.
        recursive : bool, optional
            If True, also lists events in sub-groups.

        Returns
        -------
        events : list
            List of (group, name) tuples of the found events.

        """

        # navigate to group
        weg = self._join_group(self._events.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent event group.")

        out = []
        for name, item in six.iteritems(node):
            if isinstance(item, h5py.Group):
                try:
                    _ = item.attrs['json']
                except KeyError:
                    # normal group
                    if recursive:
                        aux = self._join_group(group, name)
                        out.extend(self.list_events(group=aux,
                                                    recursive=True)['events'])
                else:
                    # event group
                    out.append((group, name))

        return utils.ReturnTuple((out,), ('events',))

    def close(self):
        """Close file descriptor."""

        # flush buffers
        self._file.flush()

        # close
        self._file.close()
