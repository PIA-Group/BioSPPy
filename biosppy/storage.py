# -*- coding: utf-8 -*-
"""
    biosppy.storage
    ---------------
    
    This module provides several data storage methods.
    
    :copyright: (c) 2015 by Instituto de Telecomunicacoes
    :license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# built-in
import json
import os

# 3rd party
import h5py
import shortuuid
import zipfile
from sklearn.externals import joblib

# local
from . import utils

# Globals


def serialize(data, path, compress=3):
    """Serialize data and save to a file using sklearn's joblib.
    
    Args:
        data (object): Object to serialize.
        
        path (str): Destination path.
        
        compress (int): Compression level; from 0 to 9 (highest compression) (optional).
    
    """
    
    # normalize path
    utils.normpath(path)
    
    joblib.dump(data, path, compress=compress)


def deserialize(path):
    """Deserialize data from a file using sklearn's joblib.
    
    Args:
        path (str): Source path.
    
    Returns:
        data (object): Deserialized object.
    
    """
    
    # normalize path
    path = utils.normpath(path)
    
    return joblib.load(path)


def dumpJSON(data, path):
    """Save JSON data to a file.
    
    Args:
        data (dict): The JSON data to dump.
        
        path (str): Destination path.
    
    """
    
    # normalize path
    path = utils.normpath(path)
    
    with open(path, 'w') as fid:
        json.dump(data, fid)


def loadJSON(path):
    """Load JSON data from a file.
    
    Args:
        path (str): Source path.
    
    Returns:
        data (dict): The loaded JSON data.
    
    """
    
    # normalize path
    path = utils.normpath(path)
    
    with open(path, 'r') as fid:
        return json.load(fid)


def zip_write(fid, files, recursive=True, root=None):
    """Write files to zip archive.
    
    Args:
        fid (file-like object): The zip file to write into.
        
        files (iterable): List of files or directories to pack.
        
        recursive (bool): If True, sub-directories and sub-folders are
                          also written to the archive.
        
        root (str): Relative folder path.
    
    Notes:
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
            rfiles = [os.path.join(fpath, subitem) for subitem in os.listdir(fpath)]
            zip_write(fid, rfiles, recursive=recursive, root=arcname)


def pack_zip(files, path, recursive=True, forceExt=True):
    """Pack files into a zip archive.
    
    Args:
        files (iterable): List of files or directories to pack.
        
        path (str): Destination path.
        
        recursive (bool): If True, sub-directories and sub-folders are
                          also written to the archive.
        
        forceExt (bool): Append default extension.
    
    Returns:
        zip_path (str): Full path to created zip archive.
    
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
    
    Args:
        zip_path (str): Path to zip archive.
        
        path (str): Destination path (directory).
    
    """
    
    # allowZip64 is True to allow files > 2 GB
    with zipfile.ZipFile(zip_path, 'r', allowZip64=True) as fid:
        fid.extractall(path)


def alloc_h5(path):
    """Prepare an HDF5 file.
    
    Args:
        path (str): Path to file.
    
    """
    
    # normalize path
    path = utils.normpath(path)
    
    with h5py.File(path):
        pass


def store_h5(path, label, data):
    """Store data to HDF5 file.
    
    Args:
        path (str): Path to file.
        
        label (hashable): Data label.
        
        data (array): Data to store.
    
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
    
    Args:
        path (str): Path to file.
        
        label (hashable): Data label.
    
    Returns:
        data (array): Loaded data.
    
    """
    
    # normalize path
    path = utils.normpath(path)
    
    with h5py.File(path) as fid:
        label = str(label)
        
        try:
            return fid[label][...]
        except KeyError:
            return None


class HDF(object):
    """Wrapper class to operate on BioSPPy HDF5 files.
    
    Args:
        path (str): Path to the HDF5 file.
        
        mode (str): File mode; one of:
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
                raise IOError("Unable to create 'signals' group with current file mode.")
            self._signals = self._file.create_group('signals')
        
        try:
            self._events = self._file['events']
        except KeyError:
            if mode == 'r':
                raise IOError("Unable to create 'events' group with current file mode.")
            self._events = self._file.create_group('events')
    
    def __enter__(self):
        """Method for with statement."""
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Method for with statement."""
        
        self.close()
    
    def _join_group(self, *args):
        """Join group elements.
        
        Args:
            *args (list): Group elements to joined.
        
        Returns:
            weg (str): Joined group path.
        
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
        
        Args:
            header (dict): Header metadata.
        
        """
        
        # check inputs
        if header is None:
            raise TypeError("Please specify the header information.")
        
        self._file.attrs['json'] = json.dumps(header)
    
    def get_header(self):
        """Retrieve header metadata.
        
        Returns:
            (ReturnTuple): containing:
                header (dict): Header metadata.
        
        """
        
        try:
            header = json.loads(self._file.attrs['json'])
        except KeyError:
            header = {}
        
        return utils.ReturnTuple((header, ), ('header', ))
    
    def add_signal(self, signal=None, mdata=None, group='', name=None, compress=False):
        """Add a signal to the file.
        
        Args:
            signal (array): Signal to add.
            
            mdata (dict): Signal metadata.
            
            group (str): Destination signal group.
            
            name (str): Name of the dataset to create (optional).
            
            compress (bool): If True, the signal will be compressed with gzip (optional).
        
        Returns:
            (ReturnTuple): containing:
                group (str): Destination group.
                
                name (str): Name of the created signal dataset.
        
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
        
        Args:
            group (str): Signal group.
            
            name (str): Name of the signal dataset.
        
        Returns:
            (ReturnTuple): containing:
                dataset (h5py.Dataset): HDF5 dataset.
        
        """
        
        # check inputs
        if name is None:
            raise TypeError("Please specify the name of the signal to retrieve.")
        
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
        
        Args:
            group (str): Signal group.
            
            name (str): Name of the signal dataset.
        
        Returns:
            (ReturnTuple): containing:
                signal (array): Retrieved signal.
                
                mdata (dict): Signal metadata.
        
        Notes:
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
        
        Args:
            group (str): Signal group.
            
            name (str): Name of the dataset.
        
        """
        
        dset = self._get_signal(group=group, name=name)
        
        try:
            del self._file[dset.name]
        except IOError:
            raise IOError("Unable to delete object with current file mode.")
    
    def del_signal_group(self, group=''):
        """Delete all signals in a file group.
        
        Args:
            group (str): Signal group.
        
        """
        
        # navigate to group
        weg = self._join_group(self._signals.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent signal group.")
        
        if node.name == '/signals':
            # delete all elements
            for _, item in node.iteritems():
                try:
                    del self._file[item.name]
                except IOError:
                    raise IOError("Unable to delete object with current file mode.")
        else:
            # delete single node
            try:
                del self._file[node.name]
            except IOError:
                raise IOError("Unable to delete object with current file mode.")
    
    def list_signals(self, group='', recursive=False):
        """List signals in the file.
        
        Args:
            group (str): Signal group.
            
            recursive (bool): It True, also lists signals in sub-groups (optional).
        
        Returns:
            (ReturnTuple): containing:
                signals (list): List of (group, name) tuples of the found signals.
        
        """
        
        # navigate to group
        weg = self._join_group(self._signals.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent signal group.")
        
        out = []
        for name, item in node.iteritems():
            if isinstance(item, h5py.Dataset):
                out.append((group, name))
            elif recursive and isinstance(item, h5py.Group):
                aux = self._join_group(group, name)
                out.extend(self.list_signals(group=aux, recursive=True)['signals'])
        
        return utils.ReturnTuple((out, ), ('signals', ))
    
    def add_event(self, ts=None, values=None, mdata=None, group='', name=None, compress=False):
        """Add an event to the file.
        
        Args:
            ts (array): Array of time stamps.
            
            values (array): Array with data for each time stamp.
            
            mdata (dict): Event metadata.
            
            group (str): Destination event group.
            
            name (str): Name of the dataset to create (optional).
            
            compress (bool): If True, the data will be compressed with gzip (optional).
        
        Returns:
            (ReturnTuple): containing:
                group (str): Destination group.
                
                name (str): Name of the created event dataset.
        
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
            _ = evt_node.create_dataset('values', data=values, compression='gzip')
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
        
        Args:
            group (str): Event group.
            
            name (str): Name of the event dataset.
        
        Returns:
            event (h5py.Group): HDF5 event group.
            
            ts (h5py.Dataset): HDF5 time stamps dataset.
            
            values (h5py.Dataset): HDF5 values dataset.
        
        """
        
        # check inputs
        if name is None:
            raise TypeError("Please specify the name of the signal to retrieve.")
        
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
        
        Args:
            group (str): Event group.
            
            name (str): Name of the event dataset.
        
        Returns:
            (ReturnTuple): containing:
                ts (array): Array of time stamps.
                
                values (array): Array with data for each time stamp.
                
                mdata (dict): Event metadata.
        
        Notes:
            * Loads the entire event data into memory.
        
        """
        
        evt_group, dset_ts, dset_values = self._get_event(group=group, name=name)
        ts = dset_ts[...]
        values = dset_values[...]
        
        try:
            mdata = json.loads(evt_group.attrs['json'])
        except KeyError:
            mdata = {}
        
        return utils.ReturnTuple((ts, values, mdata), ('ts', 'values', 'mdata'))
    
    def del_event(self, group='', name=None):
        """Delete an event from the file.
        
        Args:
            group (str): Event group.
            
            name (str): Name of the event dataset.
        
        """
        
        evt_group, _, _ = self._get_event(group=group, name=name)
        
        try:
            del self._file[evt_group.name]
        except IOError:
            raise IOError("Unable to delete object with current file mode.")
    
    def del_event_group(self, group=''):
        """Delete all events in a file group.
        
        Args:
            group (str): Event group.
        
        """
        
        # navigate to group
        weg = self._join_group(self._events.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent event group.")
        
        if node.name == '/events':
            # delete all elements
            for _, item in node.iteritems():
                try:
                    del self._file[item.name]
                except IOError:
                    raise IOError("Unable to delete object with current file mode.")
        else:
            # delete single node
            try:
                del self._file[node.name]
            except IOError:
                raise IOError("Unable to delete object with current file mode.")
    
    def list_events(self, group='', recursive=False):
        """List events in the file.
        
        Args:
            group (str): Event group.
            
            recursive (bool): It True, also lists events in sub-groups (optional).
        
        Returns:
            (ReturnTuple): containing:
                events (list): List of (group, name) tuples of the found events.
        
        """
        
        # navigate to group
        weg = self._join_group(self._events.name, group)
        try:
            node = self._file[weg]
        except KeyError:
            raise KeyError("Inexistent event group.")
        
        out = []
        for name, item in node.iteritems():
            if isinstance(item, h5py.Group):
                try:
                    _ = item.attrs['json']
                except KeyError:
                    # normal group
                    if recursive:
                        aux = self._join_group(group, name)
                        out.extend(self.list_events(group=aux, recursive=True)['events'])
                else:
                    # event group
                    out.append((group, name))
        
        return utils.ReturnTuple((out, ), ('events', ))
    
    def close(self):
        """Close file descriptor."""
        
        # flush buffers
        self._file.flush()
        
        # close
        self._file.close()

