# -*- coding: utf-8 -*-
"""
    biosppy.biometrics
    ------------------
    
    This module provides classifier interfaces for identity recognition (biometrics)
    applications. The core API methods are:
        * enroll: add a new subject;
        * dismiss: remove an existing subject;
        * identify: determine the identity of collected biometric dataset;
        * authenticate: verify the identity of collected biometric dataset.
    
    :copyright: (c) 2015 by Instituto de Telecomunicacoes
    :license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# built-in
import os

# 3rd party
import numpy as np
from bidict import bidict

# local
from . import storage, utils
from .signals import tools

# Globals


class SubjectError(Exception):
    """Exception raised when the subject is unknown."""
    
    def __init__(self, subject):
        self.subject = subject
    
    def __str__(self):
        return str("Subject %r is not enrolled." % self.subject)


class UntrainedError(Exception):
    """Exception raised when classifier is not trained."""
    
    def __str__(self):
        return str("The classifier is not trained.")


class CombinationError(Exception):
    """Exception raised when the combination method fails."""
    
    def __str__(self):
        return str("Combination of empty array.")


class SubjectDict(bidict):
    """Adaptation of bidirectional dictionary to return default values on KeyError.
    
    Attributes:
        LEFT (hashable): Left default token.
        
        Right (hashable): Right default token.
    
    """
    
    LEFT = ''
    RIGHT = -1
    
    def __getitem__(self, keyorslice):
        """Get an item; based on the bidict source."""
        
        try:
            start, stop, step = keyorslice.start, keyorslice.stop, keyorslice.step
        except AttributeError:
            # keyorslice is a key, e.g. b[key]
            try:
                return self._fwd[keyorslice]
            except KeyError:
                return self.RIGHT

        # keyorslice is a slice
        if (not ((start is None) ^ (stop is None))) or step is not None:
            raise TypeError('Slice must only specify either start or stop')

        if start is not None:
            # forward lookup (by key), e.g. b[key:]
            try:
                return self._fwd[start]
            except KeyError:
                return self.RIGHT

        # inverse lookup (by val), e.g. b[:val]
        assert stop is not None
        try:
            return self._bwd[stop]
        except KeyError:
            return self.LEFT


class BaseClassifier(object):
    """Base biometric classifier class.
    
    Args:
        path (str): Path to working directory; if None, uses in-memory storage (optional).
        
        name (str): Classifier name (optional).
    
    Attributes:
        NAME (str): Classifier name.
        
        EXT (str): Classifier file extension.
        
        EER_IDX (int): Reference index for the Equal Error Rate.
    
    """
    
    NAME = 'BaseClassifier'
    EXT = '.clf'
    EER_IDX = 0
    
    def __init__(self, path=None, name=None):
        # generic self things
        self._reset()
        
        if name is not None:
            self.NAME = name
        
        # choose IO mode
        if path is None:
            # memory-base IO
            self._iomode = 'mem'
            self._iofile = {}
        elif isinstance(path, basestring):
            # file-based IO
            path = utils.normpath(path)
            self._iomode = 'file'
            self._iopath = path
            self._iofile = os.path.join(self._iopath, self.NAME + self.EXT)
            
            # verify path
            if not os.path.exists(path):
                os.makedirs(path)
            
            self._prepareIO()
        else:
            raise ValueError("Unknown path format.")
    
    def _reset(self):
        """Reset the classifier."""
        
        self.subject2label = SubjectDict()
        self.nbSubjects = 0
        self.is_trained = False
        self._thresholds = {}
        self._autoThresholds = None
    
    def _snapshot(self):
        """Get a snapshot of the classifier."""
        
        subject2label = self.subject2label
        nbSubjects = self.nbSubjects
        
        return subject2label, nbSubjects
    
    def _rebrand(self, name):
        """Change classifier name.
        
        Args:
            name (str): New classifier name.
        
        """
        
        if self._iomode == 'file':
            new = self._iofile.replace(self.NAME, name)
            os.rename(self._iofile, new)
            self._iofile = new
        
        self.NAME = name
    
    def _update_fileIO(self, path):
        # update file IO to new path
        
        if self.iomode == 'file':
            self.iopath = path
            self.iofile = os.path.join(path, self.NAME + self.EXT)
    
    def _prepareIO(self):
        # create dirs, initialize files
        
        storage.alloc_h5(self._iofile)
    
    def io_load(self, label):
        """Load enrolled subject data.
        
        Args:
            label (int): Internal classifier subject label.
        
        Returns:
            data (array): Subject data.
        
        """
        
        if self.iomode == 'file':
            return self._fileIO_load(label)
        elif self.iomode == 'mem':
            return self._memIO_load(label)
    
    def io_save(self, label, data):
        """Save subject data.
        
        Args:
            label (int): Internal classifier subject label.
            
            data (array): Subject data.
        
        """
        
        if self.iomode == 'file':
            self._fileIO_save(label, data)
        elif self.iomode == 'mem':
            self._memIO_save(label, data)
    
    def _fileIO_load(self, label):
        """Load data in file mode.
        
        Args:
            label (int): Internal classifier subject label.
        
        Returns:
            data (array): Subject data.
        
        """
        
        data = storage.load_h5(self._iofile, label)
        
        return data
    
    def _fileIO_save(self, label, data):
        """Save data in file mode.
        
        Args:
            label (int): Internal classifier subject label.
            
            data (array): Subject data.
        
        """
        
        storage.store_h5(self._iofile, label, data)
    
    def _memIO_load(self, label):
        """Load data in memory mode.
        
        Args:
            label (int): Internal classifier subject label.
        
        Returns:
            data (array): Subject data.
        
        """
        
        return self._iofile[label]
    
    def _memIO_save(self, label, data):
        """Save data in memory mode.
        
        Args:
            label (int): Internal classifier subject label.
            
            data (array): Subject data.
        
        """
        
        self._iofile[label] = data
    
    def io_iterator(self):
        """Iterate over the files used by the classifier.
        
        Returns:
            files (iterator): Iterator over the classifier files.
        
        """
        
        yield self.NAME + self.EXT
    
    
    def fileIterator(self):
        # iterator for the classifier files
        
        yield self.NAME + self.EXT
    
    def dirIterator(self):
        # iterator for the directories
        
        return iter([])
    
    def save(self, dstPath):
        # save the classifier to the path
        
        # classifier files
        if self.iomode == 'file':
            tmpPath = os.path.join(self.iopath, 'clf-tmp')
            if not os.path.exists(tmpPath):
                os.makedirs(tmpPath)
            
            # dirs
            for d in self.dirIterator():
                path = os.path.join(tmpPath, d)
                if not os.path.exists(path):
                    os.makedirs(path)
            
            # files
            for f in self.fileIterator():
                src = os.path.join(self.iopath, f)
                dst = os.path.join(tmpPath, f)
                try:
                    shutil.copy(src, dst)
                except IOError:
                    pass
        else:
            tmpPath = os.path.abspath(os.path.expanduser('~/clf-tmp'))
            if not os.path.exists(tmpPath):
                os.makedirs(tmpPath)
        
        # save classifier instance to temp file
        datamanager.skStore(os.path.join(tmpPath, 'clfInstance.p'), self)
        
        # save to zip archive
        dstPath = os.path.join(dstPath, self.NAME)
        datamanager.zipArchiveStore(tmpPath, dstPath)
        
        # remove temp dir
        shutil.rmtree(tmpPath, ignore_errors=True)
        
        return dstPath
    
    @classmethod
    def load(cls, srcPath, dstPath=None):
        # load a classifier instance from a file
        # do not include the extension in the path
        
        if dstPath is None:
            dstPath, _ = os.path.split(srcPath)
        
        # unzip files
        datamanager.zipArchiveLoad(srcPath, dstPath)
        
        # load classifier
        tmpPath = os.path.join(dstPath, 'clfInstance.p')
        clf = datamanager.skLoad(tmpPath)
        
        # classifier files
        clf._update_fileIO(dstPath)
        
        # remove temp file
        os.remove(tmpPath)
        
        if not isinstance(clf, cls):
            raise TypeError, "Mismatch between target class and loaded file."
        
        return clf
    
    def checkSubject(self, subject):
        """Check if a subject is enrolled.
        
        Args:
            subject (hashable): Subject identity.
        
        Returns:
            check (bool): If True, the subject is enrolled.
        
        """
        
        if self.is_trained:
            return subject in self.subject2label
        
        return False
    
    def listSubjects(self):
        """List all the enrolled subjects.
        
        Returns:
            subjects (list): Enrolled subjects.
        
        """
        
        subjects = [self.subject2label[:i] for i in xrange(self.nbSubjects)]
        
        return subjects
    
    def _prepareData(self, data):
        # prepare date
        ### user
        
        return data
    
    def _updateStrategy(self, oldData, newData):
        # update the training data of a class when new data is available
        
        return newData
    
    def authThreshold(self, subject, ready=False):
        # get the user threshold (authentication)
        
        if not ready:
            aux = subject
            subject = self.subject2label[subject]
            if subject == -1:
                raise SubjectError(aux)
        
        return self.thresholds[subject]['auth']
    
    def setAuthThreshold(self, subject, threshold, ready=False):
        # set the user threshold (authentication)
        
        if not ready:
            aux = subject
            subject = self.subject2label[subject]
            if subject == -1:
                raise SubjectError(aux)
        try:
            self.thresholds[subject]['auth'] = threshold
        except KeyError:
            self.thresholds[subject] = {'auth': threshold, 'id': None}
    
    def idThreshold(self, subject, ready=False):
        # get the user threshold (identification)
        
        if not ready:
            aux = subject
            subject = self.subject2label[subject]
            if subject == -1:
                raise SubjectError(aux)
        
        return self.thresholds[subject]['id']
    
    def setIdThreshold(self, subject, threshold, ready=False):
        # set the user threshold (identification)
        
        if not ready:
            aux = subject
            subject = self.subject2label[subject]
            if subject == -1:
                raise SubjectError(aux)
        try:
            self.thresholds[subject]['id'] = threshold
        except KeyError:
            self.thresholds[subject] = {'auth': None, 'id': threshold}
    
    def autoRejectionThresholds(self):
        # generate thresholds automatically
        ### user
        
        if self._autoThresholds is not None:
            return self._autoThresholds
        
        ths = np.array([])
        
        return ths
    
    def _subThrIterator(self, overwrite):
        # iterate over the subjects in order to update the thresholds
        
        if overwrite:
            for i in xrange(self.nbSubjects):
                yield i
        else:
            for i in xrange(self.nbSubjects):
                try:
                    _ = self.authThreshold(i, ready=True)
                except KeyError:
                    yield i
    
    def updateThresholds(self, overwrite=False, fraction=1.):
        # update the user thresholds based on the enrolled data
        
        ths = self.autoRejectionThresholds()
        
        # gather data to test
        data = {}
        for lbl in self._subThrIterator(overwrite):
            subject = self.subject2label[:lbl]
            
            # select a random fraction of the training data
            aux = self.io_load(lbl)
            indx = range(len(aux))
            use, _ = parted.randomFraction(indx, 0, fraction)
            
            data[subject] = aux[use]
        
        # evaluate classifier
        if len(data.keys()) > 42:
            out = self.evaluate(data, ths)
        else:
            out = self.seqEvaluate(data, ths)
        
        # choose thresholds at EER
        for lbl in self._subThrIterator(overwrite):
            subject = self.subject2label[:lbl]
            
            EER_auth = out['assessment']['subject'][subject]['authentication']['rates']['EER']
            self.setAuthThreshold(lbl, EER_auth[self.EER_IDX, 0], ready=True)
             
            EER_id = out['assessment']['subject'][subject]['identification']['rates']['EER']
            self.setIdThreshold(lbl, EER_id[self.EER_IDX, 0], ready=True)
    
    def train(self, data=None, updateThresholds=True):
        # data is {subject: features (array)}
        ### user
        
        # check inputs
        if data is None:
            raise TypeError, "Please provide input data."
        
        # check if classifier was already trained
        if self.is_trained:
            # this is a retrain
            self.re_train(data)
        else:
            # get subjects
            subjects = data.keys()
            self.nbSubjects = len(subjects)
            
            # determine classifier mode
            if self.nbSubjects == 0:
                raise ValueError, "Please provide input data - empty dict."
            else:
                for i in xrange(self.nbSubjects):
                    # build dicts
                    sub = subjects[i]
                    self.subject2label[sub] = i
                    
                    # save data
                    self.io_save(i, data[sub])
                    
                    # prepare data
                    _ = self._prepareData(data[sub])
                    
                    ### user
        
        # train flag
        self.is_trained = True
        
        if updateThresholds:
            # update thresholds
            self.updateThresholds()
    
    def re_train(self, data):
        # data is {subject: features (array)}
        ### user
        
        for sub in data.iterkeys():
            if sub in self.subject2label:
                # existing subject
                if data[sub] is not None:
                    # change subject's data
                    label = self.subject2label[sub]
                    
                    # update templates
                    aux = self._updateStrategy(self.io_load(label), data[sub])
                    
                    # save data
                    self.io_save(label, aux)
                    
                    # prepare data
                    _ = self._prepareData(aux)
                    
                    ### user
                    
                    
                else:
                    # delete subject
                    if self.nbSubjects == 1:
                        # reset classifier to untrained
                        self._reset()
                    else:
                        # reorder the labels/subjects and models dicts
                        subject2label, nbSubjects = self._snapshot()
                        self._reset()
                        
                        label = subject2label[sub]
                        clabels = np.setdiff1d(np.unique(subject2label.values()),
                                                  [label], assume_unique=True)
                        self.nbSubjects = nbSubjects - 1
                        
                        i = 0
                        for ii in xrange(len(clabels)):
                            # build dicts
                            sub = subject2label[:clabels[ii]]
                            self.subject2label[sub] = i
                            
                            # move data
                            self.io_save(i, self.io_load(clabels[ii]))
                            
                            ### user
                            
                            # update i
                            i += 1
            else:
                # new subject
                # add to dicts
                label = self.nbSubjects
                self.subject2label[sub] = label
                
                # save data
                self.io_save(label, data[sub])
                
                # prepare data
                _ = self._prepareData(data[sub])
                
                ### user
                
                
                # increment number of subjects
                self.nbSubjects += 1
    
    def authenticate(self, data, subject, threshold=None, ready=False, labels=False, **kwargs):
        # data is a list of feature vectors, allegedly belonging to the given subject
        ### user
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # translate subject ID to class label
        label = self.subject2label[subject]
        if label == -1:
            raise SubjectError(subject)
        
        if threshold is None:
            # get user-tuned threshold
            threshold = self.authThreshold(label, ready=True)
        
        # prepare data
        if not ready:
            _ = self._prepareData(data)
        else:
            _ = data
        
        # outputs
        decision = []
        prediction = []
        
        ### user
        
        # convert to numpy
        decision = np.array(decision)
        
        if labels:
            # translate class label to subject ID
            subPrediction = [self.subject2label[:item] for item in prediction]
            return decision, subPrediction
        else:
            return decision
    
    def _identify(self, data, threshold=None, ready=False):
        # data is list of feature vectors
        ### user
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # threshold
        if threshold is None:
            _ = lambda label: self.idThreshold(label, ready=True)
        else:
            _ = lambda label: threshold
        
        # prepare data
        if not ready:
            _ = self._prepareData(data)
        else:
            _ = data
        
        # outputs
        labels = []
        
        ### user
        
        return np.array(labels)
    
    def identify(self, data, threshold=None, ready=False, **kwargs):
        # data is list of feature vectors
        
        labels = self._identify(data=data, threshold=threshold, ready=ready, **kwargs)
        
        # translate class labels to subject IDs
        subjects = [self.subject2label[:item] for item in labels]
        
        return subjects
    
    def seqEvaluate(self, data, thresholds=None):
        """
        Assess the performance of the classifier in both biometric scenarios: authentication and identification.
        
        Workflow:
            For each test subject and for each threshold, test authentication and identification;
            Authentication results stored in a 3 dimensional array of booleans, shape = (N thresholds, M subjects, K samples);
            Identification results stored in a 2 dimensional array, shape = (N thresholds, K samples);
            Subject and global statistics are then computed by evaluation.assessClassification.
        
        Kwargs:
            data (dict): Dictionary holding the testing samples for each subject.
            
            rejection_thresholds (array): Thresholds used to compute the ROCs.
            
            dstPath (string): Path for multiprocessing.
            
            log2file (bool): Flag to control the use of logging in multiprocessing.
        
        Kwrvals:
            classification (dict): Results of the classification.
            
            assessment (dict): Biometric statistics.
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        # data is {subject: features (array)}
        
        # check train state
        if not self.is_trained:
            raise UntrainedError
        
        # choose thresholds
        if thresholds is None:
            thresholds = self.autoRejectionThresholds()
        nth = len(thresholds)
        
        subjects = data.keys()
        results = {'subjectList': subjects,
                   'subjectDict': self.subject2label,
                   }
        
        for subject in subjects:
            # prepare data
            aux = self._prepareData(data[subject])
            
            # test
            auth_res = []
            id_res = []
            for i in xrange(nth):
                th = thresholds[i]
                
                auth = []
                for subject_tst in subjects:
                    auth.append(self.authenticate(aux, subject_tst, th, ready=True))
                auth_res.append(np.array(auth))
                
                id_res.append(self._identify(aux, th, ready=True))
            
            auth_res = np.array(auth_res)
            id_res = np.array(id_res)
            results[subject] = {'authentication': auth_res,
                                'identification': id_res,
                                }
        
        # assess classification results
        assess, = assess_classification(results, thresholds)
        
        # final output
        output = {'classification': results,
                  'assessment': assess,
                  }
        
        return output


def get_auth_rates(TP=None, FP=None, TN=None, FN=None, thresholds=None):
    """Compute authentication rates from the confusion matrix.
    
    Args:
        TP (array): True Positive counts for each classifier threshold.
        
        FP (array): False Positive counts for each classifier threshold.
        
        TN (array): True Negative counts for each classifier threshold.
        
        FN (array): False Negative counts for each classifier threshold.
        
        thresholds (array): Classifier thresholds.
    
    Returns:
        Acc (array): Accuracy at each classifier threshold.
        
        TAR (array): True Accept Rate at each classifier threshold.
        
        FAR (array): False Accept Rate at each classifier threshold.
        
        FRR (array): False Reject Rate at each classifier threshold.
        
        TRR (array): True Reject Rate at each classifier threshold.
        
        EER (array): Equal Error Rate points, with format (threshold, rate).
    
    """
    
    # check inputs
    if TP is None:
        raise TypeError("Please specify the input TP counts.")
    if FP is None:
        raise TypeError("Please specify the input FP counts.")
    if TN is None:
        raise TypeError("Please specify the input TN counts.")
    if FN is None:
        raise TypeError("Please specify the input FN counts.")
    if thresholds is None:
        raise TypeError("Please specify the input classifier thresholds.")
    
    # ensure numpy
    TP = np.array(TP)
    FP = np.array(FP)
    TN = np.array(TN)
    FN = np.array(FN)
    thresholds = np.array(thresholds)
    
    Acc = (TP + TN) / (TP + TN + FP + FN)
    
    # should accept counts
    SA = TP + FN
    
    # should reject counts
    SR = TN + FP
    
    # avoid division by zero
    SA[SA <= 0] = 1.
    SR[SR <= 0] = 1.
    
    TAR = TP / SA
    FAR = FP / SR
    FRR = FN / SA
    TRR = TN / SR
    
    # determine EER
    roots, values = tools.find_intersection(thresholds, FAR, thresholds, FRR)
    EER = np.vstack((roots, values)).T
    
    # output
    args = (Acc, TAR, FAR, FRR, TRR, EER)
    names = ('Acc', 'TAR', 'FAR', 'FRR', 'TRR', 'EER')
    
    return utils.ReturnTuple(args, names)


def get_id_rates(H=None, M=None, R=None, N=None, thresholds=None):
    """Compute identification rates from the confusion matrix.
    
    Args:
        H (array): Hit counts for each classifier threshold.
        
        M (array): Miss counts for each classifier threshold.
        
        R (array): Reject counts for each classifier threshold.
        
        N (int): Number of test samples.
        
        thresholds (array): Classifier thresholds.
    
    Returns:
        Acc (array): Accuracy at each classifier threshold.
        
        Err (array): Error rate at each classifier threshold.
        
        MR (array): Miss Rate at each classifier threshold.
        
        RR (array): Reject Rate at each classifier threshold.
        
        EID (array): Error of Identification points, with format (threshold, rate).
        
        EER (array): Equal Error Rate points, with format (threshold, rate).
    
    """
    
    # check inputs
    if H is None:
        raise TypeError("Please specify the input H counts.")
    if M is None:
        raise TypeError("Please specify the input M counts.")
    if R is None:
        raise TypeError("Please specify the input R counts.")
    if N is None:
        raise TypeError("Please specify the total number of test samples.")
    if thresholds is None:
        raise TypeError("Please specify the input classifier thresholds.")
    
    # ensure numpy
    H = np.array(H)
    M = np.array(M)
    R = np.array(R)
    thresholds = np.array(thresholds)
    
    Acc = H / N
    Err = 1 - Acc
    MR = M / N
    RR = R / N
    
    # EER
    roots, values = tools.find_intersection(thresholds, MR, thresholds, RR)
    EER = np.vstack((roots, values)).T
    
    # EID
    y2 = np.min(Err) * np.ones(len(thresholds), dtype='float')
    roots, values = tools.find_intersection(thresholds, Err, thresholds, y2)
    EID = np.vstack((roots, values)).T
    
    # output
    args = (Acc, Err, MR, RR, EID, EER)
    names = ('Acc', 'Err', 'MR', 'RR', 'EID', 'EER')
    
    return utils.ReturnTuple(args, names)


def get_subject_results(results=None, subject=None, thresholds=None, subjects=None,
                        subject_dict=None, subject_idx=None):
    """Compute authentication and identification performance metrics for a given subject.
    
    Args:
        results (dict): Classification results.
        
        subject (hashable): True subject label.
        
        thresholds (array): Classifier thresholds.
        
        subjects (list): Target subject classes.
        
        subject_dict (SubjectDict): Subject-label conversion dictionary.
        
        subject_idx (list): Subject index.
    
    Returns:
        assessment (dict): Authentication and identification results.
    
    """
    
    # check inputs
    if results is None:
        raise TypeError("Please specify the input classification results.")
    if subject is None:
        raise TypeError("Please specify the input subject class.")
    if thresholds is None:
        raise TypeError("Please specify the input classifier thresholds.")
    if subjects is None:
        raise TypeError("Please specify the target subject classes.")
    if subject_dict is None:
        raise TypeError("Please specify the subject-label dictionary.")
    if subject_idx is None:
        raise TypeError("Plase specify subject index.")
    
    nth = len(thresholds)
    auth_res = results['authentication']
    id_res = results['identification']
    ns = auth_res.shape[2]
    
    # sanity checks
    if auth_res.shape[0] != id_res.shape[0]:
        raise ValueError("Authentication and identification number of thresholds do not match.")
    if auth_res.shape[0] != nth:
        raise ValueError("Number of thresholds in vector does not match biometric results.")
    if auth_res.shape[2] != id_res.shape[1]:
        raise ValueError("Authentication and identification number of tests do not match.")
    
    label = subject_dict[subject]
    
    # authentication vars
    TP = np.zeros(nth, dtype='float')
    FP = np.zeros(nth, dtype='float')
    TN = np.zeros(nth, dtype='float')
    FN = np.zeros(nth, dtype='float')
    
    # identification vars
    H = np.zeros(nth, dtype='float')
    M = np.zeros(nth, dtype='float')
    R = np.zeros(nth, dtype='float')
    CM = []
    
    for i in xrange(nth):  # for each threshold
        # authentication
        for k, lbl in enumerate(subject_idx):  # for each subject
            subject_tst = subjects[k]
            
            d = auth_res[i, lbl, :]
            if subject == subject_tst:
                # true positives
                aux = np.sum(d)
                TP[i] += aux
                # false negatives
                FN[i] += (ns - aux)
            else:
                # false positives
                aux = np.sum(d)
                FP[i] += aux
                # true negatives
                TN[i] += (ns - aux)
        
        # identification
        res = id_res[i, :]
        hits = res == label
        nhits = np.sum(hits)
        rejects = res == -1
        nrejects = np.sum(rejects)
        misses = np.logical_not(np.logical_or(hits, rejects))
        nmisses = ns - (nhits + nrejects)
        missCounts = {subject_dict[:ms]: np.sum(res == ms) for ms in np.unique(res[misses])}
        
        # appends
        H[i] = nhits
        M[i] = nmisses
        R[i] = nrejects
        CM.append(missCounts)
    
    output = {'authentication': {'confusionMatrix': {'TP': TP,
                                                     'FP': FP,
                                                     'TN': TN,
                                                     'FN': FN,
                                                     },
                                 'rates': get_auth_rates(TP, FP, TN, FN, thresholds).as_dict(),
                                 },
              'identification': {'confusionMatrix': {'H': H,
                                                     'M': M,
                                                     'R': R,
                                                     'CM': CM,
                                                     },
                                 'rates': get_id_rates(H, M, R, ns, thresholds).as_dict(),
                                 },
              }
    
    return utils.ReturnTuple((output, ), ('assessment', ))


def assess_classification(results=None, thresholds=None):
    """Assess the performance of a biometric classification test.
    
    Args:
        results (dict): Classification results.
        
        thresholds (array): Classifier thresholds.
    
    Returns:
        assessment (dict): Classification assessment.
    
    """
    
    # check inputs
    if results is None:
        raise TypeError("Please specify the input classification results.")
    if thresholds is None:
        raise TypeError("Please specify the input classifier thresholds.")
    
    # test subjects
    subjectDict = results['subjectDict']
    subParent = results['subjectList']
    subIdx = [subParent.index(item) for item in subParent]
    subIdx.sort()
    subjects = [subParent[item] for item in subIdx]
    
    # output object
    output = {'global': {'authentication': {'confusionMatrix': {'TP': 0.,
                                                                'TN': 0.,
                                                                'FP': 0.,
                                                                'FN': 0.,
                                                                },
                                            },
                         'identification': {'confusionMatrix': {'H': 0.,
                                                                'M': 0.,
                                                                'R': 0.,
                                                                },
                                            },
                         },
              'subject': {},
              'thresholds': thresholds,
              }
    
    nth = len(thresholds)
    C = np.zeros((nth, len(subjects)), dtype='float')
    
    # update variables
    auth = output['global']['authentication']['confusionMatrix']
    authM = ['TP', 'TN', 'FP', 'FN']
    iden = output['global']['identification']['confusionMatrix']
    idenM = ['H', 'M', 'R']
    
    for test_user in subjects:
        aux, = get_subject_results(results[test_user], test_user, thresholds,
                                   subjects, subjectDict, subIdx)
        
        # copy to subject
        output['subject'][test_user] = aux
        
        # authentication
        for m in authM:
            auth[m] += aux['authentication']['confusionMatrix'][m]
        
        # identification
        for m in idenM:
            iden[m] += aux['identification']['confusionMatrix'][m]
        
        # subject misses
        for i, item in enumerate(aux['identification']['confusionMatrix']['CM']):
            for k, sub in enumerate(subjects):
                try:
                    C[i, k] += item[sub]
                except KeyError:
                    pass
    
    # normalize subject misses
    sC = C.sum(axis=1).reshape((nth, 1))
    # avoid division by zero
    sC[sC <= 0] = 1.
    CR = C / sC
    
    # update subjects
    for k, sub in enumerate(subjects):
        output['subject'][sub]['identification']['confusionMatrix']['C'] = C[:, k]
        output['subject'][sub]['identification']['rates']['CR'] = CR[:, k]
    
    # compute global rates
    aux = get_auth_rates(auth['TP'], auth['FP'], auth['TN'], auth['FN'], thresholds)
    output['global']['authentication']['rates'] = aux.as_dict()
    
    # identification
    Ns = iden['H'] + iden['M'] + iden['R']
    aux = get_id_rates(iden['H'], iden['M'], iden['R'], Ns, thresholds)
    output['global']['identification']['rates'] = aux.as_dict()
    
    return utils.ReturnTuple((output, ), ('assessment', ))


def assess_runs(results=None, subjects=None):
    """Assess the performance of multiple biometric classification runs.
    
    Args:
        results (list): Classification results for each run.
        
        subjects (list): Common target subject classes.
    
    Returns:
        assessment (dict): Global classification assessment.
    
    """
    
    # check inputs
    if results is None:
        raise TypeError("Please specify the input classification results.")
    if subjects is None:
        raise TypeError("Please specify the common subject classes.")
    
    nb = len(results)
    if nb == 0:
        raise ValueError("Please provide at least one classification run.")
    elif nb == 1:
        return utils.ReturnTuple((results[0], ), ('assessment', ))
    
    # output
    output = {'global': {'authentication': {'confusionMatrix': {'TP': 0.,
                                                                'TN': 0.,
                                                                'FP': 0.,
                                                                'FN': 0.,
                                                                },
                                            },
                         'identification': {'confusionMatrix': {'H': 0.,
                                                                'M': 0.,
                                                                'R': 0.,
                                                                },
                                            },
                         },
              'subject': {},
              'thresholds': None,
              }
    
    thresholds = output['thresholds'] = results[0]['thresholds']
    
    # global helpers
    auth = output['global']['authentication']['confusionMatrix']
    iden = output['global']['identification']['confusionMatrix']
    authM = ['TP', 'TN', 'FP', 'FN']
    idenM1 = ['H', 'M', 'R', 'C']
    idenM2 = ['H', 'M', 'R']
    
    for sub in subjects:
        # create subject confusion matrix, rates
        output['subject'][sub] = {'authentication': {'confusionMatrix': {'TP': 0.,
                                                                         'TN': 0.,
                                                                         'FP': 0.,
                                                                         'FN': 0.,
                                                                         },
                                                     'rates': {},
                                                     },
                                  'identification': {'confusionMatrix': {'H': 0.,
                                                                         'M': 0.,
                                                                         'R': 0.,
                                                                         'C': 0.,
                                                                         },
                                                     'rates': {},
                                                     },
                                  }
        
        # subject helpers
        authS = output['subject'][sub]['authentication']['confusionMatrix']
        idenS = output['subject'][sub]['identification']['confusionMatrix']
        
        # update confusions
        for run in results:
            # authentication
            auth_run = run['subject'][sub]['authentication']['confusionMatrix']
            for m in authM:
                auth[m] += auth_run[m]
                authS[m] += auth_run[m]
            
            # identification
            iden_run = run['subject'][sub]['identification']['confusionMatrix']
            for m in idenM1:
                idenS[m] += iden_run[m]
            for m in idenM2:
                iden[m] += iden_run[m]
        
        # compute subject mean
        # authentication
        for m in authM:
            authS[m] /= float(nb)
        
        # identification
        for m in idenM1:
            idenS[m] /= float(nb)
        
        # compute subject rates
        aux = get_auth_rates(authS['TP'], authS['FP'], authS['TN'], authS['FN'], thresholds)
        output['subject'][sub]['authentication']['rates'] = aux.as_dict()
        
        Ns = idenS['H'] + idenS['M'] + idenS['R']
        aux = get_id_rates(idenS['H'], idenS['M'], idenS['R'], Ns, thresholds)
        output['subject'][sub]['identification']['rates'] = aux.as_dict()
        M = np.array(idenS['M'], copy=True)
        M[M <= 0] = 1.
        output['subject'][sub]['identification']['rates']['CR'] = idenS['C'] / M
    
    # compute global mean
    # authentication
    for m in authM:
        auth[m] /= float(nb)
    
    # identification
    for m in idenM2:
        iden[m] /= float(nb)
    
    # compute rates
    aux = get_auth_rates(auth['TP'], auth['FP'], auth['TN'], auth['FN'], thresholds)
    output['global']['authentication']['rates'] = aux.as_dict()
    
    Ns = iden['H'] + iden['M'] + iden['R']
    aux = get_id_rates(iden['H'], iden['M'], iden['R'], Ns, thresholds)
    output['global']['identification']['rates'] = aux.as_dict()
    
    return utils.ReturnTuple((output, ), ('assessment', ))


def combination(results=None, weights=None):
    """Combine results from multiple classifiers.
    
    Args:
        results (dict): Results for each classifier.
        
        weights (dict): Weight for each classifier (optional).
    
    Returns:
        decision (bool, int, str): Consensus decision.
        
        confidence (float): Confidence estimate of the decision.
        
        counts (array): Weight for each possible decision outcome.
        
        classes (array): List of possible decision outcomes.
    
    """
    
    # check inputs
    if results is None:
        raise TypeError("Please specify the input classification results.")
    if weights is None:
        weights = {}
    
    # compile results to find all classes
    vec = results.values()
    if len(vec) == 0:
        raise CombinationError("No keys found.")
    
    unq = np.unique(np.concatenate(vec))
    
    nb = len(unq)
    if nb == 0:
        # empty array
        raise CombinationError("No values found.")
    elif nb == 1:
        # unanimous result
        decision = unq[0]
        confidence = 1.
        counts = [1.]
    else:
        # multi-class
        counts = np.zeros(nb, dtype='float')
        
        for n in results.iterkeys():
            # ensure array
            res = np.array(results[n])
            ns = float(len(res))
            
            # get count for each unique class
            for i in xrange(nb):
                aux = float(np.sum(res == unq[i]))
                w = weights.get(n, 1.)
                counts[i] += ((aux / ns) * w)
        
        # most frequent class
        predMax = counts.argmax()
        counts /= counts.sum()
        
        decision = unq[predMax]
        confidence = counts[predMax]
    
    # output
    args = (decision, confidence, counts, unq)
    names = ('decision', 'confidence', 'counts', 'classes')
    
    return utils.ReturnTuple(args, names)









