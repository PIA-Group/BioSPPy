# -*- coding: utf-8 -*-
"""
biosppy.biometrics
------------------

This module provides classifier interfaces for identity recognition
(biometrics) applications. The core API methods are:
* enroll: add a new subject;
* dismiss: remove an existing subject;
* identify: determine the identity of collected biometric dataset;
* authenticate: verify the identity of collected biometric dataset.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range
import six

# built-in
import collections

# 3rd party
import numpy as np
import shortuuid
from bidict import bidict
from sklearn import model_selection as skcv
from sklearn import svm as sksvm

# local
from . import metrics, plotting, storage, utils
from .signals import tools


class SubjectError(Exception):
    """Exception raised when the subject is unknown."""

    def __init__(self, subject=None):
        self.subject = subject

    def __str__(self):
        if self.subject is None:
            return str("Subject is not enrolled.")
        else:
            return str("Subject %r is not enrolled." % self.subject)


class UntrainedError(Exception):
    """Exception raised when classifier is not trained."""

    def __str__(self):
        return str("The classifier is not trained.")


class CombinationError(Exception):
    """Exception raised when the combination method fails."""

    def __str__(self):
        return str("Combination of empty array.")


class BaseClassifier(object):
    """Base biometric classifier class.

    This class is a skeleton for actual classifier classes.
    The following methods must be overridden or adapted to build a
    new classifier:
    
    * __init__
    * _authenticate
    * _get_thresholds
    * _identify
    * _prepare
    * _train
    * _update

    Attributes
    ----------
    EER_IDX : int
        Reference index for the Equal Error Rate.

    """

    EER_IDX = 0

    def __init__(self):
        # generic self things
        self.is_trained = False
        self._subject2label = bidict()
        self._nbSubjects = 0
        self._thresholds = {}
        self._autoThresholds = None

        # init data storage
        self._iofile = {}

        # defer flag
        self._defer_flag = False
        self._reset_defer()

    def _reset_defer(self):
        """Reset defer buffer."""

        self._defer_dict = {'enroll': set(), 'dismiss': set()}

    def _defer(self, label, case):
        """Add deferred task.

        Parameters
        ----------
        label : str
            Internal classifier subject label.
        case : str
            One of 'enroll' or 'dismiss'.

        Notes
        -----
        * An enroll overrides a previous dismiss for the same subject.
        * A dismiss overrides a previous enroll for the same subject.

        """

        if case == 'enroll':
            self._defer_dict['enroll'].add(label)
            if label in self._defer_dict['dismiss']:
                self._defer_dict['dismiss'].remove(label)
        elif case == 'dismiss':
            self._defer_dict['dismiss'].add(label)
            if label in self._defer_dict['enroll']:
                self._defer_dict['enroll'].remove(label)

        self._defer_flag = True

    def _check_state(self):
        """Check and update the train state."""

        if self._nbSubjects > 0:
            self.is_trained = True
        else:
            self.is_trained = False

    def io_load(self, label):
        """Load enrolled subject data.

        Parameters
        ----------
        label : str
            Internal classifier subject label.

        Returns
        -------
        data : array
            Subject data.

        """

        return self._iofile[label]

    def io_save(self, label, data):
        """Save subject data.

        Parameters
        ----------
        label : str
            Internal classifier subject label.
        data : array
            Subject data.

        """

        self._iofile[label] = data

    def io_del(self, label):
        """Delete subject data.

        Parameters
        ----------
        label : str
            Internal classifier subject label.

        """

        del self._iofile[label]

    def save(self, path):
        """Save classifier instance to a file.

        Parameters
        ----------
        path : str
            Destination file path.

        """

        storage.serialize(self, path)

    @classmethod
    def load(cls, path):
        """Load classifier instance from a file.

        Parameters
        ----------
        path : str
            Source file path.

        Returns
        -------
        clf : object
            Loaded classifier instance.

        """

        # load classifier
        clf = storage.deserialize(path)

        # check class type
        if not isinstance(clf, cls):
            raise TypeError("Mismatch between target class and loaded file.")

        return clf

    def check_subject(self, subject):
        """Check if a subject is enrolled.

        Parameters
        ----------
        subject : hashable
            Subject identity.

        Returns
        -------
        check : bool
            If True, the subject is enrolled.

        """

        if self.is_trained:
            return subject in self._subject2label

        return False

    def list_subjects(self):
        """List all the enrolled subjects.

        Returns
        -------
        subjects : list
            Enrolled subjects.

        """

        subjects = list(self._subject2label)

        return subjects

    def enroll(self, data=None, subject=None, deferred=False):
        """Enroll new data for a subject.

        If the subject is already enrolled, new data is combined with
        existing data.

        Parameters
        ----------
        data : array
            Data to enroll.
        subject : hashable
            Subject identity.
        deferred : bool, optional
            If True, computations are delayed until `flush` is called.

        Notes
        -----
        * When using deferred calls, an enroll overrides a previous dismiss
          for the same subject.

        """

        # check inputs
        if data is None:
            raise TypeError("Please specify the data to enroll.")

        if subject is None:
            raise TypeError("Plase specify the subject identity.")

        if self.check_subject(subject):
            # load existing
            label = self._subject2label[subject]
            old = self.io_load(label)

            # combine data
            data = self._update(old, data)
        else:
            # create new label
            label = shortuuid.uuid()
            self._subject2label[subject] = label
            self._nbSubjects += 1

        # store data
        self.io_save(label, data)

        if deferred:
            # delay computations
            self._defer(label, 'enroll')
        else:
            self._train([label], None)
            self._check_state()
            self.update_thresholds()

    def dismiss(self, subject=None, deferred=False):
        """Remove a subject.

        Parameters
        ----------
        subject : hashable
            Subject identity.
        deferred : bool, optional
            If True, computations are delayed until `flush` is called.

        Raises
        ------
        SubjectError
            If the subject to remove is not enrolled.

        Notes
        -----
        * When using deferred calls, a dismiss overrides a previous enroll
          for the same subject.

        """

        # check inputs
        if subject is None:
            raise TypeError("Please specify the subject identity.")

        if not self.check_subject(subject):
            raise SubjectError(subject)

        label = self._subject2label[subject]
        del self._subject2label[subject]
        del self._thresholds[label]
        self._nbSubjects -= 1
        self.io_del(label)

        if deferred:
            self._defer(label, 'dismiss')
        else:
            self._train(None, [label])
            self._check_state()
            self.update_thresholds()

    def batch_train(self, data=None):
        """Train the classifier in batch mode.

        Parameters
        ----------
        data : dict
            Dictionary holding training data for each subject; if the object
            for a subject is `None`, performs a `dismiss`.

        """

        # check inputs
        if data is None:
            raise TypeError("Please specify the data to train.")

        for sub, val in six.iteritems(data):
            if val is None:
                try:
                    self.dismiss(sub, deferred=True)
                except SubjectError:
                    continue
            else:
                self.enroll(val, sub, deferred=True)

        self.flush()

    def flush(self):
        """Flush deferred computations."""

        if self._defer_flag:
            self._defer_flag = False

            # train
            enroll = list(self._defer_dict['enroll'])
            dismiss = list(self._defer_dict['dismiss'])
            self._train(enroll, dismiss)

            # update thresholds
            self._check_state()
            self.update_thresholds()

            # reset
            self._reset_defer()

    def update_thresholds(self, fraction=1.):
        """Update subject-specific thresholds based on the enrolled data.

        Parameters
        ----------
        fraction : float, optional
            Fraction of samples to select from training data.

        """

        ths = self.get_thresholds(force=True)

        # gather data to test
        data = {}
        for subject, label in six.iteritems(self._subject2label):
            # select a random fraction of the training data
            aux = self.io_load(label)
            indx = list(range(len(aux)))
            use, _ = utils.random_fraction(indx, fraction, sort=True)

            data[subject] = aux[use]

        # evaluate classifier
        _, res = self.evaluate(data, ths)

        # choose thresholds at EER
        for subject, label in six.iteritems(self._subject2label):
            EER_auth = res['subject'][subject]['authentication']['rates']['EER']
            self.set_auth_thr(label, EER_auth[self.EER_IDX, 0], ready=True)

            EER_id = res['subject'][subject]['identification']['rates']['EER']
            self.set_id_thr(label, EER_id[self.EER_IDX, 0], ready=True)

    def set_auth_thr(self, subject, threshold, ready=False):
        """Set the authentication threshold of a subject.

        Parameters
        ----------
        subject : hashable
            Subject identity.
        threshold : int, float
            Threshold value.
        ready : bool, optional
            If True, `subject` is the internal classifier label.

        """

        if not ready:
            if not self.check_subject(subject):
                raise SubjectError(subject)
            subject = self._subject2label[subject]

        try:
            self._thresholds[subject]['auth'] = threshold
        except KeyError:
            self._thresholds[subject] = {'auth': threshold, 'id': None}

    def get_auth_thr(self, subject, ready=False):
        """Get the authentication threshold of a subject.

        Parameters
        ----------
        subject : hashable
            Subject identity.
        ready : bool, optional
            If True, `subject` is the internal classifier label.

        Returns
        -------
        threshold : int, float
            Threshold value.

        """

        if not ready:
            if not self.check_subject(subject):
                raise SubjectError(subject)
            subject = self._subject2label[subject]

        return self._thresholds[subject].get('auth', None)

    def set_id_thr(self, subject, threshold, ready=False):
        """Set the identification threshold of a subject.

        Parameters
        ----------
        subject : hashable
            Subject identity.
        threshold : int, float
            Threshold value.
        ready : bool, optional
            If True, `subject` is the internal classifier label.

        """

        if not ready:
            if not self.check_subject(subject):
                raise SubjectError(subject)
            subject = self._subject2label[subject]

        try:
            self._thresholds[subject]['id'] = threshold
        except KeyError:
            self._thresholds[subject] = {'auth': None, 'id': threshold}

    def get_id_thr(self, subject, ready=False):
        """Get the identification threshold of a subject.

        Parameters
        ----------
        subject : hashable
            Subject identity.
        ready : bool, optional
            If True, `subject` is the internal classifier label.

        Returns
        -------
        threshold : int, float
            Threshold value.

        """

        if not ready:
            if not self.check_subject(subject):
                raise SubjectError(subject)
            subject = self._subject2label[subject]

        return self._thresholds[subject].get('id', None)

    def get_thresholds(self, force=False):
        """Get an array of reasonable thresholds.

        Parameters
        ----------
        force : bool, optional
            If True, forces generation of thresholds.

        Returns
        -------
        ths : array
            Generated thresholds.

        """

        if force or (self._autoThresholds is None):
            self._autoThresholds = self._get_thresholds()

        return self._autoThresholds

    def authenticate(self, data, subject, threshold=None):
        """Authenticate a set of feature vectors, allegedly belonging to the
        given subject.

        Parameters
        ----------
        data : array
            Input test data.
        subject : hashable
            Subject identity.
        threshold : int, float, optional
            Authentication threshold.

        Returns
        -------
        decision : array
            Authentication decision for each input sample.

        """

        # check train state
        if not self.is_trained:
            raise UntrainedError

        # check subject
        if not self.check_subject(subject):
            raise SubjectError(subject)

        label = self._subject2label[subject]

        # check threshold
        if threshold is None:
            threshold = self.get_auth_thr(label, ready=True)

        # prepare data
        aux = self._prepare(data, targets=label)

        # authenticate
        decision = self._authenticate(aux, label, threshold)

        return decision

    def identify(self, data, threshold=None):
        """Identify a set of feature vectors.

        Parameters
        ----------
        data : array
            Input test data.
        threshold : int, float, optional
            Identification threshold.

        Returns
        -------
        subjects : list
            Identity of each input sample.

        """

        # check train state
        if not self.is_trained:
            raise UntrainedError

        # prepare data
        aux = self._prepare(data)

        # identify
        labels = self._identify(aux, threshold)

        # translate class labels
        subjects = [self._subject2label.inv.get(item, '') for item in labels]

        return subjects

    def evaluate(self, data, thresholds=None, show=False):
        """Assess the performance of the classifier in both authentication and
        identification scenarios.

        Parameters
        ----------
        data : dict
            Dictionary holding test data for each subject.
        thresholds : array, optional
            Classifier thresholds to use.
        show : bool, optional
            If True, show a summary plot.

        Returns
        -------
        classification : dict
            Classification results.
        assessment : dict
            Biometric statistics.

        """

        # check train state
        if not self.is_trained:
            raise UntrainedError

        # check thresholds
        if thresholds is None:
            thresholds = self.get_thresholds()

        # get subjects
        subjects = [item for item in data if self.check_subject(item)]
        if len(subjects) == 0:
            raise ValueError("No enrolled subjects in test set.")

        results = {
            'subjectList': subjects,
            'subjectDict': self._subject2label,
        }

        for subject in subjects:
            # prepare data
            aux = self._prepare(data[subject])

            # test
            auth_res = []
            id_res = []
            for th in thresholds:
                # authentication
                auth = []
                for subject_tst in subjects:
                    label = self._subject2label[subject_tst]
                    auth.append(self._authenticate(aux, label, th))

                auth_res.append(np.array(auth))

                # identification
                id_res.append(self._identify(aux, th))

            auth_res = np.array(auth_res)
            id_res = np.array(id_res)
            results[subject] = {'authentication': auth_res,
                                'identification': id_res,
                                }

        # assess classification results
        assess, = assess_classification(results, thresholds)

        # output
        args = (results, assess)
        names = ('classification', 'assessment')
        out = utils.ReturnTuple(args, names)

        if show:
            # plot
            plotting.plot_biometrics(assess, self.EER_IDX, show=True)

        return out

    @classmethod
    def cross_validation(cls, data, labels, cv, thresholds=None, **kwargs):
        """Perform Cross Validation (CV) on a data set.

        Parameters
        ----------
        data : array
            An m by n array of m data samples in an n-dimensional space.
        labels : list, array
            A list of m class labels.
        cv : CV iterator
            A `sklearn.model_selection` iterator.
        thresholds : array, optional
            Classifier thresholds to use.
        ``**kwargs`` : dict, optional
            Classifier parameters.

        Returns
        -------
        runs : list
            Evaluation results for each CV run.
        assessment : dict
            Final CV biometric statistics.

        """

        runs = []
        aux = []
        for train, test in cv:
            # train data set
            train_idx = collections.defaultdict(list)
            for item in train:
                lbl = labels[item]
                train_idx[lbl].append(item)

            train_data = {sub: data[idx] for sub, idx in six.iteritems(train_idx)}

            # test data set
            test_idx = collections.defaultdict(list)
            for item in test:
                lbl = labels[item]
                test_idx[lbl].append(item)

            test_data = {sub: data[idx] for sub, idx in six.iteritems(test_idx)}

            # instantiate classifier
            clf = cls(**kwargs)
            clf.batch_train(train_data)
            res = clf.evaluate(test_data, thresholds=thresholds)
            del clf

            aux.append(res['assessment'])
            runs.append(res)

        # assess runs
        if len(runs) > 0:
            subjects = runs[0]['classification']['subjectList']
            assess, = assess_runs(results=aux, subjects=subjects)
        else:
            raise ValueError("CV iterator empty or exhausted.")

        # output
        args = (runs, assess)
        names = ('runs', 'assessment')

        return utils.ReturnTuple(args, names)

    def _authenticate(self, data, label, threshold):
        """Authenticate a set of feature vectors, allegedly belonging to the
        given subject.

        Parameters
        ----------
        data : array
            Input test data.
        label : str
            Internal classifier subject label.
        threshold : int, float
            Authentication threshold.

        Returns
        -------
        decision : array
            Authentication decision for each input sample.

        """

        decision = np.zeros(len(data), dtype='bool')

        return decision

    def _get_thresholds(self):
        """Generate an array of reasonable thresholds.

        Returns
        -------
        ths : array
            Generated thresholds.

        """

        ths = np.array([])

        return ths

    def _identify(self, data, threshold=None):
        """Identify a set of feature vectors.

        Parameters
        ----------
        data : array
            Input test data.
        threshold : int, float
            Identification threshold.

        Returns
        -------
        labels : list
            Identity (internal label) of each input sample.

        """

        labels = [''] * len(data)

        return labels

    def _prepare(self, data, targets=None):
        """Prepare data to be processed.

        Parameters
        ----------
        data : array
            Data to process.
        targets : list, str, optional
            Target subject labels.

        Returns
        -------
        out : object
            Processed data.

        """

        # target class labels
        if targets is None:
            targets = list(self._subject2label.values())
        elif isinstance(targets, six.string_types):
            targets = [targets]

        return data

    def _train(self, enroll=None, dismiss=None):
        """Train the classifier.

        Parameters
        ----------
        enroll : list, optional
            Labels of new or updated subjects.
        dismiss : list, optional
            Labels of deleted subjects.

        """

        if enroll is None:
            enroll = []
        if dismiss is None:
            dismiss = []

        # process dismiss
        for _ in dismiss:
            pass

        # process enroll
        for _ in enroll:
            pass

    def _update(self, old, new):
        """Combine new data with existing templates (for one subject).

        Parameters
        ----------
        old : array
            Existing data.
        new : array
            New data.

        Returns
        -------
        out : array
            Combined data.

        """

        return new


class KNN(BaseClassifier):
    """K Nearest Neighbors (k-NN) biometric classifier.

    Parameters
    ----------
    k : int, optional
        Number of neighbors.
    metric : str, optional
        Distance metric.
    metric_args : dict, optional
        Additional keyword arguments are passed to the distance function.

    Attributes
    ----------
    EER_IDX : int
        Reference index for the Equal Error Rate.

    """

    EER_IDX = 0

    def __init__(self, k=3, metric='euclidean', metric_args=None):
        # parent __init__
        super(KNN, self).__init__()

        # algorithm self things
        self.k = k
        self.metric = metric
        if metric_args is None:
            metric_args = {}
        self.metric_args = metric_args

        # test metric args
        _ = metrics.pdist(np.zeros((2, 2)), metric, **metric_args)

        # minimum threshold
        self.min_thr = 10 * np.finfo('float').eps

    def _sort(self, dists, train_labels):
        """Sort the computed distances.

        Parameters
        ----------
        dists : array
            Unsorted computed distances.
        train_labels : list
            Unsorted target subject labels.

        Returns
        -------
        dists : array
            Sorted computed distances.
        train_labels : list
            Sorted target subject labels.

        """

        ind = dists.argsort()
        # sneaky trick from http://stackoverflow.com/questions/6155649
        static_inds = np.arange(dists.shape[0]).reshape((dists.shape[0], 1))
        dists = dists[static_inds, ind]
        train_labels = train_labels[static_inds, ind]

        return dists, train_labels

    def _authenticate(self, data, label, threshold):
        """Authenticate a set of feature vectors, allegedly belonging to the
        given subject.

        Parameters
        ----------
        data : array
            Input test data.
        label : str
            Internal classifier subject label.
        threshold : int, float
            Authentication threshold.

        Returns
        -------
        decision : array
            Authentication decision for each input sample.

        """

        # unpack prepared data
        dists = data['dists']
        train_labels = data['train_labels']

        # select based on subject label
        aux = []
        ns = len(dists)
        for i in range(ns):
            aux.append(dists[i, train_labels[i, :] == label])

        dists = np.array(aux)

        # nearest neighbors
        dists = dists[:, :self.k]

        decision = np.zeros(ns, dtype='bool')
        for i in range(ns):
            # compare distances to threshold
            count = np.sum(dists[i, :] <= threshold)

            # decide accept
            if count > (self.k // 2):
                decision[i] = True

        return decision

    def _get_thresholds(self):
        """Generate an array of reasonable thresholds.

        For metrics other than 'cosine' or 'pcosine', which have a clear
        limits, generates an array based on the maximum distances between
        enrolled subjects.

        Returns
        -------
        ths : array
            Generated thresholds.

        """

        if self.metric == 'cosine':
            return np.linspace(self.min_thr, 2., 100)
        elif self.metric == 'pcosine':
            return np.linspace(self.min_thr, 1., 100)

        maxD = []
        for _ in range(3):
            for label in list(six.itervalues(self._subject2label)):
                # randomly select samples
                aux = self.io_load(label)
                ind = np.random.randint(0, aux.shape[0], 3)
                obs = aux[ind]

                # compute distances
                dists = self._prepare(obs)['dists']
                maxD.append(np.max(dists))

        # maximum distance
        maxD = 1.5 * np.max(maxD)

        ths = np.linspace(self.min_thr, maxD, 100)

        return ths

    def _identify(self, data, threshold=None):
        """Identify a set of feature vectors.

        Parameters
        ----------
        data : array
            Input test data.
        threshold : int, float
            Identification threshold.

        Returns
        -------
        labels :list
            Identity (internal label) of each input sample.

        """

        if threshold is None:
            thrFcn = lambda label: self.get_id_thr(label, ready=True)
        else:
            thrFcn = lambda label: threshold

        # unpack prepared data
        dists = data['dists']
        train_labels = data['train_labels']

        # nearest neighbors
        dists = dists[:, :self.k]
        train_labels = train_labels[:, :self.k]
        ns = len(dists)

        labels = []
        for i in range(ns):
            lbl, _ = majority_rule(train_labels[i, :], random=True)

            # compare distances to threshold
            count = np.sum(dists[i, :] <= thrFcn(lbl))

            # decide
            if count > (self.k // 2):
                # accept
                labels.append(lbl)
            else:
                # reject
                labels.append('')

        return labels

    def _prepare(self, data, targets=None):
        """Prepare data to be processed.

        Computes the distances of the input data set to the target subjects.

        Parameters
        ----------
        data : array
            Data to process.
        targets : list, str, optional
            Target subject labels.

        Returns
        -------
        out : dict
            Processed data containing the computed distances (`dists`) and the
            target subject labels (`train_labels`).

        """

        # target class labels
        if targets is None:
            targets = list(six.itervalues(self._subject2label))
        elif isinstance(targets, six.string_types):
            targets = [targets]

        dists = []
        train_labels = []

        for label in targets:
            # compute distances
            D = metrics.cdist(data, self.io_load(label),
                              metric=self.metric, **self.metric_args)

            dists.append(D)
            train_labels.append(np.tile(label, D.shape))

        dists = np.concatenate(dists, axis=1)
        train_labels = np.concatenate(train_labels, axis=1)

        # sort
        dists, train_labels = self._sort(dists, train_labels)

        return {'dists': dists, 'train_labels': train_labels}

    def _update(self, old, new):
        """Combine new data with existing templates (for one subject).

        Simply concatenates old data with new data.

        Parameters
        ----------
        old : array
            Existing data.
        new : array
            New data.

        Returns
        -------
        out : array
            Combined data.

        """

        out = np.concatenate([old, new], axis=0)

        return out


class SVM(BaseClassifier):
    """Support Vector Machines (SVM) biometric classifier.

    Wraps the 'OneClassSVM' and 'SVC' classes from 'scikit-learn'.

    Parameters
    ----------
    C : float, optional
        Penalty parameter C of the error term.
    kernel : str, optional
        Specifies the kernel type to be used in the algorithm. It must be one
        of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
        If none is given, ‘rbf’ will be used. If a callable is given it is
        used to precompute the kernel matrix.
    degree : int, optional
        Degree of the polynomial kernel function (‘poly’). Ignored by all other
        kernels.
    gamma : float, optional
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is 'auto'
        then 1/n_features will be used instead.
    coef0 : float, optional
        Independent term in kernel function. It is only significant in ‘poly’
        and ‘sigmoid’.
    shrinking : bool, optional
        Whether to use the shrinking heuristic.
    tol : float, optional
        Tolerance for stopping criterion.
    cache_size : float, optional
        Specify the size of the kernel cache (in MB).
    max_iter : int, optional
        Hard limit on iterations within solver, or -1 for no limit.
    random_state : int, RandomState, optional
        The seed of the pseudo random number generator to use when shuffling
        the data for probability estimation.

    Attributes
    ----------
    EER_IDX : int
        Reference index for the Equal Error Rate.

    """

    EER_IDX = -1

    def __init__(self,
                 C=1.0,
                 kernel='linear',
                 degree=3,
                 gamma='auto',
                 coef0=0.0,
                 shrinking=True,
                 tol=0.001,
                 cache_size=200,
                 max_iter=-1,
                 random_state=None):
        # parent __init__
        super(SVM, self).__init__()

        # algorithm self things
        self._models = {}
        self._clf_kwargs = {
            'C': C,
            'kernel': kernel,
            'degree': degree,
            'gamma': gamma,
            'coef0': coef0,
            'shrinking': shrinking,
            'tol': tol,
            'cache_size': cache_size,
            'max_iter': max_iter,
            'random_state': random_state,
        }

        # minimum threshold
        self.min_thr = 10 * np.finfo('float').eps

    def _get_weights(self, n1, n2):
        """Compute class weights.

        The weights are inversely proportional to the number of samples in each
        class.

        Parameters
        ----------
        n1 : int
            Number of samples in the first class.
        n2 : int
            Number of samples in the second class.

        Returns
        -------
        weights : dict
            Weights for each class.

        """

        w = np.array([1. / n1, 1. / n2])
        w *= 2 / np.sum(w)
        weights = {-1: w[0], 1: w[1]}

        return weights

    def _get_single_clf(self, X, label):
        """Instantiate and train a One Class SVM classifier.

        Parameters
        ----------
        X : array
            Training data.
        label : str
            Class label.

        """

        clf = sksvm.OneClassSVM(kernel='rbf', nu=0.1)
        clf.fit(X)

        # add to models
        self._models[('', label)] = clf

    def _get_kernel_clf(self, X1, X2, n1, n2, label1, label2):
        """Instantiate and train a SVC SVM classifier.

        Parameters
        ----------
        X1 : array
            Trainig data for the first class.
        X2 : array
            Training data for the second class.
        n1 : int
            Number of samples in the first class.
        n2 : int
            Number of samples in the second class.
        label1 : str
            Label for the first class.
        label2 : str
            Label for the first class.

        """

        # prepare data to train
        X = np.concatenate((X1, X2), axis=0)
        Y = np.ones(n1 + n2)

        pair = self._convert_pair((label1, label2))
        if pair[0] == label1:
            Y[:n1] = -1
        else:
            Y[n1:] = -1

        # class weights
        weights = self._get_weights(n1, n2)

        # instantiate and fit
        clf = sksvm.SVC(class_weight=weights, **self._clf_kwargs)
        clf.fit(X, Y)

        # add to models
        self._models[pair] = clf

    def _del_clf(self, pair):
        """Delete a binary classifier.

        Parameters
        ----------
        pair : list, tuple
            Label pair.

        """

        pair = self._convert_pair(pair)
        m = self._models.pop(pair)
        del m

    def _convert_pair(self, pair):
        """Sort and convert a label pair to the internal representation format.

        Parameters
        ----------
        pair : list, tuple
            Input label pair.

        Returns
        -------
        pair : tuple
            Sorted label pair.

        """

        pair = tuple(sorted(pair))

        return pair

    def _predict(self, pair, X):
        """Get a classifier prediction of the input data, given the label pair.

        Parameters
        ----------
        pair : list, tuple
            Label pair.
        X : array
            Input data to classify.

        Returns
        -------
        prediction : array
            Prediction for each sample in the input data.

        """

        # convert pair
        pair = self._convert_pair(pair)

        # classify
        aux = self._models[pair].predict(X)

        prediction = []
        for item in aux:
            if item < 0:
                prediction.append(pair[0])
            elif item > 0:
                prediction.append(pair[1])
            else:
                prediction.append('')

        prediction = np.array(prediction)

        return prediction

    def _authenticate(self, data, label, threshold):
        """Authenticate a set of feature vectors, allegedly belonging to the
        given subject.

        Parameters
        ----------
        data : array
            Input test data.
        label : str
            Internal classifier subject label.
        threshold : int, float
            Authentication threshold.

        Returns
        -------
        decision : array
            Authentication decision for each input sample.

        """

        # unpack prepared data
        aux = data['predictions']
        ns = aux.shape[1]
        pairs = data['pairs']

        # normalization
        if self._nbSubjects > 1:
            norm = float(self._nbSubjects - 1)
        else:
            norm = 1.0

        # select pairs
        sel = np.nonzero([label in p for p in pairs])[0]
        aux = aux[sel, :]

        decision = []
        for i in range(ns):
            # determine majority
            predMax, count = majority_rule(aux[:, i], random=True)
            rate = float(count) / norm

            if predMax == '':
                decision.append(False)
            else:
                # compare with threshold
                if rate > threshold:
                    decision.append(predMax == label)
                else:
                    decision.append(False)

        decision = np.array(decision)

        return decision

    def _get_thresholds(self):
        """Generate an array of reasonable thresholds.

        The thresholds correspond to the relative number of binary classifiers
        that agree on a class.

        Returns
        -------
        ths : array
            Generated thresholds.

        """

        ths = np.linspace(self.min_thr, 1.0, 100)

        return ths

    def _identify(self, data, threshold=None):
        """Identify a set of feature vectors.

        Parameters
        ----------
        data : array
            Input test data.
        threshold : int, float
            Identification threshold.

        Returns
        -------
        labels : list
            Identity (internal label) of each input sample.

        """

        if threshold is None:
            thrFcn = lambda label: self.get_id_thr(label, ready=True)
        else:
            thrFcn = lambda label: threshold

        # unpack prepared data
        aux = data['predictions']
        ns = aux.shape[1]

        # normalization
        if self._nbSubjects > 1:
            norm = float(self._nbSubjects - 1)
        else:
            norm = 1.0

        labels = []
        for i in range(ns):
            # determine majority
            predMax, count = majority_rule(aux[:, i], random=True)
            rate = float(count) / norm

            if predMax == '':
                labels.append('')
            else:
                # compare with threshold
                if rate > thrFcn(predMax):
                    # accept
                    labels.append(predMax)
                else:
                    # reject
                    labels.append('')

        return labels

    def _prepare(self, data, targets=None):
        """Prepare data to be processed.

        Computes the predictions for each of the targeted classifier pairs.

        Parameters
        ----------
        data : array
            Data to process.
        targets : list, str, optional
            Target subject labels.

        Returns
        -------
        out : dict
            Processed data containing an array with the predictions of each
            input sample (`predictions`) and a list with the target label
            pairs (`pairs`).

        """

        # target class labels
        if self._nbSubjects == 1:
            pairs = list(self._models)
        else:
            if targets is None:
                pairs = list(self._models)
            elif isinstance(targets, six.string_types):
                labels = list(
                    set(self._subject2label.values()) - set([targets]))
                pairs = [[targets, lbl] for lbl in labels]
            else:
                pairs = []
                for t in targets:
                    labels = list(set(self._subject2label.values()) - set([t]))
                    pairs.extend([t, lbl] for lbl in labels)

        # predict
        predictions = np.array([self._predict(p, data) for p in pairs])

        out = {'predictions': predictions, 'pairs': pairs}

        return out

    def _train(self, enroll=None, dismiss=None):
        """Train the classifier.

        Parameters
        ----------
        enroll : list, optional
            Labels of new or updated subjects.
        dismiss : list, optional
            Labels of deleted subjects.

        """

        if enroll is None:
            enroll = []
        if dismiss is None:
            dismiss = []

        # process dismiss
        src_pairs = list(self._models)
        pairs = []
        for t in dismiss:
            pairs.extend([p for p in src_pairs if t in p])

        for p in pairs:
            self._del_clf(p)

        # process enroll
        existing = list(set(self._subject2label.values()) - set(enroll))
        for i, t1 in enumerate(enroll):
            X1 = self.io_load(t1)
            n1 = len(X1)

            # existing subjects
            for t2 in existing:
                X2 = self.io_load(t2)
                n2 = len(X2)
                self._get_kernel_clf(X1, X2, n1, n2, t1, t2)

            # new subjects
            for t2 in enroll[i + 1:]:
                X2 = self.io_load(t2)
                n2 = len(X2)
                self._get_kernel_clf(X1, X2, n1, n2, t1, t2)

        # check singles
        if self._nbSubjects == 1:
            label = list(six.itervalues(self._subject2label))[0]
            X = self.io_load(label)
            self._get_single_clf(X, label)
        elif self._nbSubjects > 1:
            aux = [p for p in self._models if '' in p]
            if len(aux) != 0:
                for p in aux:
                    self._del_clf(p)

    def _update(self, old, new):
        """Combine new data with existing templates (for one subject).

        Simply concatenates old data with new data.

        Parameters
        ----------
        old : array
            Existing data.
        new : array
            New data.

        Returns
        -------
        out : array
            Combined data.

        """

        out = np.concatenate([old, new], axis=0)

        return out


def get_auth_rates(TP=None, FP=None, TN=None, FN=None, thresholds=None):
    """Compute authentication rates from the confusion matrix.

    Parameters
    ----------
    TP : array
        True Positive counts for each classifier threshold.
    FP : array
        False Positive counts for each classifier threshold.
    TN : array
        True Negative counts for each classifier threshold.
    FN : array
        False Negative counts for each classifier threshold.
    thresholds : array
        Classifier thresholds.

    Returns
    -------
    Acc : array
        Accuracy at each classifier threshold.
    TAR : array
        True Accept Rate at each classifier threshold.
    FAR : array
        False Accept Rate at each classifier threshold.
    FRR : array
        False Reject Rate at each classifier threshold.
    TRR : array
        True Reject Rate at each classifier threshold.
    EER : array
        Equal Error Rate points, with format (threshold, rate).
    Err : array
        Error rate at each classifier threshold.
    PPV : array
        Positive Predictive Value at each classifier threshold.
    FDR : array
        False Discovery Rate at each classifier threshold.
    NPV : array
        Negative Predictive Value at each classifier threshold.
    FOR : array
        False Omission Rate at each classifier threshold.
    MCC : array
        Matthrews Correlation Coefficient at each classifier threshold.

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

    # helper variables
    A = TP + FP
    B = TP + FN
    C = TN + FP
    D = TN + FN
    E = A * B * C * D
    F = A + D

    # avoid divisions by zero
    A[A == 0] = 1.
    B[B == 0] = 1.
    C[C == 0] = 1.
    D[D == 0] = 1.
    E[E == 0] = 1.
    F[F == 0] = 1.

    # rates
    Acc = (TP + TN) / F # accuracy
    Err = (FP + FN) / F # error rate

    TAR = TP / B # true accept rate /true positive rate
    FRR = FN / B # false rejection rate / false negative rate

    TRR = TN / C # true rejection rate / true negative rate
    FAR = FP / C # false accept rate / false positive rate

    PPV = TP / A # positive predictive value
    FDR = FP / A # false discovery rate

    NPV = TN / D # negative predictive value
    FOR = FN / D # false omission rate

    MCC = (TP*TN - FP*FN) / np.sqrt(E) # matthews correlation coefficient 

    # determine EER
    roots, values = tools.find_intersection(thresholds, FAR, thresholds, FRR)
    EER = np.vstack((roots, values)).T

    # output
    args = (Acc, TAR, FAR, FRR, TRR, EER, Err, PPV, FDR, NPV, FOR, MCC)
    names = ('Acc', 'TAR', 'FAR', 'FRR', 'TRR', 'EER', 'Err', 'PPV', 'FDR',
             'NPV', 'FOR', 'MCC')

    return utils.ReturnTuple(args, names)


def get_id_rates(H=None, M=None, R=None, N=None, thresholds=None):
    """Compute identification rates from the confusion matrix.

    Parameters
    ----------
    H : array
        Hit counts for each classifier threshold.
    M : array
        Miss counts for each classifier threshold.
    R : array
        Reject counts for each classifier threshold.
    N : int
        Number of test samples.
    thresholds : array
        Classifier thresholds.

    Returns
    -------
    Acc : array
        Accuracy at each classifier threshold.
    Err : array
        Error rate at each classifier threshold.
    MR : array
        Miss Rate at each classifier threshold.
    RR : array
        Reject Rate at each classifier threshold.
    EID : array
        Error of Identification points, with format (threshold, rate).
    EER : array
        Equal Error Rate points, with format (threshold, rate).

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


def get_subject_results(results=None,
                        subject=None,
                        thresholds=None,
                        subjects=None,
                        subject_dict=None,
                        subject_idx=None):
    """Compute authentication and identification performance metrics for a
    given subject.

    Parameters
    ----------
    results : dict
        Classification results.
    subject : hashable
        True subject label.
    thresholds : array
        Classifier thresholds.
    subjects : list
        Target subject classes.
    subject_dict : bidict
        Subject-label conversion dictionary.
    subject_idx : list
        Subject index.

    Returns
    -------
    assessment : dict
        Authentication and identification results.

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
        raise ValueError("Authentication and identification number of \
                          thresholds do not match.")
    if auth_res.shape[0] != nth:
        raise ValueError("Number of thresholds in vector does not match \
                          biometric results.")
    if auth_res.shape[2] != id_res.shape[1]:
        raise ValueError("Authentication and identification number of tests \
                          do not match.")

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

    for i in range(nth):  # for each threshold
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
        rejects = res == ''
        nrejects = np.sum(rejects)
        misses = np.logical_not(np.logical_or(hits, rejects))
        nmisses = ns - (nhits + nrejects)
        missCounts = {
            subject_dict.inv[ms]: np.sum(res == ms)
            for ms in np.unique(res[misses])
        }

        # appends
        H[i] = nhits
        M[i] = nmisses
        R[i] = nrejects
        CM.append(missCounts)

    # compute rates
    auth_rates = get_auth_rates(TP, FP, TN, FN, thresholds).as_dict()
    id_rates = get_id_rates(H, M, R, ns, thresholds).as_dict()

    output = {
        'authentication': {
            'confusionMatrix': {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN},
            'rates': auth_rates,
        },
        'identification': {
            'confusionMatrix': {'H': H, 'M': M, 'R': R, 'CM': CM},
            'rates': id_rates,
        },
    }

    return utils.ReturnTuple((output,), ('assessment',))


def assess_classification(results=None, thresholds=None):
    """Assess the performance of a biometric classification test.

    Parameters
    ----------
    results : dict
        Classification results.
    thresholds : array
        Classifier thresholds.

    Returns
    -------
    assessment : dict
        Classification assessment.

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
    output = {
        'global': {
            'authentication': {
                'confusionMatrix': {'TP': 0., 'TN': 0., 'FP': 0., 'FN': 0.},
            },
            'identification': {
                'confusionMatrix': {'H': 0., 'M': 0., 'R': 0.},
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
        output['subject'][sub]['identification']['confusionMatrix']['C'] = C[:,
                                                                             k]
        output['subject'][sub]['identification']['rates']['CR'] = CR[:, k]

    # compute global rates
    aux = get_auth_rates(auth['TP'], auth['FP'], auth['TN'], auth['FN'],
                         thresholds)
    output['global']['authentication']['rates'] = aux.as_dict()

    # identification
    Ns = iden['H'] + iden['M'] + iden['R']
    aux = get_id_rates(iden['H'], iden['M'], iden['R'], Ns, thresholds)
    output['global']['identification']['rates'] = aux.as_dict()

    return utils.ReturnTuple((output,), ('assessment',))


def assess_runs(results=None, subjects=None):
    """Assess the performance of multiple biometric classification runs.

    Parameters
    ----------
    results : list
        Classification assessment for each run.
    subjects : list
        Common target subject classes.

    Returns
    -------
    assessment : dict
        Global classification assessment.

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
        return utils.ReturnTuple((results[0],), ('assessment',))

    # output
    output = {
        'global': {
            'authentication': {
                'confusionMatrix': {'TP': 0., 'TN': 0., 'FP': 0., 'FN': 0.},
            },
            'identification': {
                'confusionMatrix': {'H': 0., 'M': 0., 'R': 0.},
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
        output['subject'][sub] = {
            'authentication': {
                'confusionMatrix': {'TP': 0., 'TN': 0., 'FP': 0., 'FN': 0.},
                'rates': {},
            },
            'identification': {
                'confusionMatrix': {'H': 0., 'M': 0., 'R': 0., 'C': 0.},
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
        aux = get_auth_rates(authS['TP'], authS['FP'], authS['TN'],
                             authS['FN'], thresholds)
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
    aux = get_auth_rates(auth['TP'], auth['FP'], auth['TN'], auth['FN'],
                         thresholds)
    output['global']['authentication']['rates'] = aux.as_dict()

    Ns = iden['H'] + iden['M'] + iden['R']
    aux = get_id_rates(iden['H'], iden['M'], iden['R'], Ns, thresholds)
    output['global']['identification']['rates'] = aux.as_dict()

    return utils.ReturnTuple((output,), ('assessment',))


def combination(results=None, weights=None):
    """Combine results from multiple classifiers.

    Parameters
    ----------
    results : dict
        Results for each classifier.
    weights : dict, optional
        Weight for each classifier.

    Returns
    -------
    decision : object
        Consensus decision.
    confidence : float
        Confidence estimate of the decision.
    counts : array
        Weight for each possible decision outcome.
    classes : array
        List of possible decision outcomes.

    """

    # check inputs
    if results is None:
        raise TypeError("Please specify the input classification results.")
    if weights is None:
        weights = {}

    # compile results to find all classes
    vec = list(six.itervalues(results))
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

        for n in results:
            # ensure array
            res = np.array(results[n])
            ns = float(len(res))

            # get count for each unique class
            for i in range(nb):
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


def majority_rule(labels=None, random=True):
    """Determine the most frequent class label.

    Parameters
    ----------
    labels : array, list
        List of clas labels.
    random : bool, optional
        If True, will choose randomly in case of tied classes, otherwise the
        first element is chosen.

    Returns
    -------
        decision : object
            Consensus decision.
        count : int
            Number of elements of the consensus decision.

    """

    # check inputs
    if labels is None:
        raise TypeError("Please specify the input list of class labels.")

    if len(labels) == 0:
        raise CombinationError("Empty list of class labels.")

    # count unique occurrences
    unq, counts = np.unique(labels, return_counts=True)

    # most frequent class
    predMax = counts.argmax()

    if random:
        # check for repeats
        ind = np.nonzero(counts == counts[predMax])[0]
        length = len(ind)

        if length > 1:
            predMax = ind[np.random.randint(0, length)]

    decision = unq[predMax]
    cnt = counts[predMax]

    out = utils.ReturnTuple((decision, cnt), ('decision', 'count'))

    return out


def cross_validation(labels,
                     n_iter=10,
                     test_size=0.1,
                     train_size=None,
                     random_state=None):
    """Return a Cross Validation (CV) iterator.

    Wraps the StratifiedShuffleSplit iterator from sklearn.model_selection.
    This iterator returns stratified randomized folds, which preserve the
    percentage of samples for each class.

    Parameters
    ----------
    labels : list, array
        List of class labels for each data sample.
    n_iter : int, optional
        Number of splitting iterations.
    test_size : float, int, optional
        If float, represents the proportion of the dataset to include in the
        test split; if int, represents the absolute number of test samples.
    train_size : float, int, optional
        If float, represents the proportion of the dataset to include in the
        train split; if int, represents the absolute number of train samples.
    random_state : int, RandomState, optional
        The seed of the pseudo random number generator to use when shuffling
        the data.

    Returns
    -------
    cv : CV iterator
        Cross Validation iterator.

    """

    cv = skcv.StratifiedShuffleSplit(
        n_splits=n_iter,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
    ).split(np.zeros(len(labels)), labels)

    return utils.ReturnTuple((cv,), ('cv',))
