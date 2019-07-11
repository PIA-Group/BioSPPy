# Import libraries
import numpy as np
import numba
from sklearn.mixture import GaussianMixture
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from biosppy import utils
from biosppy import metrics
from biosppy import clustering
from biosppy import biometrics


@numba.jit
def findNumberOfClusters(X_train, kmin, kmax):
    """ Returns optimal number of clusters through the minimum Bayes Information criterion (BIC)  .

        Parameters
        ----------
        X_train : array
            Training set.
        kmin : int
            Minimum number of clusters.
        kmax : int
            Maximum number of clusters.

        Returns
        -------
        bestk : array
            Number of clusters.

        """
    bic = []
    for idx, _i in enumerate(range(kmin, kmax, 1)):
        model = GaussianMixture(n_components=_i, reg_covar=1e0, covariance_type='full', tol=1e-1).fit(X_train)
        bic.append(model.bic(X_train))
    bestk = list(range(kmin, kmax, 1))[int(np.argmin(bic))]
    print('Bestk: ', bestk)
    return bestk


def get_class_targets_idx(y_train):
    """ Indexes location for the different class labels.

        Parameters
        ----------
        y_train : array
            Training set class labels.

        Returns
        -------
        class_labels_idx : array
            Indexes location for the diffrent class labels.

    """
    class_labels_idx = []
    for i in range(len(np.unique(y_train))):
        _labels_idx = []
        for idx, lab_sample in enumerate(y_train):
            if lab_sample == (i+1):
                _labels_idx += [idx]
        class_labels_idx.append(_labels_idx)

    args = (class_labels_idx, )
    names = ('class_labels_idx', )

    return utils.ReturnTuple(args, names)


def dissimilarity_based(X_train, y_train, X_test, y_test, classifier, train_signal=None, test_signal=None, clustering_space='Feature', testing_space='Feature', method='medoid', by_file=False):
    """ Returns a feature vector describing the signal.

    Parameters
    ----------
    X_train : array
        Training set feature vector.

    y_train : array
        Training set class labels.

    X_test : array
        Test set feature vector.

    y_test : array
        Test set class labels.

    classifier : object
        Classifier.

    train_signal : array
        Training set segmented signal.

    test_signal : array
        Test set segmented signal.

    clustering_space : string
        Selects the clustering representation space. Options: 'Feature', 'Signal'.

    testing_space : string
        Selects the test set representation space. Options: 'Feature', 'Signal'.

    method : string
        Clustering aggloremation method. Option: 'mean', 'medoid'.

    by_file : bool
        Prediction for the file using majority voting.

    Returns
    -------
    y_predicted : list
        Classifier class label test set predictions.

    """
    train_idx_sig = get_class_targets_idx(y_train)[0]
    test_idx_sig = get_class_targets_idx(y_test)[0]

    templates = {}
    templates[method] = []
    for s in range(len(np.unique(y_train))):  # gets template for each signal
        if clustering_space == 'Feature':
            y = X_train[train_idx_sig[s]]
        else:
            y = train_signal[train_idx_sig[s]]
        kmin = 2
        kmax = int(y.shape[0] // 2)
        print('Number of clusters in class label: kmin: ', kmin, 'kmax: ', kmax)
        if kmin <= kmax:
            _i = 1
        else:
            _i = findNumberOfClusters(y, kmin, kmax)
        model = GaussianMixture(reg_covar=1e-06, covariance_type='full', n_components=_i)
        y_predicted = model.fit_predict(y)
        _Cl = clustering._extract_clusters(y_predicted)
        if testing_space == 'Feature':
            y = X_train[train_idx_sig[s]]
        else:
            y = train_signal[train_idx_sig[s]]
            X_train = train_signal
            X_test = test_signal
        template = clustering.metrics_templates(data=y, clusters=_Cl, metric_args=method)['templates']
        for nt in range(len(_Cl)):
            templates[method] += [template[nt].reshape(1, -1)]

    # Get data into new data representation
    X_train_DBRep = metrics.matrix_eucd_dist(X_train, templates[method])
    X_test_DBRep = metrics.matrix_eucd_dist(X_test, templates[method])

    # Fit classifier on new data-dissimilarity space
    model = classifier.fit(X_train_DBRep, y_train.ravel())

    if not by_file:
        y_predicted = model.predict(X_test_DBRep)
    else:
        y_predicted = []
        for s in range(len(np.unique(y_test))):
            DB_X_test = X_test[test_idx_sig[s]]
            X_test_DBRep = metrics.matrix_eucd_dist(DB_X_test, templates[method])
            y_predicted.append(biometrics.majority_rule(labels=model.predict(X_test_DBRep), random=True)['decision'])

    args = (y_predicted,)
    names = ('y_test_prediction',)
    return utils.ReturnTuple(args, names)

