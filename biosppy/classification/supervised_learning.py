from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def supervised_classification(features, labels, CV=5):
    """ Supervised Learning classification.
    Parameters
    ----------
    features : array
        Feature-vector

    labels : array
        Ground truth class labels.

    CV : int
        Number of folds for the cross validation.
    Returns
    -------
    c : object
        Classifier with the best performance in terms of accuracy.
    """

    # Classifiers
    names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "ExtraTree", "AdaBoost", "GradientBoosting", "Gaussian NB",
             "Multinomial NB", "Complement NB", "Bernoulli NB", "Linear Discriminant Analysis","Quadratic Discriminant Analysis", "MLPClassifier", "SVM", "Linear SVM", "Gaussian Process", "MLP",
             "SGD", "LogisticRegression"]
    classifiers = [
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        ExtraTreesClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        MultinomialNB(),
        ComplementNB(),
        BernoulliNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        MLPClassifier(),
        svm.SVC(),
        svm.LinearSVC(),
        GaussianProcessClassifier(),
        MLPClassifier(),
        SGDClassifier(),
        LogisticRegression()
    ]

    best = 0
    best_classifier = None
    print("<START Classification>")

    for n, c in zip(names, classifiers):
        print('Classifier: ', n)
        try:
            accuracy = cross_val_score(c, features, labels, cv=CV)*100
        except:
            accuracy = 0.0
        print("Accuracy: " + str(np.mean(accuracy)) + ' +- ' + str(np.std(accuracy)) + '%')
        print('-----------------------------------------')
        if np.mean(accuracy) > best:
            best_classifier = n
            best = np.mean(accuracy)
    print("<END Classification>")

    print('Best Classifier: ' + str(best_classifier))
    print('Accuracy: ' + str(best) + '%')

    return c
