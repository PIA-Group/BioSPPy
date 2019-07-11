import numpy as np
from sklearn.model_selection import train_test_split
import biosppy.features.feature_vector as fv
import biosppy.classification.supervised_learning as sl
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import biosppy.classification.dissimilarity_based as db
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()

# Load data
signals = pd.read_csv('../examples/BITALINO.csv', sep=';')
data = pd.concat((signals.A1, signals.A2, signals.A3, signals.A4)).values

# get sampling rate
SR = int(signals['Sampling Rate'][0])
windows_len = 5

# Feature extraction
_l = np.array([1] * len(signals.A1) + [2] * len(signals.A2) + [3] * len(signals.A3) + [4] * len(signals.A4))
window_size = int(SR * windows_len)

feature_vector, _ = fv.get_feat(data, sig_lab='ECG', sampling_rate=SR, windows_len=windows_len, segment=True, save=False)


# labels = np.array([_l[i:i + window_size][0] for i in range(0, len(_l), window_size)])[:-1]
# feature_vector, data = fv.get_feat(data, sig_lab='Signal', sampling_rate=SR, windows_len=windows_len, segment=True, save=False)
#
# # Remove redundant features
# feature_vector = fv.remove_correlatedFeatures(feature_vector)
# print('\n')
#
# # Separate in train and set set
# X_train, X_test, y_train, y_test, train_data, test_data = train_test_split(feature_vector.values, labels, data, test_size=0.33, random_state=42)
#
# # Fit supervised Learning Classifiers on the train set data
# classifier = sl.supervised_classification(X_train, y_train)
# print('\n')
#
# # Feature selection
# FS_idx, FS_features_names, FS_X_train = fv.FSE(X_train, y_train, feature_vector.columns, classifier, CV=4)
#
# FS_X_test = X_test[:, FS_idx]
# FS_X_train = X_train[:, FS_idx]
#
# # Update best classifier for the best feature set
# classifier = sl.supervised_classification(FS_X_train, y_train)
# print('\n')
#
# # Example for each data representation
# y_predicted = db.dissimilarity_based(FS_X_train, y_train, FS_X_test, y_test, classifier, train_signal=train_data, test_signal=test_data, clustering_space='Feature', testing_space='Signal', method='medoid', by_file=False)[0]
#
# # Ground truth
# target_names = ['Resp', 'ECG', 'EDA', 'BVP']
# print('Accuracy (%): ', accuracy_score(y_test, y_predicted)*100)
# print(classification_report(y_test, y_predicted, target_names=target_names))
# print(classification_report(y_test, y_predicted))
# print('\n')
#
# y_predicted = db.dissimilarity_based(FS_X_train, y_train, FS_X_test, y_test, classifier, train_signal=train_data, test_signal=test_data, clustering_space='Signal', testing_space='Signal', method='medoid', by_file=False)[0]
# print('Accuracy (%): ', accuracy_score(y_test, y_predicted)*100)
# print(classification_report(y_test, y_predicted, target_names=target_names))
# print(classification_report(y_test, y_predicted))
# print('\n')
#
# # FOR THE ONE RETURNING THE BEST RESULTS
#
# # Majority Voting for each sensor file
# # Supposing we do not know to which column the data belongs to
# y_predicted = db.dissimilarity_based(FS_X_train, y_train, FS_X_test, y_test, classifier, method='medoid', by_file=True)[0]
# y_test_by_file = np.unique(y_test)
# print('Accuracy (%): ', accuracy_score(y_test_by_file, y_predicted)*100)
# print(classification_report(y_test_by_file, y_predicted, target_names=target_names))
#
# #
# # # Confusion matrix
# class_labels = ['Resp','ECG','EDA','BVP']
# cm = confusion_matrix(y_test_by_file, y_predicted)
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# df_cm = pd.DataFrame(cm, index=[i for i in class_labels], columns=[i for i in class_labels])
# my_dpi = 300
# plt.figure(figsize=(2560 / my_dpi, 1440 / my_dpi), dpi=my_dpi)
# plt.tight_layout()
# ax = sb.heatmap(df_cm,  cbar = False, cmap="BuGn", annot=True)
# plt.setp(ax.get_xticklabels(), rotation=45)
# plt.ylabel('True label', fontweight='bold', fontsize = 18)
# plt.xlabel('Predicted label', fontweight='bold', fontsize = 18)
# #plt.savefig('ConfMaxtrixAP.eps', dpi=300, format='eps', bbox_inches='tight')
# plt.show()
#
#
#
#
