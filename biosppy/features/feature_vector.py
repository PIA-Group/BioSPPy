import numpy as np
import pandas as pd
from . import bvp_features, ecg_features, eda_features, nonlinear_geo_features, resp_features, spectral_features, statistic_features, temporal_features
import pandas_profiling
from sklearn.model_selection import cross_val_score
import time

def get_feat(signal, sig_lab, sampling_rate=1000., windows_len=5, segment=True, save=True):
    """ Returns a feature vector describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    sig_lab : string
        Signal label.

    sampling_rate : float
        Sampling frequency.

    windows_len : int
        Windows size in seconds.

    segment : bool
        True: the data is segmented into to windows of size windows_len seconds; False: No segmentation.

    save : bool
        If True a the feature vector is stored in a .csv file.

    Returns
    -------
    df : dataframe
        Feature vetor in dataframe format, each column is a feature.

    signal : array
        Segmented data.
    """
    labels_name, segdata, feat_val = [], [], None
    sig_lab += '_'
    window_size = int(windows_len*sampling_rate)
    if segment:
        signal = [signal[i:i + window_size] for i in range(0, len(signal), window_size)]
        if len(signal) > 1:
            signal = signal[:-1]
    print("<START Feature Extraction>")
    t0 = time.time()
    for wind_idx, wind_sig in enumerate(signal):
        # Statistical Features
        _f = statistic_features.signal_stats(wind_sig, hist=True)
        labels_name = [str(sig_lab + i) for i in _f.keys()]
        row_feat = _f.values()
        # Temporal Features
        _f = temporal_features.signal_temp(wind_sig, sampling_rate)
        labels_name += [str(sig_lab + i) for i in _f.keys()]
        row_feat += _f.values()
        # Spectral Features
        _f = spectral_features.signal_spectral(wind_sig, sampling_rate, hist=True)
        labels_name += [str(sig_lab + i) for i in _f.keys()]
        row_feat += _f.values()
        # Sensor Specific Features
        if 'EDA' in sig_lab:
            if wind_sig is not []:
                # On SCR Signal
                eda_f = eda_features.eda_features(wind_sig)
                for item in eda_f.keys():
                    _f = statistic_features.signal_stats(eda_f[item], hist=False)
                    labels_name += [str(sig_lab + item + '_' + i) for i in _f.keys()]
                    row_feat += [i for i in _f.values()]

                    _f = temporal_features.signal_temp(eda_f[item], sampling_rate)
                    labels_name += [str(sig_lab + item + '_' + i) for i in _f.keys()]
                    row_feat += [i for i in _f.values()]
                # Non-liner and geometric features
                _f = nonlinear_geo_features.nonlinear_geo_features(eda_f['onsets'], wind_sig)
                labels_name += [str(sig_lab + i) for i in _f.keys()]
                row_feat += _f.values()
        elif 'ECG' in sig_lab:
            if wind_sig is not []:
                ecg_f = ecg_features.ecg_features(wind_sig)
                for item in ecg_f.keys():
                    if item == 'nn_intervals':
                        _f = statistic_features.signal_stats(ecg_f[item], hist=False)
                        labels_name += [str(sig_lab + item + '_' + i) for i in _f.keys()]
                        row_feat += [i for i in _f.values()]

                        _f = temporal_features.signal_temp(ecg_f[item], sampling_rate)
                        labels_name += [str(sig_lab + item + '_' + i) for i in _f.keys()]
                        row_feat += [i for i in _f.values()]
                    elif item == 'hr':
                        _f = statistic_features.signal_stats(ecg_f[item], hist=False)
                        labels_name += [str(sig_lab + item + '_' + i) for i in _f.keys()]
                        row_feat += [i for i in _f.values()]

                        _f = temporal_features.signal_temp(ecg_f[item], sampling_rate)
                        labels_name += [str(sig_lab + item + '_' + i) for i in _f.keys()]
                        row_feat += [i for i in _f.values()]
                    elif item == 'rpeaks':
                        _f = statistic_features.signal_stats(ecg_f[item], hist=False)
                        labels_name += [str(sig_lab + item + '_' + i) for i in _f.keys()]
                        row_feat += [i for i in _f.values()]

                        _f = temporal_features.signal_temp(ecg_f[item], sampling_rate)
                        labels_name += [str(sig_lab + item + '_' + i) for i in _f.keys()]
                        row_feat += [i for i in _f.values()]
                    else:
                        labels_name += [sig_lab + item]
                        row_feat += [ecg_f[item]]
                _f = nonlinear_geo_features.nonlinear_geo_features(ecg_f['rpeaks'], wind_sig)
                labels_name += [str(sig_lab + i) for i in _f.keys()]
                row_feat += _f.values()
        elif 'BVP' in sig_lab:
            if wind_sig is not []:
                bvp_f = bvp_features.bvp_features(wind_sig)
                for item in bvp_f.keys():
                    _f = statistic_features.signal_stats(bvp_f[item], hist=False)
                    labels_name += [str(sig_lab + item + '_' + i) for i in _f.keys()]
                    row_feat += [i for i in _f.values()]

                    _f = temporal_features.signal_temp(bvp_f[item], sampling_rate)
                    labels_name += [str(sig_lab + item + '_' + i) for i in _f.keys()]
                    row_feat += [i for i in _f.values()]
                _f = nonlinear_geo_features.nonlinear_geo_features(bvp_f['onsets'], wind_sig)
                labels_name += [str(sig_lab + i) for i in _f.keys()]
                row_feat += _f.values()
        if 'Resp' in sig_lab:
            if wind_sig is not []:
                resp_f = resp_features.resp_features(wind_sig)
                for item in resp_f.keys():
                    _f = statistic_features.signal_stats(resp_f[item], hist=False)
                    labels_name += [str(sig_lab + item + '_' + i) for i in _f.keys()]
                    row_feat += [i for i in _f.values()]

                    _f = temporal_features.signal_temp(resp_f[item], sampling_rate)
                    labels_name += [str(sig_lab + item + '_' + i) for i in _f.keys()]
                    row_feat += [i for i in _f.values()]
                _f = nonlinear_geo_features.nonlinear_geo_features(resp_f['zeros'], signal)
                labels_name += [str(sig_lab + i) for i in _f.keys()]
                row_feat += _f.values()
        if not wind_idx:
            segdata = wind_sig
            feat_val = np.nan_to_num(np.array(row_feat)).reshape(1, -1)
        else:
            segdata = np.vstack((segdata, wind_sig))
            feat_val = np.vstack((feat_val, np.nan_to_num(np.array(row_feat)).reshape(1, -1)))
    print("<END Feature Extraction>")
    print('Time: ', time.time()-t0, ' seconds')
    d = {str(lab): feat_val[:, idx] for idx, lab in enumerate(labels_name)}
    df = pd.DataFrame(data=d, columns=labels_name)
    df = df.replace([np.inf, -np.inf, np.nan], 0.0)
    if save:
        df.to_csv('Features.csv', sep=',', encoding='utf-8', index_label="Sample")
    return df, segdata


def remove_correlatedFeatures(df, threshold=0.85):
    """ Removes highly correlated features.
    Parameters
    ----------
    df : dataframe
        Feature vector.
    threshold : float
        Threshold for correlation.

    Returns
    -------
    df : dataframe
        Feature dataframe without high correlated features.

    """
    df = df.replace([np.inf, -np.inf, np.nan], 0.0)
    profile = pandas_profiling.ProfileReport(df)
    reject = profile.get_rejected_variables(threshold=threshold)
    for rej in reject:
        print('Removing ' + str(rej))
        df = df.drop(rej, axis=1)

    return df


def FSE(X_train, y_train, features_descrition, classifier, CV=10):
    """ Performs a sequential forward feature selection.
    Parameters
    ----------
    X_train : array
        Training set feature-vector.

    y_train : array
        Training set class-labels groundtruth.

    features_descrition : array
        Features labels.

    classifier : object
        Classifier.

    Returns
    -------
    FS_idx : array
        Selected set of best features indexes.

    FS_lab : array
        Label of the selected best set of features.

    FS_X_train : array
        Transformed feature-vector with the best feature set.

    References
    ----------
    TSFEL library: https://github.com/fraunhoferportugal/tsfel
    """
    total_acc, FS_lab, acc_list, FS_idx = [], [], [], []
    X_train = np.array(X_train)

    print("*** Feature selection started ***")
    for feat_idx, feat_name in enumerate(features_descrition):
        acc_list.append(np.mean(cross_val_score(classifier, X_train[:, feat_idx].reshape(-1,1), y_train, cv=CV)))

    curr_acc_idx = np.argmax(acc_list)
    FS_lab.append(features_descrition[curr_acc_idx])
    last_acc = acc_list[curr_acc_idx]
    FS_X_train = X_train[:, curr_acc_idx]
    total_acc.append(last_acc)
    FS_idx.append(curr_acc_idx)
    while 1:
        acc_list = []
        for feat_idx, feat_name in enumerate(features_descrition):
            if feat_name not in FS_lab:
                curr_train = np.column_stack((FS_X_train, X_train[:, feat_idx]))
                acc_list.append(np.mean(cross_val_score(classifier, curr_train, y_train, cv=CV)*100))
            else:
                acc_list.append(0)
        curr_acc_idx = np.argmax(acc_list)
        if last_acc < acc_list[curr_acc_idx]:
            FS_lab.append(features_descrition[curr_acc_idx])
            last_acc = acc_list[curr_acc_idx]
            total_acc.append(last_acc)
            FS_idx.append(curr_acc_idx)
            FS_X_train = np.column_stack((FS_X_train, X_train[:, curr_acc_idx]))
        else:
            print("FINAL Features: " + str(FS_lab))
            print("Number of selected features", len(FS_lab))
            print("Features idx: ", FS_idx)
            print("Acc: ", str(total_acc))
            print("From ", str(X_train.shape[1]), " features to ", str(len(FS_lab)))
            break
    print("*** Feature selection finished ***")

    return np.array(FS_idx), np.array(FS_lab), FS_X_train
