import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, roc_auc_score
from scipy.spatial.distance import euclidean


_SQRT2 = np.sqrt(2)

def compute_classification_prestations(y_true: np.array, y_pred: np.array) -> (float, float, float, float):

    prestations = classification_report(y_true, y_pred)
    matrix = pd.crosstab(y_true, y_pred, rownames=['Real'], colnames=['Predicted'], margins=True)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc_val = accuracy_score(y_true, y_pred)
    specificity_val = tn / (tn + fp)
    recall_val = recall_score(y_true, y_pred)
    roc_val = roc_auc_score(y_true, y_pred)

    print('acc: ', acc_val)
    print('specificity: ', specificity_val)
    print('recall: ', recall_val)
    print('roc: ', roc_val)

    return acc_val, specificity_val, recall_val, roc_val


def compute_hellinger_distance(df_real, df_syn, v_column_names, list_categorical_variables):
    list_hellinger = []

    for col_name in v_column_names:
        if col_name in list_categorical_variables:
            pmf_real, pmf_syn = compute_pmf(df_syn, df_real, col_name)
            hellinger_value = euclidean(np.sqrt(pmf_real), np.sqrt(pmf_syn)) / _SQRT2
        else:
            print('holi')

        list_hellinger.append(hellinger_value)

    v_hellinger = np.array(list_hellinger)
    avg_helliger = np.sum(v_hellinger) / v_hellinger.shape[0]

    return avg_helliger


def compute_pmf(df_synthetic, df_real, feature_name, flag_normalize=True):

    v_real_categories = list(np.unique(df_real.loc[:, feature_name]))
    df_probs_real = build_dataframe_probs(df_real, v_real_categories, feature_name, description='real',
                                          flag_normalize=flag_normalize)

    df_probs_synthetic = build_dataframe_probs(df_synthetic, v_real_categories, feature_name, description='syn',
                                               flag_normalize=flag_normalize)

    pmf_real = df_probs_real['prob'].values
    pmf_syn = df_probs_synthetic['prob'].values

    return pmf_real, pmf_syn


def build_dataframe_probs(df_target, list_real_categories, var_name, description, flag_normalize=False):
    x_target = df_target.loc[:, var_name].astype('category').values
    n_samples = x_target.shape[0]
    counter_samples = Counter(x_target)
    dict_counter_samples = dict(counter_samples)

    list_unique_categories = list(map(itemgetter(0), sorted(counter_samples.items(), key=itemgetter(0))))
    list_diff = list(set(list_real_categories) - set(list_unique_categories))
    dict_diff_categories = dict.fromkeys(list_diff, 0)

    dict_total = {**dict_counter_samples, **dict_diff_categories}
    dict_total_ordered = dict(sorted(dict_total.items()))

    x_unique = np.array(list(dict_total_ordered.keys()))
    counts_categories = np.array(list(dict_total_ordered.values()))

    if flag_normalize:
        probs_target = np.array(counts_categories) / n_samples
    else:
        probs_target = np.array(counts_categories)

    x_probs = np.c_[x_unique, probs_target]
    df_probs = pd.DataFrame(x_probs, columns=['x', 'prob'])

    return df_probs
