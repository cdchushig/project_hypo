import numpy as np
import pandas as pd
from pathlib import Path
import logging
import coloredlogs
from collections import Counter
import math
import utils.consts as consts

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def load_train_test_set_by_partition(id_partition: int, disease: str, type_encoding: str) -> (pd.DataFrame,
                                                                                              pd.DataFrame,
                                                                                              pd.DataFrame,
                                                                                              pd.DataFrame):

    path_dataset = Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, disease, type_encoding)

    df_x_train = pd.read_csv(str(Path.joinpath(path_dataset, 'x_train_{}{}.csv'.format(type_encoding, id_partition))))

    df_y_train = pd.read_csv(str(Path.joinpath(path_dataset, 'y_train_{}{}.csv'.format(type_encoding, id_partition))),
                             header=None)

    df_x_test = pd.read_csv(str(Path.joinpath(path_dataset, 'x_test_{}{}.csv'.format(type_encoding, id_partition))))

    df_y_test = pd.read_csv(str(Path.joinpath(path_dataset, 'y_test_{}{}.csv'.format(type_encoding, id_partition))),
                            header=None)


def load_raw_dataset(bbdd_name: str) -> pd.DataFrame:
    df_data = None
    if bbdd_name in consts.LISTS_BBDD_CLINICAL:
        df_data = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_RAW, bbdd_name, 'bbdd_{}.csv'.format(bbdd_name))))
    else:
        ValueError('Dataset not found!')

    return df_data


def load_preprocessed_dataset(bbdd_name: str, show_info=False) -> (np.array, np.array, np.array, list, list):

    df_data = None

    if bbdd_name in consts.LISTS_BBDD_CLINICAL or bbdd_name in consts.LISTS_BBDD_GENERAL:
        df_data = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, 'bbdd_{}.csv'.format(bbdd_name))))
    else:
        ValueError('Dataset not found!')

    y_label = df_data['label'].values
    df_features = df_data.drop(['label'], axis=1)
    v_column_names = df_features.columns.values
    m_features = df_features.values

    list_vars_categorical, list_vars_numerical = get_categorical_numerical_names(df_features, bbdd_name)

    if show_info:
        logger.info('n_samples: {}, n_features: {}'.format(df_features.shape[0], df_features.shape[1]))
        logger.info('Classes: {}, # samples: {}'.format(np.unique(y_label, return_counts=True)[0],
                                                        np.unique(y_label, return_counts=True)[1]))
        logger.info('List of numerical features {}, total: {}'.format(list_vars_numerical, len(list_vars_numerical)))
        logger.info('List of categorical features {}, total: {}'.format(list_vars_categorical, len(list_vars_categorical)))

    return m_features, y_label, v_column_names, list_vars_categorical, list_vars_numerical
    # return m_features, y_label, v_column_names


def get_categorical_numerical_var_names(df_data: pd.DataFrame) -> (list, list):

    df_info = identify_type_features(df_data)
    list_numerical_vars = list(df_info[df_info['type'] == consts.TYPE_FEATURE_CONTINUOUS].index)
    list_categorical_vars = list(df_info[df_info['type'] == consts.TYPE_FEATURE_DISCRETE].index)

    return list_categorical_vars, list_numerical_vars


def identify_type_features(df, discrete_threshold=10, debug=False):
    """
    Categorize every feature/column of df as discrete or continuous according to whether or not the unique responses
    are numeric and, if they are numeric, whether or not there are fewer unique renponses than discrete_threshold.
    Return a dataframe with a row for each feature and columns for the type, the count of unique responses, and the
    count of string, number or null/nan responses.
    """
    counts = []
    string_counts = []
    float_counts = []
    null_counts = []
    types = []
    for col in df.columns:
        responses = df[col].unique()
        counts.append(len(responses))
        string_count, float_count, null_count = 0, 0, 0
        for value in responses:
            try:
                val = float(value)
                if not math.isnan(val):
                    float_count += 1
                else:
                    null_count += 1
            except:
                try:
                    val = str(value)
                    string_count += 1
                except:
                    print('Error: Unexpected value', value, 'for feature', col)

        string_counts.append(string_count)
        float_counts.append(float_count)
        null_counts.append(null_count)
        types.append('d' if len(responses) < discrete_threshold or string_count > 0 else 'c')

    feature_info = pd.DataFrame({'count': counts,
                                 'string_count': string_counts,
                                 'float_count': float_counts,
                                 'null_count': null_counts,
                                 'type': types}, index=df.columns)
    if debug:
        print(f'Counted {sum(feature_info["type"] == "d")} discrete features and {sum(feature_info["type"] == "c")} continuous features')

    return feature_info



