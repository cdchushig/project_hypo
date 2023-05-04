import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import utils.consts as consts
from utils.noise_generator import NoiseGenerator
from utils.loader import get_categorical_numerical_var_names, identify_type_features
import itertools
from scipy.stats import pointbiserialr, spearmanr
from phik import phik_from_array
import scipy
import scipy.cluster.hierarchy as sch


def cluster_corr(corr_array, inplace=False):
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


def plot_hists_real_noisy_var(df):

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), sharey=True)
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("Real vs Noisy", fontsize=14, y=0.95)

    tickers = df.columns.values

    for ticker, ax in zip(tickers, axs.ravel()):
        df_counts = df.loc[:, ticker].value_counts().reset_index()
        sns.barplot(x='index', y=ticker, data=df_counts, ax=ax)
        ax.bar_label(ax.containers[0])
        ax.set_title(ticker)
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.show()


def plot_corr_mixed_dataset(df_corr, figsize=(8, 8)):

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_corr,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1.0, vmax=1.0,
                annot=True, fmt=".2f",
                square=True, ax=ax)
    plt.show()


def compute_corr_mixed_dataset(df_data, target_col_id):

    df_features = df_data.drop(columns=[target_col_id])

    list_discrete_vars, list_cont_vars = get_categorical_numerical_var_names(df_features)
    df_features_info = identify_type_features(df_features)

    print('d', list_discrete_vars)
    print('c', list_cont_vars)
    print(df_features_info)

    n_samples, n_features = df_features.shape[0], df_features.shape[1]

    # pairs = list(itertools.combinations(range(n_features), 2))
    pairs = list(itertools.combinations_with_replacement(range(n_features), 2))

    m_corr = np.zeros((n_features, n_features))

    for i, j in pairs:
        x_var_name = df_features_info.index[i]
        y_var_name = df_features_info.index[j]

        x_var = df_features.loc[:, x_var_name]
        y_var = df_features.loc[:, y_var_name]

        if df_features_info['type'][i] == 'c':
            if df_features_info['type'][j] == 'c':
                pair_corr = spearmanr(x_var, y_var).statistic
            else:
                pair_corr = pointbiserialr(x_var, y_var).statistic
        else:  # categorical
            if df_features_info['type'][j] == 'c':
                pair_corr = pointbiserialr(x_var, y_var).statistic
            else:
                pair_corr = phik_from_array(x_var, y_var)

        print(x_var_name, y_var_name, pair_corr)

        m_corr[i, j] = pair_corr
        m_corr[j, i] = pair_corr

    v_var_names = df_features.columns.values
    df_corr = pd.DataFrame(m_corr, columns=v_var_names)
    df_corr = df_corr.set_index(v_var_names)

    return df_corr


seed_value = 3232

df_data = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA, 'others', 'bbdd_steno.csv')), sep=',')

v_sex = df_data.loc[:, 'sex'].values.reshape(-1, 1)
v_exercise = df_data.loc[:, 'exercise'].values.reshape(-1, 1)

v_sex_noisy1 = NoiseGenerator().corrupt_input('masking', v_sex, fraction_noise=10, seed_value=seed_value)
v_sex_noisy2 = NoiseGenerator().corrupt_input('masking', v_sex, fraction_noise=20, seed_value=seed_value)
v_sex_noisy3 = NoiseGenerator().corrupt_input('masking', v_sex, fraction_noise=30, seed_value=seed_value)

v_exercise_noisy1 = NoiseGenerator().corrupt_input('salt_and_pepper', v_exercise, fraction_noise=10, seed_value=seed_value)
v_exercise_noisy2 = NoiseGenerator().corrupt_input('salt_and_pepper', v_exercise, fraction_noise=20, seed_value=seed_value)

m_real_noisy = np.hstack((v_sex, v_sex_noisy1, v_sex_noisy2, v_sex_noisy3))
df_real_noisy = pd.DataFrame(m_real_noisy)
# print(df_real_noisy)
# plot_hists_real_noisy_var(df_real_noisy)

df_data['sexn1'] = v_sex_noisy1
df_data['sexn2'] = v_sex_noisy2
df_data['sexn3'] = v_sex_noisy3

df_data['exercisen1'] = v_exercise_noisy1
df_data['exercisen2'] = v_exercise_noisy2

df_corr = compute_corr_mixed_dataset(df_data, 'label')
plot_corr_mixed_dataset(df_corr, figsize=(10, 10))

df_corr_sorted = cluster_corr(df_corr)
plot_corr_mixed_dataset(df_corr_sorted, figsize=(10, 10))


