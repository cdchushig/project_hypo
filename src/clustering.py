from ExKMC.Tree import Tree
from sklearn.datasets import make_blobs
from pathlib import Path
import numpy as np
import pandas as pd
from utils import consts as consts
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from validclust import ValidClust
from sklearn.decomposition import PCA
# from utils import calc_cost, plot_kmeans, plot_tree_boundary

from sklearn.metrics import silhouette_score, davies_bouldin_score
import gower
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
import seaborn as sns


def plot_tree_boundary(cluster_tree, k, x_data, kmeans, plot_mistakes=False):
    cmap = plt.cm.get_cmap('PuBuGn')

    plt.figure(figsize=(4, 4))

    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .1),
                         np.arange(y_min, y_max, .1))

    values = np.c_[xx.ravel(), yy.ravel()]

    y_cluster_tree = cluster_tree.predict(x_data)

    Z = cluster_tree.predict(values)
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=cmap,
               aspect='auto', origin='lower', alpha=0.4)

    plt.scatter([x[0] for x in x_data], [x[1] for x in x_data], c=y_cluster_tree, edgecolors='black', s=20, cmap=cmap)
    for c in range(k):
        center = x_data[y_cluster_tree == c].mean(axis=0)
        plt.scatter([center[0]], [center[1]], c="white", marker='$%s$' % c, s=350, linewidths=.5, zorder=10,
                    edgecolors='black')

    if plot_mistakes:
        y = kmeans.predict(x_data)
        mistakes = x_data[y_cluster_tree != y]
        plt.scatter([x[0] for x in mistakes], [x[1] for x in mistakes], marker='x', c='red', s=60, edgecolors='black',
                    cmap=cmap)

    plt.xticks([])
    plt.yticks([])
    plt.title("Approximation Ratio: %.2f" % (cluster_tree.score(x_data) / -kmeans.score(x_data)), fontsize=14)
    plt.show()


def plot_profile_clusters(dfagg, v_column_names, flag_save_figure=False):

    plt.figure(figsize=(20, 60))

    for index, var_name in enumerate(v_column_names):
        plt.subplot(5, 3, index + 1)
        sns.histplot(data=dfagg, x=var_name, hue="cluster_label", stat='probability', palette='tab10',
                     common_norm=False, multiple="dodge", shrink=.8)

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    if flag_save_figure:
        print('holi')
    else:
        plt.show()


def compute_cvis(x_features, v_pos_categorical_variables, method='kmeans', k=3):

    print('****', v_pos_categorical_variables)

    # range_n_clusters = [2, 3, 4, 5, 6, 7]
    range_n_clusters = [2]
    print('**Method clustering %s*' % method)

    for n_clusters in range_n_clusters:

        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters)
            v_cluster_labels = kmeans.fit_predict(x_features)
        elif method == 'kprototypes':
            # kprototype = KPrototypes(n_jobs=-1, n_clusters=n_clusters, init='Huang', random_state=0)
            kprototype = KPrototypes(n_jobs=-1, n_clusters=n_clusters, init='Cao', n_init=20, max_iter=200, gamma=0.5)
            kprototype.fit_predict(x_features, categorical=v_pos_categorical_variables)
            v_cluster_labels = kprototype.labels_

            kprototypes_opt = KPrototypes(n_jobs=-1, n_clusters=k, init='Cao', n_init=20, max_iter=200, gamma=0.5)
            kprototypes_opt.fit_predict(x_features, categorical=v_pos_categorical_variables)
            v_labels = kprototypes_opt.labels_
            v_centroids = kprototypes_opt.cluster_centroids_

        else:
            dist_m = gower.gower_matrix(x_features, cat_features=[False, True, False, False, False, False, False,
                                                                  True, True, True, True, True])
            ahc = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete', affinity='precomputed')
            v_cluster_labels = ahc.fit_predict(dist_m)

            ahc_opt = AgglomerativeClustering(n_clusters=k, linkage='average', affinity='precomputed')
            v_labels = ahc_opt.fit_predict(dist_m)

            # model_clustering_agglomerative = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)
            # v_cluster_labels = model_clustering_agglomerative.fit_predict(x_features)

        silhouette_avg = silhouette_score(x_features, v_cluster_labels)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

        db_score = davies_bouldin_score(x_features, v_cluster_labels)
        print("For n_clusters =", n_clusters, "Davies score is :", db_score)

    # save_silhouette_index_plot(path_model, num_clusters, v_cluster_labels, silhouette_avg, sample_silhouette_values)
    # return m_cluster_labels[:, 1:].copy()

    return v_labels, v_centroids


def scale_data(df_data, v_numerical_names, v_categorical_names):

    # x_features = StandardScaler().fit_transform(x_features)
    # x_features = MinMaxScaler().fit_transform(x_features)
    df_data_scaled = df_data.copy()

    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    df_data_scaled[v_numerical_names] = scaler.fit_transform(df_data[v_numerical_names])
    df_data_scaled[v_categorical_names] = df_data[v_categorical_names].astype(object)

    return df_data_scaled


k = 6

df_raw = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA, 'hugo', 'data_original.csv')), sep=',')
df_raw = df_raw.iloc[:, 1:]
print(df_raw.head())

v_eskd_risk = df_raw['eskd_risk_5y'].values
v_cvd_risk = df_raw['cvd_risk_10y'].values
df_x = df_raw.drop(columns=['eskd_risk_5y', 'cvd_risk_10y'])
v_feature_names = df_x.columns.values
v_categorical_names = ['sex', 'album_0', 'album_1', 'album_2', 'smoking', 'exercise']
v_numerical_names = ['age', 'dm_duration', 'sbp', 'ldl', 'hba1c', 'egfr']
v_labels = ['eskd_risk_5y', 'cvd_risk']

df_x_scaled = scale_data(df_x, v_numerical_names, v_categorical_names)
print(df_x.head())
print(df_x_scaled.head())

x_features = df_x_scaled.values
print(df_x_scaled.info())
v_pos_categorical_variables = [df_x_scaled.columns.get_loc(col) for col in list(df_x_scaled.select_dtypes('object').columns)]

v_cluster_labels, v_centroids = compute_cvis(x_features, v_pos_categorical_variables, 'kprototypes', k=k)
# v_cluster_labels = compute_cvis(x_features, v_pos_categorical_variables, 'ahc', k=3)
v_column_names_all = list(v_feature_names) + list(v_labels) + ['cluster_label']
df_features_with_label = pd.DataFrame(np.c_[x_features, v_eskd_risk, v_cvd_risk, v_cluster_labels],
                                      columns=v_column_names_all)

# df_features_with_label.to_csv('df_kprototypes_6clusters.csv', index=False)

# plot_profile_clusters(df_features_with_label, v_column_names_all)

# kmeans = KMeans(k)
# kmeans.fit(x_features)
# p = kmeans.predict(x_features)

# Initialize tree with up to 6 leaves, predicting 3 clusters
# tree = Tree(k=k, max_leaves=2*k)
tree = Tree(k=k)
prediction = tree.fit_predict(x_features, v_cluster_labels, v_centroids)
tree.plot('filename', feature_names=v_feature_names)

# Plot decision areas in 2D
# pca = PCA(n_components=2)
# x_features_pca = pca.fit_transform(x_features)
# tree = Tree(k)
# tree.fit(x_features, kmeans)
# plot_tree_boundary(tree, k, x_features, kmeans, plot_mistakes=True)

# Construct the tree, and return cluster labels
# prediction = tree.fit_predict(x_features, kmeans)
