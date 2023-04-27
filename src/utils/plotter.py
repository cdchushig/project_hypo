import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
# import altair as alt
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import utils.consts as consts
from scipy import stats as st
import collections
import pprint


def plot_confusion_matrix(y_true, y_pred, ax, class_names, vmax=None,
                          normed=True, title='Confusion matrix'):
    matrix = confusion_matrix(y_true,y_pred)
    if normed:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    sns.heatmap(matrix, vmax=vmax, annot=True, square=True, ax=ax,
                cmap=plt.cm.Blues_r, cbar=False, linecolor='black',
                linewidths=1, xticklabels=class_names)

    ax.set_title(title, y=1.20, fontsize=12)
    ax.set_ylabel('True labels', fontsize=12)
    ax.set_xlabel('Predicted labels', y=1.10, fontsize=12)
    ax.set_yticklabels(class_names, rotation=0)

    plt.show()


def plot_wordcloud(df_medications, col_name, type_patient, flag_save_figure=True):
    all_words = ' '.join([text for text in df_medications[col_name]])
    wordcloud = WordCloud(width=800, height=500, random_state=21,
                          max_font_size=110, background_color='white').generate(all_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'wordcloud_{}_{}.pdf'.format(col_name, type_patient))))
    else:
        plt.show()


def plot_stacked_barplot(df_data):

    bars = alt.Chart().mark_bar().encode(
        x=alt.X('sum(yield)', stack='zero'),
        y=alt.Y('variety'),
        color=alt.Color('site')
    )

    text = alt.Chart().mark_text(dx=-15, dy=3).encode(
        y=alt.Y('variety'),
        x=alt.X('sum(yield)', stack='zero'),
        color=alt.Color('site', legend=None, scale=alt.Scale(range=['white'])),
        text=alt.Text('sum(yield)', format='.1f')
    )

    alt.layer(bars, text, data=df_data).resolve_scale(color='independent')


def plot_survey_profiles(df_data, dict_mapping_responses, title_figure, save_figure=False):

    sns.set_style("whitegrid")

    inv_map_responses = {v: k for k, v in dict_mapping_responses.items()}
    list_vars = df_data.iloc[:, :-1].columns.values
    cluster_labels = np.unique(df_data['cluster_label'].values)

    for marker, cluster_label in zip('xosd', cluster_labels):
        df_cluster = df_data[df_data['cluster_label'] == cluster_label]
        df_cluster = df_cluster.drop('cluster_label', axis=1)
        v_profile_cluster = st.mode(df_cluster.values)[0][0]
        plt.plot(list_vars, v_profile_cluster, marker=marker,
                ms=8, markeredgewidth=1, linestyle='dashed',
                label='Cluster {}'.format(int(cluster_label)))

    plt.xticks(rotation=90)
    plt.xlabel('Response')
    plt.ylabel('Mode')
    plt.title("Comparing profiles of clusters")
    plt.grid(alpha=0.5, linestyle='--')
    plt.legend(loc='best')

    if save_figure:
        plt.tight_layout()
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_REPORTS_EFA, 'survey_profiles_{}.png'.format(title_figure))))
        plt.close()
    else:
        plt.show()


def plot_survey_responses(df_data, dict_mapping_responses, title_figure, save_figure):
    df_data = df_data.astype('category')
    inv_map_responses = {v: k for k, v in dict_mapping_responses.items()}
    category_names = list(inv_map_responses.values())
    list_responses = list(inv_map_responses.keys())
    col_names = df_data.columns.values

    results = {}

    for var_name in col_names:
        df_count = df_data[var_name].value_counts(normalize=True)
        df_count.index = df_count.index.astype('int')
        list_available_responses = list(df_count.sort_index().index)
        list_remaining_responses = list(set(list_responses).difference(list_available_responses))
        dict_zeros = dict.fromkeys(list_remaining_responses, 0)
        dict_map = df_count.to_dict()
        dict_all = {**dict_map, **dict_zeros}
        print(dict_all)
        od = collections.OrderedDict(sorted(dict_all.items()))
        v_percentage_responses = list(map(lambda x: x * 100, list(od.values())))
        results[var_name] = v_percentage_responses

    pprint.pprint(results)

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if c != 0:
                ax.text(x, y, str(int(c)), ha='center', va='center', color=text_color)

    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    if save_figure:
        plt.tight_layout()
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_REPORTS_EFA, 'survey_responses_{}.png'.format(title_figure))))
        plt.close()
    else:
        plt.show()




