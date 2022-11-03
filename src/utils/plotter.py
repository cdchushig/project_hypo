import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix
from wordcloud import WordCloud
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from utils import consts as consts


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
