import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from pprint import pprint
from time import time
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
# from utils.utils import clean_all
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import consts as consts
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
import ast
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import random

from utils.metrics import compute_classification_prestations


def plot_with_matplotlib(x_vals, y_vals, labels):

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 30)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

    plt.show()


def build_word_vector(model_w2v, tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def reduce_dimensions(model, num_dimensions=2, random_state=0):

    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)

    print(labels)

    tsne = TSNE(n_components=num_dimensions, random_state=random_state)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def get_vect(word, model):
    try:
        return model.wv[word]
    except KeyError:
        return np.zeros((model.vector_size,))


def sum_vectors(phrase, model):
    return sum(get_vect(w, model) for w in phrase)


def word2vec_features(X, model):
    feats = np.vstack([sum_vectors(p, model) for p in X])
    return feats


def get_w2v_features(w2v_model, sentence_group):
    """
    Transform a sentence_group (containing multiple lists of words)
    into a feature vector. It averages out all the word vectors of the sentence_group.
    """

    print(sentence_group)

    words = np.array(sentence_group)
    # index2word_set = set(w2v_model.wv.vocab.keys())
    index2word_set = set(w2v_model.wv.index_to_key)

    featureVec = np.zeros(w2v_model.vector_size, dtype="float32")

    # Initialize a counter for number of words in a review
    nwords = 0
    # Loop over each word in the comment and, if it is in the model's vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            featureVec = np.add(featureVec, w2v_model[word])
            nwords += 1.

    # Divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)

    return featureVec


def transform_list(x):
    return ast.literal_eval(x)


def search_best_hyperparams(x_train, y_train, x_test, y_test):

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", SGDClassifier(max_iter=5000)),
        ]
    )

    parameters = {
        "vect__max_df": (0.7, 0.8, 0.9, 1.0),
        "vect__min_df": (0.01, 0.02, 0.05, 0.1),
        "vect__max_features": (20, 40, 60, 80, 100),
        "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        "clf__max_iter": (20,),
        "clf__alpha": (0.00001, 0.000001),
        "clf__penalty": ("l2", "elasticnet"),
        # 'clf__max_iter': (10, 50, 80),
    }

    # Find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(x_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    clf_model = grid_search.best_estimator_
    y_pred = clf_model.predict(x_test)

    compute_classification_prestations(y_test, y_pred)


def parse_arguments(parser):
    parser.add_argument('--type_encoding', default='tfidf', type=str)
    parser.add_argument('--dataset', default='medcon', type=str)
    parser.add_argument('--classifier', default='lasso', type=str)
    parser.add_argument('--embedding_size', default=50, type=int)
    parser.add_argument('--flag_partitions', default=True, type=bool)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='Process representations.')
args = parse_arguments(parser)

embedding_size = args.embedding_size
seed_value = 100

if args.dataset == 'medcon':
    df_data = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, 'bbdd_medcon.csv')), sep=',')
    var_name = 'medcon'
elif args.dataset == 'medications':
    df_data = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, 'bbdd_medications.csv')), sep=',')
    var_name = 'medications'
else:
    df_data = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, 'bbdd_allmed.csv')), sep=',')
    var_name = 'all_med'

print(df_data.head())

# df_medications['medications_post'] = df_medications['medications'].apply(transform_list)

# list_words = list(df_data['medications'])

# df_clean_medications = clean_all(df_medications, 'medications_post')

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=4, max_features=embedding_size)
m_bow_sparse = bow_vectorizer.fit_transform(df_data[var_name])

# tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.05, max_features=embedding_size)
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.1)
# tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
# tfidf_vectorizer = TfidfVectorizer()
m_tfidf_sparse = tfidf_vectorizer.fit_transform(df_data[var_name].values)

df_tfidf_sklearn = pd.DataFrame(m_tfidf_sparse.toarray(), columns=tfidf_vectorizer.get_feature_names())
print(df_tfidf_sklearn.mean(axis=0).sort_values(ascending=False).head(30))

m_bow = m_bow_sparse.toarray()
m_tfidf = m_tfidf_sparse.toarray()

tokenized_words = df_data[var_name].apply(lambda x: x.split())
# tokenized_words = list(df_data[var_name].values)
model_w2v = Word2Vec(min_count=2,
                     # sentences=tokenized_words,
                     seed=seed_value,
                     window=10,
                     vector_size=embedding_size)

model_w2v.build_vocab(tokenized_words)
model_w2v.train(tokenized_words, total_examples=len(tokenized_words), epochs=100)

# print(model_w2v.wv.most_similar("h04aa"))

m_word2vec = word2vec_features(df_data[var_name], model_w2v)

# x_vals, y_vals, labels = reduce_dimensions(model_w2v)
# plot_with_matplotlib(x_vals, y_vals, labels)

print(m_bow.shape)
print(m_tfidf.shape)
print(m_word2vec.shape)
y_label = df_data['label'].values

list_acc = []
list_aucroc = []

for idx in np.arange(1, 6, 1):

    x_train, x_test, y_train, y_test = train_test_split(m_bow, y_label, random_state=idx, test_size=0.2)
    # x_train, x_test, y_train, y_test = train_test_split(m_tfidf, y_label, random_state=idx, test_size=0.2)

    # x_train, x_test, y_train, y_test = train_test_split(df_data.medcon.values, df_data.label.values, random_state=idx, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(x_train)

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print('partition: ', idx, x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    if args.classifier == 'lasso':

        model_lasso = Lasso(max_iter=5000, random_state=idx)

        hyperparameter_space = {
            'alpha': np.logspace(-1.5, 0.4, 10)
        }

        grid_cv = GridSearchCV(estimator=model_lasso, param_grid=hyperparameter_space, scoring='accuracy', cv=3)
        grid_cv.fit(x_train, y_train)

        clf_model = grid_cv.best_estimator_
        clf_model.fit(x_train, y_train)
        y_pred = clf_model.predict(x_test)

        for i in range(len(y_pred)):
            if y_pred[i] >= 0.5:
                y_pred[i] = 1.0
            else:
                y_pred[i] = 0.0

    elif args.classifier == 'sgd':

        search_best_hyperparams(x_train, y_train, x_test, y_test)

    elif args.classifier == 'mlp':

        mlp = MLPClassifier(max_iter=40000, random_state=idx)

        mlp_param_grid = {
            'hidden_layer_sizes': [(40, 20), (20, 10)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.3],
            'learning_rate': ['constant', 'sgd']
        }

        grid_cv = GridSearchCV(mlp, param_grid=mlp_param_grid, cv=3, scoring='accuracy')
        grid_cv.fit(x_train, y_train)
        print(grid_cv.best_params_)
        mlp_clf = grid_cv.best_estimator_

        y_pred = mlp_clf.predict(x_test)

    elif args.classifier == 'svm':

        model_clf_generic = LinearSVC(max_iter=100000, random_state=idx)

        hyperparameter_space = {
            'C': np.logspace(-0.9, 0.9, 10)
        }

        grid_cv = GridSearchCV(estimator=model_clf_generic, param_grid=hyperparameter_space, scoring='roc_auc', cv=3)
        grid_cv.fit(x_train, y_train)

        clf_model = grid_cv.best_estimator_
        clf_model.fit(x_train, y_train)

        y_pred = clf_model.predict(x_test)

    else:

        lenght_train = x_train.shape[0]
        lenght_15_percent_val = int(0.15 * lenght_train)
        lenght_20_percent_val = int(0.20 * lenght_train)

        tuned_parameters = {
            'max_depth': range(2, 14, 2),
            'min_samples_split': range(lenght_15_percent_val, lenght_20_percent_val),
            # 'min_samples_split': range(2, 40),
        }

        model_tree = DecisionTreeClassifier(random_state=idx)

        grid_cv = GridSearchCV(estimator=model_tree, param_grid=tuned_parameters, scoring='accuracy', cv=3)
        grid_cv.fit(x_train, y_train)
        clf_model = grid_cv.best_estimator_

        clf_model.fit(x_train, y_train)
        y_pred = clf_model.predict(x_test)

    acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations(y_test, y_pred)

    list_acc.append(acc_val)
    list_aucroc.append(roc_auc_val)

print('acc: {}+{}'.format(np.mean(np.array(list_acc)), np.std(np.array(list_acc))))
print('aucroc: {}+{}'.format(np.mean(np.array(list_acc)), np.std(np.array(list_acc))))

# print('acc: {}, specificity: {}, recall: {}, roc: {}'.format(acc_mean, acc_std))