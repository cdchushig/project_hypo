import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from collections import Counter
from ctgan import CTGANSynthesizer, TVAESynthesizer
from sdv.tabular import GaussianCopula, CopulaGAN
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTEN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
import torch
import pickle
import logging
import coloredlogs
from locale import atof, setlocale, LC_NUMERIC
from datetime import datetime
from utils import consts as consts
from utils.loader import load_train_test_set_by_partition
from utils.eval_utils import compute_classification_prestations, plot_pmfs_by_varname
from utils.utils import join_real_synthetic_data, save_parameters, check_repeated_patients, \
    save_samples_wrong_classified, get_x_train_classes_and_ids, get_filtered_datasets

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

list_interpretable_models = ['dt', 'lasso', 'random-forest', 'svm']

setlocale(LC_NUMERIC, '')


def parse_arguments(parser):
    parser.add_argument('--disease', default='cvd', type=str)
    parser.add_argument('--classifier', default='svm', type=str)
    parser.add_argument('--type_encoding', default='tfidf', type=str)
    parser.add_argument('--embedding_size', default='0.5', type=str)
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--flag_fs', default=True, type=bool)
    parser.add_argument('--flag_fs', default=True, type=bool)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='Process representations.')
args = parse_arguments(parser)

list_acc_values = []
list_specificity_values = []
list_recall_values = []
list_auc_values = []
list_feature_importance = []

generic_name = '{}_{}_{}_{}'.format(args.disease, args.type_encoding, args.oversampler, args.classifier)

for idx in np.arange(1, 6, 1):

    generic_name_partition = '{}_seed_{}'.format(generic_name, idx)

    logger.info('Experiment with: {}'.format(generic_name_partition))

    df_x_train, y_train, df_x_test, y_test = load_train_test_set_by_partition_smooth(idx, args.disease, args.type_encoding)

    v_column_names_all = df_x_train.columns.values

    if args.flag_fs:
        df_x_train, df_x_test = get_filtered_datasets(args.disease, args.type_encoding, df_x_train, df_x_test)

    v_column_names = df_x_train.columns.values
    x_train = df_x_train.to_numpy().astype('float')
    x_test = df_x_test.to_numpy().astype('float')


    if args.oversampler == 'tvae':
        oversampler_model = TVAESynthesizer(epochs=args.num_epochs, batch_size=args.batch_size)
        oversampler_model.fit(df_x_train_class_min_real, v_column_names)
        logger.info('Training with TVAE {}'.format(''))
    elif args.oversampler == 'gaussian_copula':
        logger.info('Training with Gaussian Copula')
        oversampler_model = GaussianCopula(categorical_transformer='categorical')
        oversampler_model.fit(df_x_train_class_min_real)
    elif args.oversampler == 'copula_gan':
        logger.info('Training with CopulaGAN')
        oversampler_model = GaussianCopula(categorical_transformer='categorical')
        oversampler_model.fit(df_x_train_class_min_real)
    elif args.oversampler == 'ctgan':
        logger.info('Training with CTGAN')
        oversampler_model = CTGANSynthesizer(epochs=args.num_epochs, batch_size=args.batch_size)
        oversampler_model.fit(df_x_train_class_min_real, v_column_names)
    elif args.oversampler == 'medgan':
        logger.info('Training with medgan')
        path_synth_data = str(Path.joinpath(consts.PATH_PROJECT_SYNTHETIC_DATA,
                                            args.oversampler, args.oversampler,
                                            'seed_{}_{}_generated.npy'.format(idx, imbalance_ratio)))
        x_synth_data = np.load(path_synth_data)
        x_synth_data[x_synth_data >= 0.5] = 1.0
        x_synth_data[x_synth_data < 0.5] = 0.0

    elif args.oversampler == 'smoten':
        logger.info('Training with SMOTEN')
        oversampler_model = SMOTEN(random_state=idx, sampling_strategy=imbalance_ratio)
    else:
        logger.info('Training with rus')
        oversampler_model = RandomUnderSampler(random_state=idx, sampling_strategy=imbalance_ratio)

    if args.oversampler in list_conventional_oversamplers:

        x_train_maj_resampled_min, y_train_maj_resampled_min = oversampler_model.fit_resample(x_train, y_train)
        y_meta_train_with_both_min_maj = y_train_maj_resampled_min

    elif args.oversampler in list_medgan_oversamplers:

        df_x_train_class_min_syn = pd.DataFrame(x_synth_data, columns=v_column_names_all)
        df_x_train_class_min_syn = df_x_train_class_min_syn.drop(['high_chol_yn_no'], axis=1)
        df_x_train_class_min_syn = df_x_train_class_min_syn.rename(columns={'high_chol_yn_yes': 'high_cholesterol'})
        df_x_train_class_min_syn = df_x_train_class_min_syn.loc[:, v_column_names]

        num_samples_syn = df_x_train_class_min_syn.shape[0]

        x_train_resampled_class_min, y_train_resampled_class_min, y_meta_train_with_both_min_maj = join_real_synthetic_data(
            df_x_train_class_min_real.values,
            df_x_train_class_min_syn.values,
            y_train_class_min_real,
            num_samples_maj,
            num_samples_syn,
            id_label_min)

        x_train_maj_resampled_min = np.concatenate((df_x_train_class_maj.values, x_train_resampled_class_min), axis=0)
        y_train_maj_resampled_min = np.concatenate((y_train_class_maj, y_train_resampled_class_min))

    else:

        df_x_train_class_min_syn = oversampler_model.sample(num_samples_syn)

        oversampler_model.save(
            str(Path.joinpath(consts.PATH_PROJECT_MODELS, 'model_oversampler_{}.pkl'.format(generic_name_partition))))

        df_x_train_class_min_syn.to_csv(str(Path.joinpath(consts.PATH_PROJECT_SYNTHETIC_DATA, args.oversampler,
                                                          'df_synthetic_class_min_{}.csv'.format(
                                                              generic_name_partition))), index=False)
        df_x_train_class_min_real.to_csv(str(Path.joinpath(consts.PATH_PROJECT_SYNTHETIC_DATA, args.oversampler,
                                                           'df_real_class_min_{}.csv'.format(generic_name_partition))),
                                         index=False)

        check_repeated_patients(df_x_train_class_min_real.values, df_x_train_class_min_syn.values,
                                df_x_train_class_maj.values, generic_name_partition)

        x_train_resampled_class_min, y_train_resampled_class_min, y_meta_train_with_both_min_maj = join_real_synthetic_data(
            df_x_train_class_min_real.values,
            df_x_train_class_min_syn.values,
            y_train_class_min_real,
            num_samples_maj,
            num_samples_syn,
            id_label_min)

        x_train_maj_resampled_min = np.concatenate((df_x_train_class_maj.values, x_train_resampled_class_min), axis=0)
        y_train_maj_resampled_min = np.concatenate((y_train_class_maj, y_train_resampled_class_min))

    if args.type_sampling == 'hybrid':
        logger.info('Training with hybrid approach')

        rus = RandomUnderSampler(random_state=idx)
        x_train_resampled, y_train_resampled = rus.fit_resample(x_train_maj_resampled_min, y_train_maj_resampled_min)
        v_indices_resampled = rus.sample_indices_
        y_label_real_syn_total_resampled = y_meta_train_with_both_min_maj[v_indices_resampled]

    elif args.type_sampling == 'over':
        logger.info('Training with oversampling approach')

        x_train_resampled = x_train_maj_resampled_min
        y_train_resampled = y_train_maj_resampled_min
        y_label_real_syn_total_resampled = y_meta_train_with_both_min_maj

    else:
        logger.info('Training with rus/smoten')
        x_train_resampled, y_train_resampled = oversampler_model.fit_resample(x_train, y_train)
        y_label_real_syn_total_resampled = [consts.LABEL_CLASS_MAJ_REAL if label == 0 else consts.LABEL_CLASS_MIN_REAL for label in y_train_resampled]

    logger.info('Resampled x_train dataset shape {}'.format(x_train_resampled.shape))
    logger.info('Resampled y_train dataset shape {}'.format(Counter(y_train_resampled)))

    if args.type_encoding == 'target':
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_resampled_scaled = scaler.transform(x_train_resampled)
        x_test_scaled = scaler.transform(x_test)
    else:
        x_train_resampled_scaled = x_train_resampled
        x_test_scaled = x_test

    df_x_train_resampled_scaled = pd.DataFrame(x_train_resampled_scaled, columns=v_column_names)
    df_x_train_resampled_scaled.to_csv(Path.joinpath(consts.PATH_PROJECT_SYNTHETIC_DATA, args.oversampler,
                                                     'df_x_train_resampled_scaled_{}.csv'.format(
                                                         generic_name_partition)), index=False)

    y_labels_maj_min = np.c_[y_train_resampled, y_label_real_syn_total_resampled]
    df_y_labels_maj_min = pd.DataFrame(y_labels_maj_min, columns=['label', 'label_meta'])

    df_y_labels_maj_min.to_csv(Path.joinpath(consts.PATH_PROJECT_SYNTHETIC_DATA, args.oversampler,
                                             'df_y_train_resampled_scaled_{}.csv'.format(generic_name_partition)),
                               index=False)

    if args.classifier == 'svm':
        model_clf_generic = LinearSVC(max_iter=100000, random_state=idx)

        hyperparameter_space = {
            'C': np.logspace(-0.9, 0.9, 10)
            # 'C': [0.1, 1, 10, 100, 1000],
        }

        grid_cv = GridSearchCV(estimator=model_clf_generic, param_grid=hyperparameter_space, scoring='roc_auc', cv=3)
        grid_cv.fit(x_train_resampled_scaled, y_train_resampled)
        clf_model = grid_cv.best_estimator_
        clf_model.fit(x_train_resampled_scaled, y_train_resampled)
        y_pred = clf_model.predict(x_test_scaled)

        logger.info('Training with {}, params: {}, best_params: {}'.format(args.classifier, clf_model.get_params(),
                                                                           grid_cv.best_params_))

        list_feature_importance.append(clf_model.coef_.reshape((-1)))

    elif args.classifier == 'dt':
        lenght_train = len(x_train_resampled_scaled)
        lenght_15_percent_val = int(0.15 * lenght_train)
        lenght_20_percent_val = int(0.20 * lenght_train)

        tuned_parameters = {
            'max_depth': range(2, 14, 2),
            'min_samples_split': range(lenght_15_percent_val, lenght_20_percent_val),
        }

        model_tree = DecisionTreeClassifier(random_state=idx)

        grid_cv = GridSearchCV(estimator=model_tree, param_grid=tuned_parameters, scoring='roc_auc', cv=3)
        grid_cv.fit(x_train_resampled_scaled, y_train_resampled)
        clf_model = grid_cv.best_estimator_

        clf_model.fit(x_train_resampled_scaled, y_train_resampled)
        y_pred = clf_model.predict(x_test_scaled)

        list_feature_importance.append(clf_model.feature_importances_)

    elif args.classifier == 'random-forest':
        lenght_train = len(x_train_resampled_scaled)
        lenght_15_percent_val = int(0.15 * lenght_train)
        lenght_20_percent_val = int(0.20 * lenght_train)
        model_tree = RandomForestClassifier(random_state=idx)
        # max_depth = range(1, 3, 1) #cancer y cvd+cancer
        # max_depth = range(1, 4, 1) #cvd y cancer
        max_depth = range(1, 3, 1)  # cvd-CANCER
        min_samples_split = range(lenght_15_percent_val, lenght_20_percent_val)
        tuned_parameters = dict(max_depth=max_depth, min_samples_split=min_samples_split)

        grid_cv = GridSearchCV(estimator=model_tree, param_grid=tuned_parameters, scoring='roc_auc', cv=3)
        grid_cv.fit(x_train_resampled_scaled, y_train_resampled)

        clf_model = grid_cv.best_estimator_
        clf_model.fit(x_train_resampled_scaled, y_train_resampled)

        y_pred = clf_model.predict(x_test_scaled)

        list_feature_importance.append(clf_model.feature_importances_)

    elif args.classifier == 'knn':

        hyperparameter_space = {
            'leaf_size': list(range(1, 10)),
            'n_neighbors': list(range(1, 21))
        }

        grid_cv = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=hyperparameter_space, scoring='roc_auc',
                               cv=3)
        grid_cv.fit(x_train_resampled_scaled, y_train_resampled)
        clf_model = grid_cv.best_estimator_
        clf_model.fit(x_train_resampled_scaled, y_train_resampled)

        y_pred = clf_model.predict(x_test_scaled)

        logger.info('Training with knn {}'.format(grid_cv.best_params_))

        list_feature_importance.append(0.0)

    elif args.classifier == 'mlp':

        # hyperparameter_space = {
        #     'hidden_layer_sizes': [(26,), (26, 10,)],
        #     'optimizer': ['rmsprop', 'adam'],
        #     'init': ['glorot_uniform', 'normal', 'uniform'],
        #     'dropout': [0.1, 0.2]
        # }

        # mlp_model = KerasClassifier(build_fn=build_mlp_model, verbose=0)

        # grid_cv = GridSearchCV(mlp_model, param_grid=hyperparameter_space, scoring='roc_auc', cv=3)

        hyperparameter_space = {
            'hidden_layer_sizes': [(26,), (26, 10), (26, 5)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            # 'alpha': [0.0001, 0.05],
            # 'learning_rate': ['constant', 'adaptive'],
        }

        grid_cv = GridSearchCV(MLPClassifier(max_iter=5000, random_state=idx), param_grid=hyperparameter_space, scoring='roc_auc', cv=3)

        grid_cv.fit(x_train_resampled_scaled, y_train_resampled)

        clf_model = grid_cv.best_estimator_
        history = clf_model.fit(x_train_resampled, y_train_resampled)

        print('history: ', history)

        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()

        logger.info('Training with mlp, params: {}, best params: {}, '.format(grid_cv.best_estimator_.get_params(),
                                                                              grid_cv.best_params_))

        y_pred = clf_model.predict(x_test_scaled)

        list_feature_importance.append(0.0)

    elif args.classifier == 'lasso':
        model_lasso = Lasso(max_iter=1000, random_state=idx)

        hyperparameter_space = {
            # 'alpha': np.logspace(-2.5, 2.5, 5)
            'alpha': np.logspace(-1.5, 0.4, 10)
        }

        grid_cv = GridSearchCV(estimator=model_lasso, param_grid=hyperparameter_space, scoring='roc_auc', cv=3)
        grid_cv.fit(x_train_resampled_scaled, y_train_resampled)

        clf_model = grid_cv.best_estimator_
        clf_model.fit(x_train_resampled_scaled, y_train_resampled)
        y_pred = clf_model.predict(x_test_scaled)

        for i in range(len(y_pred)):
            if y_pred[i] >= 0.5:
                y_pred[i] = 1.0
            else:
                y_pred[i] = 0.0

        list_feature_importance.append(clf_model.coef_)

        logger.info('Training with lasso {}'.format(clf_model.get_params()))

    save_parameters(generic_name_partition, clf_model.get_params())
    save_samples_wrong_classified(x_test_scaled, y_test, y_pred, args.oversampler, generic_name_partition)

    acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations(y_test, y_pred)

    list_acc_values.append(acc_val)
    list_specificity_values.append(specificity_val)
    list_recall_values.append(recall_val)
    list_auc_values.append(roc_auc_val)

    pickle.dump(clf_model, open(str(Path.joinpath(consts.PATH_PROJECT_MODELS,
                                                  'model_clf_{}.sav'.format(generic_name))), 'wb'))

if args.classifier in list_interpretable_models:
    importance_clf = sum(list_feature_importance) / len(list_feature_importance)
    df_importance_clf = pd.DataFrame(importance_clf)
    df_importance_clf.to_csv(
        str(Path.joinpath(consts.PATH_PROJECT_COEFS, args.oversampler, 'importance_{}.csv'.format(generic_name))))

    filename_coeffs = 'coeffs_{}_{}'.format(args.type_sampling, args.type_encoding)
    filename_coeffs = '{}_{}.csv'.format(filename_coeffs, 'fs') if args.flag_fs else '{}.csv'.format(filename_coeffs)
    path_coeffs = str(Path.joinpath(consts.PATH_PROJECT_COEFS, filename_coeffs))

    df_coeff_importance = pd.read_csv(path_coeffs)
    df_coeff_importance[generic_name] = importance_clf
    df_coeff_importance.to_csv(path_coeffs, index=False)

mean_std_specificity = np.mean(list_specificity_values), np.std(list_specificity_values)
mean_std_accuracy = np.mean(list_acc_values), np.std(list_acc_values)
mean_std_recall = np.mean(list_recall_values), np.std(list_recall_values)
mean_std_auc = np.mean(list_auc_values), np.std(list_auc_values)

print('accuracy:', mean_std_accuracy)
print('specificity:', mean_std_specificity)
print('recall:', mean_std_recall)
print('AUC:', mean_std_auc)

metrics = mean_std_accuracy, mean_std_recall, mean_std_specificity, mean_std_auc
current_time = datetime.now()
str_date_time = current_time.strftime("%m/%d/%Y, %H:%M:%S")
df_metrics = pd.DataFrame(metrics, columns=['mean', 'std'], index=['accuracy', 'recall', 'specificity', 'auc'])
df_metrics.to_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, args.oversampler,
                                    'metrics_{}.csv'.format(generic_name))))

exp_name = '{}+{}+{}+w{}'.format(args.oversampler, args.classifier, imbalance_ratio, weight)
exp_name = '{}+{}'.format(exp_name, 'fs') if args.flag_fs else exp_name

new_row_auc = {'run_date': str_date_time,
               'model': exp_name,
               'eval_metric': 'auc',
               'type_encoding': args.type_encoding,
               'type_sampling': args.type_sampling,
               'mean': mean_std_auc[0],
               'std': mean_std_auc[1]}

new_row_sensitivity = {'run_date': str_date_time,
                       'model': exp_name,
                       'eval_metric': 'sensitivity',
                       'type_encoding': args.type_encoding,
                       'type_sampling': args.type_sampling,
                       'mean': mean_std_recall[0],
                       'std': mean_std_recall[1]}

df_metrics_classification = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, 'metrics_classification.csv')))
df_metrics_classification = df_metrics_classification.append(new_row_auc, ignore_index=True)
df_metrics_classification = df_metrics_classification.append(new_row_sensitivity, ignore_index=True)

df_metrics_classification.to_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, 'metrics_classification.csv')),
                                 index=False)
