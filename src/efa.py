import numpy as np
import pandas as pd
import utils.consts as consts
from pathlib import Path
from sklearn import preprocessing
from utils.plotter import plot_survey_responses, plot_survey_profiles
from sklearn.cluster import AgglomerativeClustering

df_data = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_RAW, 'BBGAttitudeScale.csv')), sep='|')
dict_likert = {'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly agree': 5}

col_names = ['DealHypoEp', 'UndertreatHypo', 'HighBGDamage', 'FreqHypoDamage', 'DangersHighBG']
df_data = df_data.loc[:, col_names]

for i in df_data.columns:
    df_data[i] = df_data[i].replace(dict_likert)

df_data = df_data.dropna()

print(df_data.head())
print(df_data.shape)

scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(df_data.values)
df_data_scaled = pd.DataFrame(x_scaled, columns=df_data.columns.values)

ahc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
ahc.fit(df_data_scaled.values)
cluster_label = ahc.fit_predict(df_data_scaled.values)

x_all = np.concatenate((df_data.values, cluster_label.reshape(-1, 1)), axis=1)
df_all = pd.DataFrame(x_all, columns=np.concatenate((df_data.columns.values, np.array(['cluster_label']))))

df_cluster0 = df_all[df_all['cluster_label'] == 0]
df_cluster1 = df_all[df_all['cluster_label'] == 1]

plot_survey_responses(df_all.iloc[:, :-1], dict_likert, 'all', True)
plot_survey_responses(df_cluster0.iloc[:, :-1], dict_likert, 'cluster0', True)
plot_survey_responses(df_cluster1.iloc[:, :-1], dict_likert, 'cluster1', True)
plot_survey_profiles(df_all, dict_likert, title_figure='profiles', save_figure=True)
