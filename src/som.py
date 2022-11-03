import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from utils import consts as consts


# df_fear = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_RAW, 'BHypoFearSurvey.csv')), sep='|')
df_fear = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, 'bbdd_fear.csv')))
df_fear = df_fear.fillna(0)
df_fear = df_fear.rename({'PtID': 'patient_id'}, axis=1)

df_roster = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_RAW, 'BPtRoster.csv')), sep='|')
df_roster = df_roster.rename({'PtID': 'patient_id'}, axis=1)

df_fear = pd.merge(df_fear, df_roster, on='patient_id')

print(df_fear.head(20))
print(df_fear.shape)

# print(df_fear.head())
# print(df_fear.shape)
# print(np.unique(df_fear['PtID']).shape)
#
# print(df_fear.describe())
#
# v_column_names = df_fear.columns.values[4:]
#
# scaler = StandardScaler()
# df_fear[v_column_names] = scaler.fit_transform(df_fear[v_column_names])
# print(df_fear.describe())
#
# df_fear_filtered = df_fear[v_column_names]
# df_fear_filtered.to_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, 'bbdd_fear_filtered.csv')), index=False)


# df_fear.to_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, 'bbdd_fear.csv')), index=False)