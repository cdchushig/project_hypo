import numpy as np
import pandas as pd
import re
from pathlib import Path
from utils import consts as consts
from utils.utils import clean_all
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def remove_stopwords(article):
    # stop_words = set(stopwords.words('english'))

    article_tokens = word_tokenize(article)
    filtered_article = [word for word in article_tokens if not word in stop_words]
    return " ".join(filtered_article)


df_roster = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_RAW, 'BPtRoster.csv')), sep='|')
df_medcon = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_RAW, 'BMedicalConditions.csv')), sep='|')
df_medication = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_RAW, 'BMedication_Modified.csv')), sep=',')

# print(np.unique(df_medcon['PtID'].value_counts().reset_index().iloc[:, 0].values))
# print(np.unique(df_medcon['MCLLTReal'].value_counts().reset_index().iloc[:, 0].values).shape)

df_roster = df_roster.rename({'PtID': 'patient_id'}, axis=1)

# Group codes and medcon
df_medcon_grouped = df_medcon.groupby('PtID')['MCLLTReal'].apply(list).reset_index()
df_medication_grouped = df_medication.groupby('PtID')['atc_code_aggregated'].apply(list).reset_index()
df_medcon_grouped['medcon'] = df_medcon_grouped['MCLLTReal'].apply(lambda x: ' '.join(x))
df_medcon_grouped = clean_all(df_medcon_grouped, 'medcon')
df_medcon_grouped = df_medcon_grouped.rename({'PtID': 'patient_id', 'MCLLTReal': 'raw_medcon'}, axis=1)
df_medcon_final = pd.merge(df_medcon_grouped, df_roster, on='patient_id')

df_medication_grouped = df_medication_grouped.rename({'PtID': 'patient_id', 'atc_code_aggregated': 'raw_medications'}, axis=1)
df_medication_final = pd.merge(df_medication_grouped, df_roster, on='patient_id')
df_medication_final['medications'] = df_medication_final['raw_medications'].apply(lambda x: ' '.join(x))
df_medication_final['medications'] = df_medication_final['medications'].map(lambda x: x.lower())

le = preprocessing.LabelEncoder()
df_medication_final['label'] = le.fit_transform(df_medication_final['BCaseControlStatus'].values)
df_medcon_final['label'] = le.fit_transform(df_medcon_final['BCaseControlStatus'].values)

print(df_medcon_final)
print(df_medication_final)

df_concat = pd.merge(df_medication_final, df_medcon_final, on='patient_id')
df_concat['all_med'] = df_concat[['medications', 'medcon']].apply(lambda x: '{} {}'.format(x['medications'], x['medcon']), axis=1)

le = preprocessing.LabelEncoder()
df_concat['label'] = le.fit_transform(df_concat['BCaseControlStatus_x'].values)

df_concat.to_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, 'bbdd_allmed.csv')), index=False)

df_medication_final.to_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, 'bbdd_medications.csv')), index=False)
df_medcon_final.to_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, 'bbdd_medcon.csv')), index=False)