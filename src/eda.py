import pandas as pd
import numpy as np
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils import consts as consts
from utils import plotter as plotter

LIB_TO_KG_FACTOR = 0.45359237
INCh_TO_CM_FACTOR = 2.54


df_medchart = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_RAW, 'BMedChart.csv')), sep='|')

df_medchart['WeightProcessed'] = df_medchart[['Weight', 'WeightUnits']].apply(lambda x: x['Weight'] if x['WeightUnits'] == 'kg' else x['Weight'] * LIB_TO_KG_FACTOR, axis=1)
df_medchart['HeightProcessed'] = df_medchart[['Height', 'HeightUnits']].apply(lambda x: x['Height'] if x['HeightUnits'] == 'cm' else x['Height'] * INCh_TO_CM_FACTOR, axis=1)

print(df_medchart)


# df_medications = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_RAW, 'BMedication_Modified.csv')), sep=',')

df_medcon = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, 'bbdd_medcon.csv')), sep=',')
df_medications = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, 'bbdd_medications.csv')), sep=',')

# print(df_medcon[df_medcon.loc[:, 'PtID'] == 1])
# print(df_medcon[df_medcon.loc[:, 'PtID'] == 2])
# print(df_medcon[df_medcon.loc[:, 'PtID'] == 203])

# print(df_medcon)
# df_medcon_grouped = df_medcon.groupby('PtID')['MCLLTReal'].apply(list).reset_index()

# print(df_medcon.groupby('PtID')['MCLLTReal'].apply(list))
# print(df_medcon.groupby('PtID')['MCLLTReal'].apply(list).reset_index().shape)

# print(np.unique(df_medications['DrugName'].value_counts().reset_index().iloc[:, 0].values).shape)
# print(np.unique(df_medications['atc_code'].value_counts().reset_index().iloc[:, 0].values).shape)
# print(np.unique(df_medications['atc_code_aggregated'].value_counts().reset_index().iloc[:, 0].values).shape)

df_medications_case = df_medications[df_medications['BCaseControlStatus'] == 'Case']
df_medications_control = df_medications[df_medications['BCaseControlStatus'] == 'Control']

print('case: ', df_medications_case.shape)
print('control:', df_medications_control.shape)

# plotter.plot_wordcloud(df_medications, 'medications', type_patient='all', flag_save_figure=True)
# plotter.plot_wordcloud(df_medications_case, 'medications', type_patient='case', flag_save_figure=True)
# plotter.plot_wordcloud(df_medications_control, 'medications', type_patient='control', flag_save_figure=True)

df_medcon_case = df_medcon[df_medcon['BCaseControlStatus'] == 'Case']
df_medcon_control = df_medcon[df_medcon['BCaseControlStatus'] == 'Control']

# plotter.plot_wordcloud(df_medcon, 'medcon', type_patient='all')
plotter.plot_wordcloud(df_medcon_case, 'medcon', type_patient='case', flag_save_figure=False)
plotter.plot_wordcloud(df_medcon_control, 'medcon', type_patient='control', flag_save_figure=False)


print(df_medcon_case.head())
print(df_medcon_control.head())