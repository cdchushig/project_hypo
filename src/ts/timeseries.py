from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import functools as ft
from itertools import chain, combinations
from scipy import stats
from tqdm import tqdm
from itertools import islice, tee
from matplotlib import pyplot as plt
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import multiprocessing
from ast import literal_eval
import consts as consts


LIST_PATIENT_IDS_REMOVED = [50, 79, 190, 84, 181, 184]
FILL_MISSING_DATA_VALUE = 10000


def check_nan_values(x_set):
    nan_catcher = np.zeros((1, x_set.shape[1]))
    type_catcher = np.zeros((1, x_set.shape[1]))
    catcher = []

    i = 0
    for column_name, column_data in x_set.iteritems():
        catcher_col = []
        for row in column_data.values:
            if pd.isnull(row):
                nan_catcher[0, i] += 1
            else:
                # Check if value can be cast to numeric
                try:
                    row = float(row)
                except:
                    type_catcher[0, i] += 1
                    catcher_col.append(row)
        catcher.append(catcher_col)
        i += 1

    return nan_catcher, type_catcher, catcher


def build_datestamp(df_ts: pd.DataFrame) -> pd.DataFrame:

    df_ts_copy = df_ts.copy()

    df_ts_copy['month'] = 1
    df_ts_copy['year'] = 2020
    df_ts_copy['ndays'] = df_ts_copy['ndays'].astype(int)

    df_ts_copy['mydate'] = pd.to_datetime(
        df_ts_copy[['ndays', 'month', 'year', 'DeviceTm']].astype(str).apply(' '.join, 1),
        format='%d %m %Y %H:%M:%S'
    )

    return df_ts_copy


def create_sequence_by_columns(df_ts: pd.DataFrame) -> pd.DataFrame:
    df_ts['sequence'] = df_ts.sort_values(
        by=['PtID', 'DeviceDaysFromEnroll', 'DeviceTm']).groupby(
        ['PtID']
    ).cumcount() + 1

    return df_ts


def rescale_days(df_ts: pd.DataFrame) -> pd.DataFrame:
    df_ts_copy = df_ts.copy()
    list_dfs = []
    patient_ids = df_ts['PtID'].unique()
    for patient_id in patient_ids:
        df_ts_patient = df_ts_copy[df_ts_copy['PtID'] == patient_id]
        n_max_days = df_ts_patient['DeviceDaysFromEnroll'].unique().shape[0]
        scaler = MinMaxScaler(feature_range=(1, n_max_days))
        df_ts_patient['ndays'] = scaler.fit_transform(df_ts_patient.DeviceDaysFromEnroll.values.reshape(-1, 1))
        list_dfs.append(df_ts_patient)  

    df_ts_pre = pd.concat(list_dfs)
    return df_ts_pre


def plot_hist_glucose(df):
    ax = df.hist(column='Glucose', by='PtID', bins=25,
                 grid=False, figsize=(8, 10), layout=(3, 3),
                 sharex=True, color='#86bf91', zorder=2, rwidth=0.9)

    for i, x in enumerate(ax):

        # Despine
        # x.spines['right'].set_visible(False)
        # x.spines['top'].set_visible(False)
        # x.spines['left'].set_visible(False)

        # Switch off ticks
        x.tick_params(axis="both", which="both",
                      bottom="off", top="off", labelbottom="on",
                      left="off", right="off", labelleft="on")

        # Draw horizontal axis lines
        vals = x.get_yticks()
        for tick in vals:
            x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

        # Set x-axis label
        x.set_xlabel("Glucose", labelpad=20, weight='bold', size=12)

        # Set y-axis label
        if i == 1:
            x.set_ylabel("Sessions", labelpad=50, weight='bold', size=12)

        # Format y-axis label
        # x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

        # x.tick_params(axis='x', rotation=0)

    plt.show()


def plot_hist_glucose_time(df: pd.DataFrame, agg_func: str = 'count', flag_save_figure: bool = False):
    list_patient_ids = df['PtID'].unique()

    for patient_id in list_patient_ids:
        df_patient = df[df['PtID'] == patient_id]

        times = pd.DatetimeIndex(df_patient.DateTime)

        df_grouped = df_patient.groupby(['DeviceDaysFromEnroll', times.hour]).agg(
            mean=('Glucose', 'mean'), sum=('Glucose', 'sum'),
            count=('Glucose', 'count'), median=('Glucose', 'median'),
        ).reset_index()

        plt.figure()
        ax = sns.barplot(data=df_grouped, x='DateTime', y=agg_func)

        if flag_save_figure:
            plt.tight_layout()
            plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES,
                                          'hist_patient_{}_{}_{}_avg.png'.format(patient_id, 'time', agg_func))))
        else:
            # plt.show()
            print('showing figure')


def plot_glucose_patients(df, agg_func='count', flag_save_figure=False):

    dfx = df.groupby(['PtID']).agg(
        mean=('Glucose', 'mean'), sum=('Glucose', 'sum'),
        count=('Glucose', 'count'), median=('Glucose', 'median'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(20, 12))

    dfx = dfx.sort_values(by=[agg_func])

    ax = sns.barplot(data=dfx, x='PtID', y=agg_func, ax=ax)
    ax.tick_params(axis='x', rotation=90)

    if flag_save_figure:
        plt.tight_layout()
        plt.savefig(
            str(Path.joinpath(
                consts.PATH_PROJECT_FIGURES,
                'hist_records_{}.pdf'.format(agg_func))
            )
        )
    else:
        # plt.show()
        print('showing figure')


def get_rolling_amount(grp, freq, on_var='date', var_sum='amount'):
    return grp.rolling(freq, on=on_var)[var_sum].sum()


def powerset(iterable, n):
    s = list(iterable)
    return list(zip(*(islice(it, i, None) for i, it in enumerate(tee(s, n)))))
    # return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def select_seqs_days(dfp, patient_id, agg_func='count', max_records: int = 1000, n_seqs: int = 4):
    dfp = dfp.copy()
    list_selected_seqs = []

    for seq_days in powerset(dfp['DeviceDaysFromEnroll'], n_seqs):
        sum_records_by_seqs = int(dfp[dfp['DeviceDaysFromEnroll'].isin(seq_days)][agg_func].sum())
        if sum_records_by_seqs >= max_records:
            list_selected_seqs.append((patient_id, str(seq_days), sum_records_by_seqs))

    return list_selected_seqs


def select_glucose_days(df, agg_func='count', max_records=1100, n_seqs=4):

    df = df[~df['PtID'].isin(LIST_PATIENT_IDS_REMOVED)]

    dfx = df.groupby(['PtID', 'DeviceDaysFromEnroll']).agg(
        mean=('Glucose', 'mean'), sum=('Glucose', 'sum'),
        count=('Glucose', 'count'), median=('Glucose', 'median'),
    ).reset_index()

    list_seqs = []

    for patient_id in list(dfx['PtID'].unique()):
        dfp = dfx[dfx['PtID'] == patient_id].copy()
        list_seqs_to_select = select_seqs_days(dfp, patient_id, agg_func=agg_func, max_records=max_records, n_seqs=n_seqs)
        list_seqs.extend(list_seqs_to_select)

    df_seqs_sum = pd.DataFrame(np.asarray(list_seqs), columns=['patient_id', 'seq', 'sum'])

    df_seqs_sum = df_seqs_sum.astype(dtype={"patient_id": "int64",
                                            "seq": "object",
                                            "sum": "int64"
    })

    print('# patients pre : ', df['PtID'].unique().shape[0])
    print('# patients post: ', df_seqs_sum['patient_id'].unique().shape[0])

    return df_seqs_sum.sort_values(by=['patient_id', 'sum'], ascending=False).groupby(['patient_id']).first().reset_index()


def resample_by_mins(dfp, patient_id, freq_sampling='10min'):

    print(dfp.sort_values(by=['mydate']))

    dfp['new'] = dfp.groupby(pd.Grouper(key='mydate', freq=freq_sampling))['mydate'].transform('first')

    dfx = dfp.groupby(['new']).agg(
        mean=('Glucose', 'mean'), sum=('Glucose', 'sum'),
        count=('Glucose', 'count'), median=('Glucose', 'median'),
    ).reset_index()

    dfx['patient_id'] = patient_id

    return dfx


def plot_hist_glucose_days(df, sequence_id='ndays', agg_func='mean', flag_save_figure=True):

    list_patient_ids = df['PtID'].unique()

    for patient_id in list_patient_ids:
        df_patient = df[df['PtID'] == patient_id]

        dfx = df_patient.groupby([sequence_id]).agg(
            mean=('Glucose', 'mean'), sum=('Glucose', 'sum'),
            count=('Glucose', 'count'), median=('Glucose', 'median'),
        ).reset_index()

        plt.figure()
        ax = sns.barplot(data=dfx, x=sequence_id, y=agg_func)

        if flag_save_figure:
            plt.tight_layout()
            plt.savefig(
                str(Path.joinpath(
                    consts.PATH_PROJECT_FIGURES,
                    'hist_patient_{}_{}_{}.pdf'.format(patient_id, sequence_id.lower(), agg_func))
                )
            )
        else:
            # plt.show()
            print('showing figure')


def preprocess_raw_cgm():
    path_cgm_raw_data = str(Path.joinpath(consts.PATH_PROJECT_DATA_RAW, 'BDataCGM.csv'))
    df_ts = pd.read_csv(path_cgm_raw_data, sep='|')
    df_ts = create_sequence_by_columns(df_ts)
    df_ts_pre = rescale_days(df_ts)
    df_ts_post = build_datestamp(df_ts_pre)
    df_ts_post.to_csv(path_cgm_pre_data, index=False)


def create_datetime_cgm(df_ts: pd.DataFrame) -> pd.DataFrame:
    df_ts = create_sequence_by_columns(df_ts)
    df_ts_pre = rescale_days(df_ts)
    df_ts_post = build_datestamp(df_ts_pre)
    return df_ts_post


def preprocess_cgm_new_freq(df_ts: pd.DataFrame, df_selected_seqs: pd.DataFrame, freq_sampling=None) -> pd.DataFrame:

    list_dfs = []

    for patient_id in df_selected_seqs['patient_id'].unique():
        seq_str = df_selected_seqs.loc[df_selected_seqs['patient_id'] == patient_id, 'seq'].iat[0]
        list_seq_days = list(literal_eval(seq_str))
        print('patient_id: {}, seq: {}'.format(patient_id, list_seq_days))
        dfp = df_ts[(df_ts['PtID'] == patient_id) & (df_ts['DeviceDaysFromEnroll'].isin(list_seq_days))]
        dfp_resampled = create_datetime_cgm(dfp)
        if freq_sampling is not None:
            dfp_resampled = resample_by_mins(dfp, patient_id=patient_id, freq_sampling='10min')
        list_dfs.append(dfp_resampled)

    df_concat = pd.concat(list_dfs)

    return df_concat


def fill_missing_data(df_ts: pd.DataFrame, seq_id='Date'):
    df_ts_copy = df_ts.copy()
    idx = pd.period_range(min(df_ts[seq_id]), max(df_ts[seq_id]))
    df_ts.reindex(idx, fill_value=FILL_MISSING_DATA_VALUE)
    return df_ts


path_cgm_pre_data = str(Path.joinpath(consts.PATH_PROJECT_DATA_PRE, 'bbdd_cgm_pre.csv'))
df_ts = pd.read_csv(path_cgm_pre_data)

print(df_ts)

plot_hist_glucose_time(df_ts, agg_func='median', flag_save_figure=True)
plot_hist_glucose_days(df_ts, sequence_id='DeviceDaysFromEnroll', agg_func='count', flag_save_figure=False)
plot_glucose_patients(df_ts, flag_save_figure=True)

df_seqs_patients = select_glucose_days(df_ts, agg_func='count', max_records=964, n_seqs=5)
print(df_seqs_patients.sort_values(by=['sum'], ascending=False))

df_concat = preprocess_cgm_new_freq(df_ts, df_seqs_patients, freq_sampling=None)
df_concat.to_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PRE, 'bbdd_cgm_post.csv')), index=False)

# print(df_ts_post)
#
# df_ts_filter = df_ts_post[df_ts_post['PtID'].isin([99, 199, 142,
#                                                    85, 106, 83,
#                                                    23, 60, 124])]

# plot_hist_glucose(df_ts_filter)


# m_series = TimeSeries.from_group_dataframe(df_ts_post, group_cols=['PtID'], time_col='DateTime', value_cols="Glucose")
# m_series = TimeSeries.from_dataframe(df_ts_post, time_col='DateTime', value_cols="Glucose",
#                                      fill_missing_dates=True,
#                                      freq='5T')
# for ts in m_series:
#     ts.plot()
#     plt.show()




