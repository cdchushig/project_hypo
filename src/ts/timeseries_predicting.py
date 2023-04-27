import matplotlib.pyplot as plt
import seaborn as sns
from darts import TimeSeries
from darts.dataprocessing.transformers import (
    Scaler,
    MissingValuesFiller,
    Mapper,
    InvertibleMapper,
)
from darts.utils.missing_values import fill_missing_values
from darts.models import ExponentialSmoothing
from matplotlib.dates import DateFormatter, DayLocator


def plot_gb_time_series(df, ts_name, gb_name, value_name, figsize=(20, 7), title=None):
    xtick_locator = DayLocator(interval=1)
    xtick_dateformatter = DateFormatter('%m/%d/%Y')
    fig, ax = plt.subplots(figsize=figsize)
    for key, grp in df.groupby([gb_name]):
        ax = grp.plot(ax=ax, kind='line', x=ts_name, y=value_name, label=key, marker='o')
    ax.xaxis.set_major_locator(xtick_locator)
    ax.xaxis.set_major_formatter(xtick_dateformatter)
    ax.autoscale_view()
    ax.legend(loc='upper left')
    plt.xticks(rotation=90)
    plt.grid()
    plt.xlabel('')
    plt.ylim(0, df[value_name].max() * 1.25)
    plt.ylabel(value_name)
    if title is not None:
        plt.title(title)
    plt.show()


def plot_count_records(df_ts):

    dfx = df_ts.groupby(['patient_id']).agg(
        count=('mean', 'count')
    ).reset_index()

    ax = sns.barplot(data=dfx, x='patient_id', y='count')
    plt.show()


# df_ts = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_PRE, 'bbdd_cgm_post.csv')))
# df_ts['Date'] = pd.to_datetime(df_ts['mydate'])
# df_ts_filtered = df_ts[df_ts['PtID'].isin([1, 2, 3, 4])]


print('holi')

# path_json_sensor_post = str(Path.joinpath(consts.PATH_PROJECT_DATA_PRE, 'sensor_post.json'))
# with open(path_json_sensor_post, 'w') as outfile:
#     json.dumps(json_string, outfile)

# print(df_js.columns)


# Round to minutes
# df['Datetime'] = df['Datetime'].dt.round('min')
# df = df.set_index('Datetime').resample('600S').asfreq()


# fig, ax = plt.subplots(nrows=1, ncols=1)
# for key, grp in df_ts_filtered.groupby(['patient_id']):
#     print(grp)
#     ax.plot(grp['Date'], grp['mean'], label="patient {0:02d}".format(key))
#
# plt.legend(loc='best')
# plt.show()

# plot_gb_time_series(df_ts_filtered, 'PtID', 'Date', 'Glucose', figsize=(10, 5), title="Random data")



# df_ts = df_ts[df_ts['patient_id'] == 2]
# ts = TimeSeries.from_dataframe(df_ts, time_col='Date', value_cols=['mean'], fillna_value=True, freq='10min')
# ts = TimeSeries.from_dataframe(df_ts, time_col='Date', value_cols=['mean'], freq='10T')


