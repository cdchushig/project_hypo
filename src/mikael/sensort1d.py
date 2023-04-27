import pandas as pd
import consts as consts
import json
from pathlib import Path
import operator


def load_json_data() -> dict:
    path_json_sensor = str(Path.joinpath(consts.PATH_PROJECT_DATA_PRE, 'sensort1d.json'))

    with open(path_json_sensor) as json_file:
        json_data = json.load(json_file)

    return json_data


def preprocess_json_data(dict_json: dict, type_data: str = 'sgv', save_csv: bool = False) -> pd.DataFrame:

    dict_data = dict_json[type_data]

    list_dates_records = []
    list_dates_empty = []

    for key_date, list_dicts_data in dict_data.items():
        list_dates_records.append((key_date, len(list_dicts_data)))
        if len(list_dicts_data) == 0:
            print(key_date, list_dicts_data)
            list_dates_empty.append((key_date, list_dicts_data))

    # list_dates_records.sort(key=lambda x: x[1])
    # n_empties = sum(int(value) for key_date, len_list in list_dates_records if len_list == 0)

    n_records_bolus = sum(map(operator.itemgetter(1), list_dates_records))

    print('n_records: ', n_records_bolus)
    print('n_index_pre: ', len(dict_data.keys()))
    print('n_index_empty: ', len(list_dates_empty))

    df_meta = (pd.concat(
        [pd.json_normalize(data=dict_data[x]) for x in dict_data], keys=dict_data.keys()).droplevel(level=1
    ).reset_index())

    print(df_meta.shape)
    print(df_meta.info())
    print(df_meta.sort_values(by=['index']))

    # print('n_records: ', sum(df_meta.groupby('index')['id'].count().reset_index()['id'].astype('int')))
    # print('n_index_post: ', df_meta['index'].unique().shape[0])

    if save_csv:
        df_meta.to_csv(
            str(Path.joinpath(consts.PATH_PROJECT_DATA_T1D, 'bbdd_t1d_{}.csv'.format(type_data))),
            index=False
        )

    return df_meta


json_data = load_json_data()
print(json_data.keys())

# print(json_data['photos'])
# print(json_data['transports'])
# print(json_data['weight'])

# dict_bolus = json_data['bolus']
# preprocess_bolus(dict_bolus)

# preprocess_json_data(json_data, type_data='sgv')
# preprocess_json_data(json_data, type_data='bolus')
# preprocess_json_data(json_data, type_data='exercises', save_csv=True)
# preprocess_json_data(json_data, type_data='nutrition', save_csv=False)
# preprocess_json_data(json_data, type_data='notes', save_csv=True)
# preprocess_json_data(json_data, type_data='smbg', save_csv=True)
# preprocess_json_data(json_data, type_data='basal', save_csv=True)
# preprocess_json_data(json_data, type_data='pumpEvents', save_csv=True)
# preprocess_json_data(json_data, type_data='photos', save_csv=True)

# preprocess_json_data(json_data, type_data='sleep', save_csv=True)
# preprocess_json_data(json_data, type_data='bloodPressure', save_csv=True)
preprocess_json_data(json_data, type_data='sleep', save_csv=True)


