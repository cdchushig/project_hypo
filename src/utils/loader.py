import pandas as pd
from pathlib import Path
import logging
import coloredlogs
from collections import Counter

import utils.consts as consts

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def load_train_test_set_by_partition(id_partition: int, disease: str, type_encoding: str) -> (pd.DataFrame,
                                                                                              pd.DataFrame,
                                                                                              pd.DataFrame,
                                                                                              pd.DataFrame):

    path_dataset = Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED, disease, type_encoding)

    df_x_train = pd.read_csv(str(Path.joinpath(path_dataset, 'x_train_{}{}.csv'.format(type_encoding, id_partition))))

    df_y_train = pd.read_csv(str(Path.joinpath(path_dataset, 'y_train_{}{}.csv'.format(type_encoding, id_partition))),
                             header=None)

    df_x_test = pd.read_csv(str(Path.joinpath(path_dataset, 'x_test_{}{}.csv'.format(type_encoding, id_partition))))

    df_y_test = pd.read_csv(str(Path.joinpath(path_dataset, 'y_test_{}{}.csv'.format(type_encoding, id_partition))),
                            header=None)