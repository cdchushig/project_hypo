from pathlib import Path

PATH_PROJECT_DIR = Path(__file__).resolve().parents[2]
PATH_PROJECT_DATA = Path.joinpath(PATH_PROJECT_DIR, 'data')
PATH_PROJECT_DATA_RAW = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw')
PATH_PROJECT_DATA_PREPROCESSED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed')

PATH_PROJECT_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'figures')
PATH_PROJECT_FS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'fs')

PATH_PROJECT_REPORTS_EFA = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'efa')

TYPE_FEATURE_CONTINUOUS = 'c'
TYPE_FEATURE_DISCRETE = 'd'