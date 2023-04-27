from pathlib import Path

PATH_PROJECT_DIR = Path(__file__).resolve().parents[1]
PATH_PROJECT_DATA = Path.joinpath(PATH_PROJECT_DIR, 'data')
PATH_PROJECT_DATA_RAW = Path.joinpath(PATH_PROJECT_DIR, 'data', 'raw')
PATH_PROJECT_DATA_PRE = Path.joinpath(PATH_PROJECT_DIR, 'data', 'preprocessed')
PATH_PROJECT_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'figures')
PATH_PROJECT_DATA_T1D = Path.joinpath(PATH_PROJECT_DIR, 'data', 't1d')
