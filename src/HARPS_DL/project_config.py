# project_config.py

import os
from importlib import resources

DATASETS_PATH = ''
MODELS_BANK = ''
MODELS_TMP = ''


CSV_REAL_FILE = ''
LINELIST_FILE = ''

NORMALIZATION_PATH = resources.files('HARPS_DL.datasets').joinpath('normalization.json')

# older models
# MODELS_OLD_PATH = os.path.join(os.getcwd(), 'models')
