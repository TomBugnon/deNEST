""" default config file.
Overwrite default by writing user define globals in user_config.py.
If you write in user_config.py, please add it to your global .gitignore
(not the project's .gitignore)
"""

import os.path

INPUT_DIR = os.path.abspath('input')
SAVE_DIR = os.path.abspath('SAVE_DIR')
INPUT_SUBDIRS = {'raw_input': 'raw_input',
                 'preprocessed_input': 'preprocessed_input',
                 'raw_input_sets': 'raw_input_sets',
                 'preprocessed_input_sets': 'preprocessed_input_sets',
                 'stimuli': 'stimuli'}
METADATA_FILENAME = 'metadata.yml'
PYTHON_SEED = 94
