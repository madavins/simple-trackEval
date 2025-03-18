import os
import csv
import numpy as np
from collections import OrderedDict


def _load_simple_text_file(file_path, is_zipped=False, zip_file=None):
    """Loads a text file, handling both zipped and unzipped cases."""
    if is_zipped:
        import zipfile
        with zipfile.ZipFile(zip_file) as zf:
            with zf.open(file_path, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                raw_data = [row for row in reader]
    else:
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            raw_data = [row for row in reader]

    # Filter out empty rows and rows starting with '#'
    raw_data = [row for row in raw_data if row and not row[0].startswith('#')]

    data = raw_data

    # Convert numerical values to floats
    for row in data:
        for i in range(len(row)):
            try:
                row[i] = float(row[i])
            except ValueError:
                pass

    return data


def init_config(config, default_config, name=None):
    """Initialise non-given config values with defaults"""
    if config is None:
        config = default_config
    else:
        for k in default_config.keys():
            if k not in config.keys():
                config[k] = default_config[k]
    if name and config['PRINT_CONFIG']:
        print('\n%s Config:' % name)
        for c in config.keys():
            print('%-20s : %-30s' % (c, config[c]))
    return config


class TrackEvalException(Exception):
    """Custom exception for catching expected errors."""
    ...