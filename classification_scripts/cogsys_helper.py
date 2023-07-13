import pickle
import os
import json

import hdf5storage as h5s
import numpy as np
import yaml
import json
from scipy.io import loadmat


def setup_directory(path):
    """
    Create directory if path doesn't exists (getting around permission problems)
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    return


def load_yml(path):
    """
    read yml file into variable
    :param path: full path with extension (yaml, yml)
    :return: yml_var
    """
    with open('{0}'.format(path)) as ymlstream:
        try:
            yml_var= yaml.load(ymlstream, Loader=yaml.FullLoader)
        except yaml.YAMLError as excr:
            print(excr)
    return yml_var


def load_pickle(pklfile):
    """
    Read and return pickle file
    :param pklfile: full path with extension
    :return:
    """
    with open('{0}'.format(pklfile), 'rb') as f:
        re_pkl = pickle.load(f)

    return re_pkl


def load_metricjson(path):

    with open(path, 'r') as jsf:
        dictdump = json.load(jsf)

    return dictdump


def load_matfile(datafile):

    datamat = h5s.loadmat(datafile)
    return datamat


def export_pkl(targ_variable, path):
    """
    save variable (especially metric data structs) into pkl format for later use
    :param targ_variable: full path with extension (pkl)
    :param path:
    :return:
    """

    with open('{0}'.format(path), 'wb') as f:
        pickle.dump(targ_variable, f)

    return

def load_metricjson(path):

    with open(path, 'r') as jsf:
        dictdump = json.load(jsf)

    return dictdump

class np_to_json(json.JSONEncoder):
    '''
    Class to convert numpy objects to json serializable (aka tolist())
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def export_json(destination, data):
    with open(destination, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=np_to_json)

    return


def remove_eog(x, eog_pos, till_end=False):
    """
    Assumes x is: time x eeg x sample
    Support selective erasing of channel indices, doesn't really have to be eog but
    in most cases we only use them for eogs.
    Take care that there's a hard assumption that eeg channel dimension is always at the second dimension.
    :param x:
    :param eog_pos:
    :return:
    """
    ##x = x[:, 0:eog_pos, :]
    if till_end is True:
        x = x[:, 0:eog_pos, :]
    else:
        x = np.delete(x, eog_pos, 1)
    return x


def trim_trials(x, y, target_classes=[]):
    '''
    assumes x is preprocessed, and that trim idx shows idx of samples to save
    trim idx is defined in the main script
    :param x:
    :param y:
    :param target_classes:
    :return:
    '''

    remainder_idx = []

    for target_class in target_classes:
        class_idx = np.where(y==target_class)[0]
        remainder_idx = np.concatenate([remainder_idx, class_idx]).astype(int)

    new_x = x[remainder_idx]
    new_y = y[remainder_idx]

    #new_y = preproc_y(new_y, rearrange_dims=False, zero_numerical_y=True)

    return new_x, new_y

def preproc_y(y:np.ndarray, y_is_zeroed=False):
    if len(y.shape) > 1:
        y = y.reshape(-1)
    if not y_is_zeroed:
        y = y - np.min(y[:])
    y = y.astype(int)
    return y
