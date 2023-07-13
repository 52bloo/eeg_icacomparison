# import hdf5storage as hdf5
import os
import sys
import argparse
import random
import numpy as np
from datetime import datetime
import yaml

from classification_scripts import cogsys_helper
#from classification_scripts.cogsys_helper import setup_directory
from classification_scripts.cogsys_helper import *
from architecture.mlstm_fcn import *
#import cogsys_helper

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import NLLLoss
from torch.utils.data import DataLoader, TensorDataset, Dataset


from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import skorch
from skorch.callbacks import LRScheduler, EarlyStopping, EpochScoring
from skorch.classifier import NeuralNetClassifier

#from skorch.

import braindecode

# example script for



def get_cmdarg():
    # using argparse for parsing arguments
    # configs tend to vary a lot by experiments so let's use yaml to get around possible headaches
    psr = argparse.ArgumentParser()
    # psr.add_argument()

    """
    remember that typechecking is not always necessary!
    example: psr.add_argument("--n", default=0, help='helptext')

    """

    psr.add_argument("--run_type", default='arti vs no artifact performance', help='helptext')
    psr.add_argument("-dataset_config", default="../configs/dataset_config/bcic_42a/bcic42a_base_prepost.yaml",
                     help="config containing datapath, x and label options. yaml format")
    psr.add_argument("-network_config", default="../configs/network_config/bcic_42a/bcic42a_mlstmfcn_test.yaml",
                     help="config containing network parameters. yaml format")
    psr.add_argument("-train_config", default="../configs/train_config/bcic_42a/bcic42a_mlstmfcn_train.yaml",
                     help="config containing general training parameters such as batch size. yaml format")
    psr.add_argument("-runset_config", default="../configs/runset_config/bcic_42a/bcic42a_allptc_default.yaml",
                     help="config containing runset config specifying which pipeline to run. yaml format")

    opt, _ = psr.parse_known_args()
    return opt

# process running arguments
runopt = get_cmdarg()

# process dataset config first
with open(runopt.dataset_config, 'r') as stream:
    try:
        dataset_config = yaml.load(stream, Loader=yaml.FullLoader)
        # print(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)
# network configs
with open(runopt.network_config, 'r') as stream:
    try:
        network_config = yaml.load(stream, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)
# train configs
with open(runopt.train_config, 'r') as stream:
    try:
        train_config = yaml.load(stream, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)
# runset configs (for slicing up participant/pipeline-wise jobs into different instances)
with open(runopt.runset_config, 'r') as stream:
    try:
        runset_config = yaml.load(stream, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)


# set device params etc (experiment specific, condition invariant)\
torch.cuda.empty_cache()
device = torch.device(train_config['device'] if torch.cuda.is_available() else "cpu")
random.seed(train_config['random_seed'])
torch.manual_seed(train_config['random_seed'])
np.random.seed(train_config['random_seed'])
torch.backends.cudnn.deterministic = train_config['determinism']
torch.backends.cudnn.benchmark = train_config['determinism_bench']

# set ouptut name strings, etc
# create output configs for each participant? (considering hyperparam iterates through indiv)
output_config_root_name =  f'{dataset_config["dataset_name"]}_{network_config["network_name"]}_' \
                           f'{train_config["config_name"]}'

# set some path related definitions
proj_root_path = '../'
output_root_path = proj_root_path+'output_files/'
config_root_path = proj_root_path+f'configs/run_output_config/{dataset_config["dataset_name"]}/'
config_proj_path = config_root_path+f'{output_config_root_name}/'
hp_output_path = output_root_path+f'hp_search_results/{dataset_config["dataset_name"]}/{output_config_root_name}/'
net_output_path = output_root_path+f'saved_weights/{dataset_config["dataset_name"]}/{output_config_root_name}/'

# strategy: create directory under root config name,
setup_directory(config_proj_path)
setup_directory(hp_output_path)
setup_directory(net_output_path)
# create root config  (later  create additional participant specific ones)
with open(f'{config_root_path}{output_config_root_name}.yaml', 'w') as f:
    if f.tell() == 0:

        # no such config exists, create one
        first_run_time = datetime.now()
        out_config = {'dataset_config': runopt.dataset_config,
                      'network_config': runopt.network_config,
                      'train_config': runopt.train_config,
                      'first_creation_date': first_run_time.strftime('%y%m%d_%H%M%S'),
                      'run_name': output_config_root_name,
                      'run_metric_collated': False,
                      'out_files_root': f'../output_files/raw_train_outputs/{output_config_root_name}/',
                      'participant_n': dataset_config['participants_in_data'],
                      'folds': train_config['folds']}

        yaml_out = yaml.dump(out_config, f)


# loop for each pipeline set (runset)
for runset in runset_config['runsets']: # runset name will be used to find data paths etc
    runset_name = output_config_root_name+runset
    hp_output_runset_path = hp_output_path+f'{runset}/'
    net_output_runset_path = net_output_path+f'{runset}/'
    setup_directory(hp_output_runset_path)
    setup_directory(net_output_runset_path)
    for sbjidx in np.arange(dataset_config['participants_in_data']): #for each participant

        # if we split runs by participant, we should check here:
        if runset_config['frag_training'] is True:
            if sbjidx < runset_config['frag_start']['subj']:
                continue
            if sbjidx > runset_config['frag_end']['subj']:
                break

        # with no problems let's continue
        sbjidx_nat = sbjidx+1  # since some file numberings are base 1 instead of 0

        # load data
        loaded_data = load_matfile(
            dataset_config['datapaths'][runset] + dataset_config['datafiles_prefix'] + str(sbjidx_nat) + '.mat')

        # preprocess data
        # check if train and test files are pre-split (we don't care about splits other than premade train/test splits yet)
        x_tv = loaded_data[dataset_config['x_train_varname']]
        y_tv = preproc_y(loaded_data[dataset_config['y_train_varname']])
        if dataset_config['train_test_is_split'] is True:
            x_test = loaded_data[dataset_config['x_test_varname']]
            y_test = preproc_y(loaded_data[dataset_config['y_test_varname']])

        # create container datastruct for participant here (will be exported in json later)

        # preprocess data here
        # data at this point is N x T x C in all datasets

        # if EOG channel needs to be excluded, then do it here
        if dataset_config['remove_eog'] is True:
            x_tv = np.delete(x_tv, dataset_config['eog_channels'], 2)
            if dataset_config['train_test_is_split'] is True:
                x_test = np.delete(x_test, dataset_config['eog_channels'], 2)

        # if pre and post onset split needs to occur, then do it here

        # fix it to split time instead
        if dataset_config['split_time']['argument'] is True:
            x_tv = x_tv[:,dataset_config['split_time']['start']:dataset_config['split_time']['end'],:]
            if dataset_config['train_test_is_split'] is True:
                x_test = x_test[:,dataset_config['split_time']['start']:dataset_config['split_time']['end'], :]



        # data fed into model depends on network architecture - NTC to NCT, N1CT etc...
        # this transformation should be done at the endof preprocessing steps that specifically mess
        # with channel/time to avoid confusion
        if network_config['data_dim_reorder']['necessity'] is True:
            x_tv = np.transpose(x_tv, network_config['data_dim_reorder']['dim'])
            if dataset_config['train_test_is_split'] is True:
                x_test = np.transpose(x_test, network_config['data_dim_reorder']['dim'])


        # trim dataset by labels if necessary (this is data dimension agnostic as long as N is first)
        # Todo

        # endof preprocessing data

        # split train, test, validation
        tt_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=train_config['random_seed'])
        if dataset_config['train_test_is_split'] is False: # Need to split x_test first
            for tv_idx, tt_idx in tt_split.split(x_tv, y_tv):
                x_trv, y_trv = x_tv[tv_idx], y_tv[tv_idx]
                x_test, y_test = x_tv[tt_idx], y_tv[tt_idx]
            x_tv = x_trv
            y_tv = y_trv
        # kf_split (train and validation) happens regardless
        kf_split = StratifiedShuffleSplit(n_splits=train_config['folds'], test_size=train_config['test_portion'],
                                          random_state=train_config['random_seed'])

        # set pytorch test-set dataset for skorch feeding
        # Todo
        ic_model = MLSTM_FCN_Ref(input_size=x_tv.shape[2], max_sequence_size=x_tv.shape[1],
                                 output_classn=dataset_config['output_classn'],
                                 lstm_hidden_size=8, lstm_num_layers=1, lstm_batch=True,
                                 conv2_size=256,
                                 conv3_size=128, conv1_kernel=8, conv2_kernel=5, conv3_kernel=3, lstm_dropout=0.65)


        # set up hyperparam search space
        search_space = network_config['param_space']
        search_space['module__input_size'] = [x_tv.shape[2]]
        search_space['module__max_sequence_size'] = [x_tv.shape[1]]
        search_space['module__output_classn'] = [dataset_config['output_classn']]
        #search_space['lr'] = train_config['learning_rate']
        # also add training hp search space
        search_space.update(train_config['trainhp_grid'])

        xentropy_weight = [np.where(y_tv[:]==xm)[0].shape[0] for xm in np.unique(y_tv)]
        tv_uniq_dist = xentropy_weight
        xentropy_weight = torch.tensor([1-xm/np.sum(xentropy_weight) for xm in xentropy_weight], dtype=torch.float32)
        # define skorch callbacks

        skorch_classifier = NeuralNetClassifier(
            module=ic_model,
            criterion=nn.CrossEntropyLoss(weight=xentropy_weight),
            optimizer=AdamW,
            max_epochs=train_config['epochs'],
            #train_split=tt_split.split,
            device=train_config['device'],
            #lr=train_config['learning_rate'],
            batch_size=train_config['train_batch'],
            callbacks=[('early_stopping', EarlyStopping(patience=train_config['early_stop_patience'])),
                       ('lr_scheduler', LRScheduler(CosineAnnealingWarmRestarts, T_0=train_config['lr_cosineAWR_T0'], T_mult=train_config['lr_cosineAWR_Tm'])
                        )
            ]

        )
        jam = GridSearchCV(estimator=skorch_classifier, param_grid=search_space, scoring='f1_micro',
                           n_jobs=train_config['CV_processes'],  cv=kf_split)

        X_tv = torch.from_numpy(x_tv).float()
        Y_tv = torch.from_numpy(y_tv).long()
        X_tt = torch.from_numpy(x_test).float()
        Y_tt = torch.from_numpy(y_test).long()

        jam.fit(X_tv, Y_tv)
        # get test set result
        ytt_pred = jam.best_estimator_.predict(X_tt)
        ytt_acc = accuracy_score(y_test, ytt_pred)
        ytt_f1 = f1_score(y_test, ytt_pred, average='micro')

        # save best performer model
        jam.best_estimator_.save_params(f_params=f'{net_output_runset_path}{network_config["network_name"]}_sbj{sbjidx_nat}.pkl')

        # prepare to export
        unique_classes = np.unique(y_test)
        tt_uniq_dist = [np.where(y_test[:] == xm)[0].shape[0] for xm in np.unique(y_test)]
        # sanitize cv result to be json compatible

        json_sbj = {'runset_name': runset_name,
                    'network_name': network_config['network_name'],
                    'sbjidx': int(sbjidx),
                    'sbjidx_nat': int(sbjidx_nat),
                    'uniq_classes': unique_classes.tolist(),
                    'test_class_dist': tt_uniq_dist,
                    'tv_class_dist': tv_uniq_dist, # use this for reinitializing classifier!
                    'tt_trials': y_test.shape[0],
                    'tv_trials': y_tv.shape[0],
                    'cv_result': jam.cv_results_,
                    'cv_best_idx': int(jam.best_index_),
                    'cv_best_param': jam.best_params_,
                    'final_epo': len(jam.best_estimator_.history),
                    'test_acc': ytt_acc,
                    'test_f1': ytt_f1,
                    'test_pred': ytt_pred.tolist(),
                    'test_true': y_test.tolist(),
                    'param_grid': jam.param_grid,
                    'last_val_loss': jam.best_estimator_.history[-1]['valid_loss'],
                    #'last_val_acc': jam.best_estimator_.history[-1]['valid_acc'],
                    'last_train_loss': jam.best_estimator_.history[-1]['train_loss'],
                    'best_model_path': f'{net_output_runset_path}{network_config["network_name"]}_sbj{sbjidx_nat}.pkl'
        }

        export_json(f'{hp_output_runset_path}sbj{sbjidx_nat}.json', json_sbj)





