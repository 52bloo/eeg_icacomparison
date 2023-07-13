from classification_scripts.cogsys_helper import *
from class_vis_stat_scripts.stat_agg_class import *
from class_vis_stat_scripts.plot_funcs import *

from sklearn.metrics import confusion_matrix

import os
import argparse
import yaml
import glob
import matplotlib
matplotlib.use('Qt5Agg') # Pycharm crashes the plots otherwise!! Win64 issue possibly

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
    psr.add_argument("-stat_config", default="../../configs/stat_config/bcic_42a/bcic42a_lrsearch.yaml",
                     help="filepath for lv1 statistics config")
    opt, _ = psr.parse_known_args()
    return opt


# process running arguments
runopt = get_cmdarg()

# open res config file
with open(runopt.stat_config, 'r') as stream:
    try:
        stat_config = yaml.load(stream, Loader=yaml.FullLoader)
        # print(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

# stat config file lists runset & json output path for each runset, as well as reference to the
# output config (which helps tracking down the respective dataset, network, train config,
# but not in finding json output paths because we changed the path pattern in the middle of writing code)
# the goal is to parse json files to create easier data structs for visualization and statistics

#run_datasets = stat_config['datasets']
stat_out_dir_root = '../output_files/result_pkls/'
fig_out_dir_root = '../output_files/plots/'

for runset in stat_config['runsets']: # noica, amica_iclabel, etc..
    for dataset in stat_config['datasets']:
        for network in stat_config['networks']:
            # open json filepath for each of these setups
            with open(dataset['networks'][network]['output_config'], 'r') as stream:
                try:
                    output_config = yaml.load(stream, Loader=yaml.FullLoader)
                    # print(yaml.safe_load(stream))
                except yaml.YAMLError as exc:
                    print(exc)

            dataset_config = load_yml(output_config['dataset_config'])
            network_config = load_yml(output_config['network_config'])
            train_config = load_yml(output_config['train_config'])

            # create a list of sbj files
            sbj_filelist = glob.glob(f'{dataset["networks"][network]["runset_out_path"]}{runset}/*.json')

            run_metric = MetricAggregation(container_type='run')
            run_metric.add_container_info({
                'runset': runset,
                'dataset_config': dataset_config,
                'network_config': network_config,
                'train_config': train_config,
                'run_name': output_config['run_name']

            })

            for sbjidx in np.arange(output_config['participant_n']):

                # on this version fold wise addition is not needed; you only get representative stats per participant

                sbj_json = load_metricjson(sbj_filelist[sbjidx])
                sbj_metric = MetricContainer(container_type='individual')
                sbj_metric.add_metrics({
                    'test_acc': sbj_json['test_acc'],
                    'test_f1': sbj_json['test_f1'],
                })

                sbj_metric.add_container_info({
                    'sbjidx':sbj_json['sbjidx'],
                    'sbjidx_nat':sbj_json['sbjidx_nat'],
                    'train_n': sbj_json['tv_trials'],
                    'test_n': sbj_json['tt_trials'],
                    #'xentropy_weight': sbj_json['xentropy_weight']
                })
                sbj_metric.set_cv_result(sbj_json['cv_result'])
                sbj_metric.set_paramsets(sbj_json['param_grid'], sbj_json['cv_best_param'])

                run_metric.add_members(sbj_metric)
            print('preparing stats')
            plt.rcParams['figure.figsize'] = (6.2, 8.5)
            lr_statbag = per_parameter_statistics(run_metric, target_param='lr')
            earlystop_statbag = per_parameter_statistics(run_metric, target_param='callbacks__early_stopping__patience')
            lr_tm_statbag = per_parameter_statistics(run_metric, target_param='callbacks__lr_scheduler__T_mult')
            lr_t0_statbag = per_parameter_statistics(run_metric, target_param='callbacks__lr_scheduler__T_0')
            print('plotting')
            per_paramset_sbj_boxplot_double(lr_statbag, plot_title=f"{run_metric.container_info['network_config']['network_name'][:-10]}, LR",
                                            save=False, savepath=f"{fig_out_dir_root}stat_lr_{run_metric.container_info['network_config']['network_name']}")

            per_paramset_sbj_boxplot_double(earlystop_statbag, plot_title=f"{run_metric.container_info['network_config']['network_name'][:-10]}, Early_Patience",
                                            save=False, savepath=f"{fig_out_dir_root}stat_ep_{run_metric.container_info['network_config']['network_name']}")
            per_paramset_sbj_boxplot_double(lr_tm_statbag,
                                            plot_title=f"{run_metric.container_info['network_config']['network_name'][:-10]}, LR_Cosine_T_Mult",
                                            save=False, savepath=f"{fig_out_dir_root}stat_tm_{run_metric.container_info['network_config']['network_name']}")
            per_paramset_sbj_boxplot_double(lr_t0_statbag,
                                            plot_title=f"{run_metric.container_info['network_config']['network_name'][:-10]}, LR_Cosine_T_0",
                                            save=False, savepath=f"{fig_out_dir_root}stat_t0_{run_metric.container_info['network_config']['network_name']}")

#plt.pause()
plt.show(block=False)
print('ha')