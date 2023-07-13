from class_vis_stat_scripts.stat_agg_class import *

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import stats
from numpy.ma import masked_array
import itertools


def find_nan_cvaccs(cvresult_dict, cv_folds, chance_acc):
    nan_indices = []
    fold_splitts = []
    for i in range(cv_folds):
        nan_index = np.argwhere(np.isnan(cvresult_dict[f'split{i}_test_score'])).flatten()
        if nan_index.size>0:
            nan_indices.append(nan_index)
            cvres_ndr = np.array(cvresult_dict[f'split{i}_test_score'])
            cvres_ndr[nan_index]=chance_acc #overwrite to chance level
            cvresult_dict[f'split{i}_test_score'] = cvres_ndr.tolist()
        fold_splitts.append(cvresult_dict[f'split{i}_test_score'])

    if len(nan_indices)>0:
        cvresult_dict['mean_test_score'] = np.mean(np.array(fold_splitts), axis=0).tolist()

    return cvresult_dict, nan_indices


def set_box_color(bp, color):
    """
    function to set boxplot colors
    :param bp:
    :param color:
    :return:
    """
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def set_box_color_notmedian(bp, color):
    """
    function to set boxplot colors
    :param bp:
    :param color:
    :return:
    """
    plt.setp(bp['boxes'], color=color)
    bp['boxes'][0].set_facecolor(color)
    #bp.set(facecolor=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
#    plt.setp(bp['medians'], color=color)

def set_box_color_fillonly(bp, color):
    """
    function to set boxplot colors
    :param bp:
    :param color:
    :return:
    """
    #plt.setp(bp['boxes'], color=color)
    bp['boxes'][0].set_facecolor(color)
    #bp.set(facecolor=color)
    #plt.setp(bp['whiskers'], color=color)
    #plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color='k')


# bare functionality
def per_paramset_sbj_boxplot(per_param_stat):

    participant_n = 9
    uniq_param_v = per_param_stat['param_values']
    param_scores = per_param_stat['param_scores']

    plot_cmap = plt.cm.get_cmap("Set1", len(uniq_param_v))

    boxsets = []
    boxsets_x = []
    boxplots = []

    plt.figure()

    # iterate per param value, and then iterate per sbj
    for pridx, prval in enumerate(uniq_param_v):
        boxset = []
        boxset_x = []

        for sbjidx, sbjval in enumerate(param_scores):
            # param scores is ptc in first level and then param v in deeper level
            boxset.append(param_scores[sbjidx][pridx]) # each boxset is all subjects' result values based on a single param value
            boxset_x.append(pridx*(participant_n+1)+sbjidx+1)
            boxplots.append(plt.boxplot(boxset, positions=boxset_x))
            set_box_color(boxplots[-1], plot_cmap.colors[pridx])
        boxsets.append(boxset)
        boxsets_x.append(boxset_x)

    return boxsets, boxsets_x

def per_paramset_sbj_boxplot_double(per_param_stat, save=False, saveformat='img', savepath=None, box_cmap='Set1',
                                    plot_title=None, ):
    '''
    "Complex" version to handle plot titling, subplotting in order to serve two versions for better analysis view etc
    :param per_param_stat:
    :param save:
    :param saveformat:
    :param savepath:
    :param box_cmap:
    :param plot_title:
    :return:
    '''

    participant_n = 9
    uniq_param_v = per_param_stat['param_values']
    param_scores = per_param_stat['param_scores']

    plot_cmap_f1 = plt.cm.get_cmap("Set1", len(uniq_param_v))


    boxsets_f1 = []
    boxsets_x_f1 = []
    boxplots_f1 = []

    #plt.figure()

    xfig = plt.subplots(2,1)
    plt.axes(xfig[1][0])


    # iterate per param value, and then iterate per sbj
    for pridx, prval in enumerate(uniq_param_v):
        boxset = []
        boxset_x = []

        for sbjidx, sbjval in enumerate(param_scores):
            # param scores is ptc in first level and then param v in deeper level
            boxset.append(param_scores[sbjidx][pridx]) # each boxset is all subjects' result values based on a single param value
            boxset_x.append(pridx*(participant_n+1)+sbjidx+1)
            boxplots_f1.append(plt.boxplot(boxset[-1], positions=[boxset_x[-1]]))
            set_box_color(boxplots_f1[-1], plot_cmap_f1.colors[pridx])
        boxsets_f1.append(boxset)
        boxsets_x_f1.append(boxset_x)
        #xlabel_bag = [str(et.container_info['n_after_bal']) for et in repmet.members]
        #xlabel_pos = np.arange(repmet.members_n) * (len(target_metrics) + 1) + np.ceil(len(target_metrics) / 2)
        if pridx%2==1: # shade by group
            plt.axvspan(boxset_x[0]-0.5, boxset_x[-1]+0.5, facecolor=(0.8,0.8,0.8), alpha=0.15)

    # set ax1 title: grouped by param value
    xfig[1][0].set_ylabel('Accuracy')
    xfig[1][0].set_title(f'{plot_title} (grouped by Param value)', fontweight='bold')
    xfig[1][0].set_xlabel('Ptc.num')
    xfig[1][0].grid(visible=True, axis='y')
    xfig[1][0].set_xticks(list(itertools.chain(*boxsets_x_f1)), boxsets_x_f1[0]*len(boxsets_x_f1))
    legend_plot_idx = np.arange(0,len(uniq_param_v)*participant_n,participant_n+1).tolist()
    xfig[1][0].legend([boxplots_f1[f1]["boxes"][0] for f1 in legend_plot_idx], uniq_param_v,
                      loc='upper left', ncol=len(uniq_param_v))
    plt.xticks(rotation=45)
    plt.ylim([0.1, 1])



    ##############################
    # add grouped-by within participant plots of the same thing to show
    plt.axes(xfig[1][1])
    # ptcwise plotting area
    plot_cmap_f2 = plt.cm.get_cmap("Set1", len(param_scores))
    boxsets_f2 = []
    boxsets_x_f2 = []
    boxplots_f2 = []

    for sbjidx, sbjval in enumerate(param_scores):
        boxset = []
        boxset_x = []
        for pridx, prval in enumerate(uniq_param_v):
            boxset.append(param_scores[sbjidx][pridx])
            boxset_x.append(sbjidx*(len(uniq_param_v)+1)+pridx+1)
            boxplots_f2.append(plt.boxplot(boxset[-1], positions=[boxset_x[-1]]))
            set_box_color(boxplots_f2[-1], plot_cmap_f2.colors[pridx])
        boxsets_f2.append(boxset)
        boxsets_x_f2.append(boxset_x)
        if sbjidx%2==1: # shade by group
            plt.axvspan(boxset_x[0]-0.5, boxset_x[-1]+0.5, facecolor=(0.8,0.8,0.8), alpha=0.15)

    # set title for second plot
    xfig[1][1].set_title(f'{plot_title} (grouped by participant)', fontweight='bold')
    xfig[1][1].set_ylabel('Accuracy')
    #xfig[1][1].set_xlabel(f'{per_param_stat["param_name"]}')
    xfig[1][1].set_xlabel(f'Ptc.num')
    #xfig[1][1].set_xticks(boxsets_x_f2[0], uniq_param_v)

    #list(itertools.chain(*boxsets_x_f1)), boxsets_x_f1[0] * len(boxsets_x_f1)
    plt.xticks(rotation=45)
    plt.ylim([0.1, 1])
    legend_plot_idx = np.arange(0,len(uniq_param_v)*participant_n,len(uniq_param_v)+1).tolist()
    tick_plot_idx = np.arange(0, len(uniq_param_v) * participant_n, len(uniq_param_v)).tolist()
    xfig[1][1].set_xticks([list(itertools.chain(*boxsets_x_f2))[f1] for f1 in tick_plot_idx], np.arange(1,10))
    xfig[1][1].legend([boxplots_f2[f1]["boxes"][0] for f1 in legend_plot_idx], uniq_param_v,
                      loc='upper left', ncol=len(uniq_param_v))
    xfig[1][1].grid(visible=True, axis='y')
    xfig[0].subplots_adjust(hspace=.2)

    if savepath is not None and save is True:
        plt.savefig(fname=f"{savepath}.png", format='png')

    return boxsets_f1, boxsets_x_f1, boxsets_f2, boxsets_x_f2, xfig




