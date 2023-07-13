import scipy.stats

from classification_scripts.cogsys_helper import *
from class_vis_stat_scripts.stat_agg_class import *
from class_vis_stat_scripts.plot_funcs import *
from class_vis_stat_scripts.func_sim_chance import simulate_chance

from sklearn.metrics import confusion_matrix

import os
import argparse
import yaml
import glob
import matplotlib
import pycircos
import statsmodels
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn import preprocessing
from scipy.stats import ttest_rel
import pandas as pd
from itertools import combinations
import seaborn as sns


matplotlib.use('Qt5Agg') # Pycharm crashes the plots otherwise!! Win64 issue possibly, may not be compatible with linux

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

stat_out_dir_root = '../output_files/result_pkls/'
fig_out_dir_root = '../output_files/plots/'

dataset_mets = []
dset_ptc_clevels = []
dsets_ptc_tn = []
nan_indices = []

bonferroni_value = 10

intermediate_path = './intermediate_results/'
dataset = stat_config['datasets'][0] # assuming we only do one dataset here
# load intermediate result
runset_net_mixmets_srt = load_pickle(f'{intermediate_path}{dataset["dsetname"]}')
# loaded variable is runset_net_mixmets_srt, sorted 15 runsets of each dataset\

# parts after this are customized for each plot

# run a rm anova
rmdfs = []
ica_rmanovs = []
ica_rmanovs_res = []
rmdf_corrs = []
ica_tukeys = []
rmdf_valtestcorrs = []
rmdfv_corrs = []
rmdfallv_corrs = []

ttest_combi = combinations([0,1,2,3,4], 2)
rmdf1_ttests_nets = []
rmdf1_ttests_nets_origins = []

rmdf_allvcorr_stack = []
rmdf_allf1corr_stack = []

rmdf_bestvals = []
rmdf_ttf1s = []

for networkidx, networkname in enumerate(stat_config['networks']):
    rmdf = [rmet.metrics['all_test_f1'] for rmet in runset_net_mixmets_srt[networkidx*5:(networkidx+1)*5]]
    rmdf_bestfval = [rmet.metrics['all_val_f1'] for rmet in runset_net_mixmets_srt[networkidx*5:(networkidx+1)*5]]

    rmdf_allval = [rmem.cv_result['mean_test_score'] for rmem in
                      [val for sublist in
                       [rmet.members for rmet in runset_net_mixmets_srt[networkidx * 5:(networkidx + 1) * 5]]
                       for val in sublist]
                      ]

    #rmdf_allval_matmat = np.asarray(rmdf_allval).reshape([5,runset_net_mixmets_srt[0].members_n, -1])
    rmdf_allval_matmat = np.asarray(rmdf_allval).reshape([5, -1])



    print(f'idx is {np.arange(networkidx*5,(networkidx+1)*5)}')
    rmdf_pc = np.asarray(rmdf)
    rmdf_vpc = np.asarray(rmdf_bestfval)
    rmdf = np.asarray(rmdf).reshape([-1])
    rmdf_bestfval = np.asarray(rmdf_bestfval).reshape([-1])
    rmdf_cond = np.asarray(stat_config['runsets']).repeat(runset_net_mixmets_srt[0].members_n)
    sbj_row = np.tile(np.arange(runset_net_mixmets_srt[0].members_n), 5)

    rmdf_bestvals.append(rmdf_bestfval)
    rmdf_ttf1s.append(rmdf)


    rmdf_pd = pd.DataFrame({'sbj_id': sbj_row, 'pipeline': rmdf_cond, 'test_f1': rmdf})
    ica_rmanov = AnovaRM(data=rmdf_pd, depvar='test_f1', subject='sbj_id',
                                                within=['pipeline',])
    ica_rmanov_res = ica_rmanov.fit()
    rmdf_corr = np.corrcoef(rmdf_pc)
    rmdf_testvalcorr = scipy.stats.pearsonr(x=rmdf, y=rmdf_bestfval)
    #rmdf_allvcorr = scipy.stats.pearsonr(x=rmdf_allval_matmat, y=rmdf_allval_matmat)
    rmdf_allvcorr = np.corrcoef(rmdf_allval_matmat)

    # if networkidx ==0:
    #     rmdf_allvcorr_stack=rmdf_allval_matmat.tolist()
    #     rmdf_allf1corr_stack=rmdf_pc.tolist()
    # else:
    #     rmdf_allvcorr_stack+=rmdf_allval_matmat.tolist()
    #     rmdf_allf1corr_stack+=rmdf_pc.tolist()
    rmdf_allvcorr_stack.append(rmdf_vpc)
    rmdf_allf1corr_stack.append(rmdf_pc)


    rmdf_corrs.append(rmdf_corr)
    rmdf_valtestcorrs.append(rmdf_testvalcorr)
    rmdfallv_corrs.append(rmdf_allvcorr)

    rmdfs.append(rmdf)
    ica_rmanovs.append(ica_rmanov)
    ica_rmanovs_res.append(ica_rmanov_res)
    print(ica_rmanov_res)

    rmdf_tukey = pairwise_tukeyhsd(endog=rmdf_pd['test_f1'], groups=rmdf_pd['pipeline'], alpha=0.05)
    ica_tukeys.append(rmdf_tukey)
    print(rmdf_tukey)

    rmdf_ttests = []
    rmdf_ttests_origin = []
    combos_list = []
    ttest_combi = combinations([0, 1, 2, 3, 4], 2)
    for ttestcomb in ttest_combi:
        rmdf_ttest = scipy.stats.ttest_rel(rmdf_pc[ttestcomb[0],:], rmdf_pc[ttestcomb[1],:])
        rmdf_ttest_ar = [rmdf_ttest.statistic,rmdf_ttest.pvalue]
        combos_list.append(ttestcomb)
        rmdf_ttests.append(rmdf_ttest_ar)
        rmdf_ttests_origin.append(rmdf_ttest)
    rmdf1_ttests_nets.append(rmdf_ttests)
    rmdf1_ttests_nets_origins.append(rmdf_ttests_origin)






#runsetmet_srt convention: [n1-p1,n1-p2,...n1-p5,n2-p1,...n2-p5,n3-p1,....n3-p5]
# pipeline sort pattern follows stat_config
#runset_net_mixmets_srt[]

n1_cmap = plt.cm.get_cmap('Greens')._resample(50)(np.linspace(0,1,50))[10:]
n2_cmap = plt.cm.get_cmap('PuRd')._resample(50)(np.linspace(0,1,50))[10:]
n3_cmap = plt.cm.get_cmap('Blues')._resample(50)(np.linspace(0,1,50))[10:]
n1_cmap = matplotlib.colors.ListedColormap(n1_cmap)
n2_cmap = matplotlib.colors.ListedColormap(n2_cmap)
n3_cmap = matplotlib.colors.ListedColormap(n3_cmap)
n_cmaps = [n1_cmap, n2_cmap, n3_cmap]

set3_ptc_cm = matplotlib.colormaps['tab10']._resample(10)
accent_ptc_cm = matplotlib.colormaps['Accent']._resample(8)

set3cm_colors = np.zeros((45,4))
set3cm_colors[:10, : ] = set3_ptc_cm(np.linspace(0,1,10))
set3cm_colors[10:18, :] = accent_ptc_cm(np.linspace(0,1,8))
set3cm_colors[:,3] = 0.35

set3cm_cm = matplotlib.colors.ListedColormap(set3cm_colors)



# plot 1 : group level bar(box) plots
# test f1 for each dataset, showing 5 pipelines bars with error per dataset
#
gbfig_bynet = plt.subplots(3,1) # 3 networks, so 3 ,1

boxsets_f1 = []
boxsets_x_f1 = []
boxplots_f1 = []

pipelines_size = len(stat_config['runsets'])

for networkidx, networkname in enumerate(stat_config['networks']):
    targ_ax =gbfig_bynet[1][networkidx]
    plt.axes(targ_ax)

    boxset = []
    boxset_x = []

    for runsetidx, runsetname in enumerate(stat_config['runsets']):
        runnetmet_idx = networkidx*pipelines_size+runsetidx
        boxset.append(runset_net_mixmets_srt[runnetmet_idx].metrics['all_test_f1'])
        boxset_x.append(networkidx*pipelines_size+runsetidx+1)
        boxplots_f1.append(plt.boxplot(boxset[-1], positions=[boxset_x[-1]], patch_artist=True))
        set_box_color_fillonly(boxplots_f1[-1], n_cmaps[networkidx].colors[runsetidx])

    targ_ax.set_ylabel('F1 score')
    targ_ax.set_title(stat_config['networks'][networkidx])
    #targ_ax.set_xlabel(stat_config['runsets'])

    targ_ax.set_xticklabels(stat_config['runsets'])
    #plt.xticks(rotation=45)
    targ_ax.set_ylim([0.3, 1])
    targ_ax.grid(visible=True, axis='y')
#gbfig_bynet[0].tight_layout()
# consideration: add tukey results/group level anova results by coloring them?
# repeated measure anova


# plot 2: same stuff as plot 1 but 3 network bars per pipeline instead



# plot 3: per network plot, each subplot for each parameter, scatter(?) showing where best params lie

pipe_cmap = plt.cm.get_cmap('tab10')._resample(10)(np.linspace(0,1,10))
#pipe_cmap = matplotlib.colors.ListedColormap(pipe_cmap)

netparamfigs = []
scatter_marker_types = ['o', 'v', 's', '*', 'D'] # by pipelines

for networkidx, networkname in enumerate(stat_config['networks']):
    paramtypes = stat_config['datasets'][0]['networks'][networkname]['search_params']
    paramtypes_count = len(paramtypes)
    if paramtypes_count < 6:
        #netparamfig = plt.subplots(paramtypes_count, 1)
        netparamfig = plt.subplots(3, 2)
    else:
        netparamfig = plt.subplots(4, 2)

    param_grid = runset_net_mixmets_srt[networkidx*5].members[0].param_grid
    net_bestparams = [rmem.cv_best_param for rmem in
                      [val for sublist in
                       [rmet.members for rmet in runset_net_mixmets_srt[networkidx * 5:(networkidx + 1) * 5]]
                       for val in sublist]
                      ]

    #plt.subplots_adjust(bottom=0.2, hspace=0.3)

    for paramidx, paramname in enumerate(paramtypes):
        idx_nd = np.unravel_index(paramidx, netparamfig[1].shape)
        targ_ax = netparamfig[1][idx_nd]
        plt.axes(targ_ax)

        param_possibles = sorted(param_grid[paramname])
        net_testf1 = [val for sublist in [rmet.metrics['all_test_f1'] for rmet in runset_net_mixmets_srt[networkidx * 5:(networkidx + 1) * 5]] for val in sublist]
        #parambag = {parami : {'f1': [], 'pipeline': []} for parami in param_possibles}
        parambag_l = []

        for netpipe_i, netpipe_indiv in enumerate(net_bestparams):
            #parambag[netpipe_indiv[paramname]]['f1'] = net_testf1[netpipe_i]
            #parambag[netpipe_indiv[paramname]]['pipeline'] = int(netpipe_i/runset_net_mixmets_srt[0].members_n)
            parambag_l.append({'f1':net_testf1[netpipe_i],
                               'pipelines':int(netpipe_i/runset_net_mixmets_srt[0].members_n),
                               'paramv_i': np.where(np.asarray(param_possibles)==netpipe_indiv[paramname])[0][0]})

            scatter_x = parambag_l[netpipe_i]['paramv_i']+np.random.uniform(-0.2,0.2)
            targ_ax.scatter(scatter_x, parambag_l[netpipe_i]['f1'],
                            color=pipe_cmap[parambag_l[netpipe_i]['pipelines'],:],
                            marker=scatter_marker_types[parambag_l[netpipe_i]['pipelines']],
                            edgecolors='black',
                            alpha=0.5)

        targ_ax.set_xlim([-1, len(param_possibles)+0.5])
        targ_ax.set_ylim([0.4, 0.9])
        targ_ax.spines['top'].set_visible(False)
        targ_ax.spines['left'].set_visible(False)
        targ_ax.spines['right'].set_visible(False)
        targ_ax.set_xticks(np.arange(len(param_possibles)), param_possibles)
        targ_ax.set_title(paramname[8:])
        targ_ax.set_ylabel('F1 scores')
        targ_ax.grid(visible=True, axis='y')

        # for paramv_i, paramv in enumerate(param_possibles):
        #     scatter_x = np.ones(len(parambag[paramv]['f1']))*(paramv_i+1)
        #
        #     targ_ax.scatter(scatter_x,parambag[paramv]['f1'], c='b', marker=scatter_marker_types[])





    boxset = []
    boxset_x = []
    netparamfig[0].delaxes(netparamfig[1][-1,-1])
    netparamfig[0].tight_layout()

# plot 4: cross-network correalation matrix of all val, and all test
rmdf_allvcorr_stack = np.asarray(rmdf_allvcorr_stack).reshape([15,-1])
rmdf_allf1corr_stack = np.asarray(rmdf_allf1corr_stack).reshape([15,-1])
rmdf_allvcorr_allnet = np.corrcoef(rmdf_allvcorr_stack)
rmdf_alltcorr_allnet = np.corrcoef(rmdf_allf1corr_stack)


corr_label_combi_bag = stat_config['networks']+stat_config['runsets']
corr_labels = ['Sha_', 'MLS_', 'EEG_']
pipe_labels = ['No', 'Run_MAR', 'Run_ICL', 'AMI_MAR', 'AMI_ICL']
#corr_combibag = combinations(corr_label_combi_bag, 2)
corr_combibag = itertools.product(corr_labels, pipe_labels)
corrcombi_labels = []
for corrcombi in corr_combibag:
    corrcombi_labels.append(corrcombi[0]+corrcombi[1])


vcorrfig = plt.subplots(1, 1) #plt.figure()
sns.heatmap(rmdf_allvcorr_allnet, cmap='Greens', annot=True, fmt='.3f')
vcorrfig[1].set_xticklabels(corrcombi_labels)
vcorrfig[1].set_yticklabels(corrcombi_labels)
plt.xticks(rotation=90)
plt.yticks(rotation=0)

tcorrfig = plt.subplots(1, 1) # plt.figure()
sns.heatmap(rmdf_alltcorr_allnet, cmap='Reds', annot=True, fmt='.3f')

tcorrfig[1].set_xticklabels(corrcombi_labels)
tcorrfig[1].set_yticklabels(corrcombi_labels)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

#plot 5: indiv test plots sorted by ptc and pipeline
netbars_fig = plt.subplots(3,1)
barw = 1

for networkidx, networkname in enumerate(stat_config['networks']):
    netbar_ax = netbars_fig[1][networkidx]
    barv1_x = np.arange(0,runset_net_mixmets_srt[0].members_n*(5*2+1),barw*5*2+1)
    bart1_x = np.arange(0.5,runset_net_mixmets_srt[0].members_n*(5*2+1)+0.5,barw*5*2+1)

    barv2_x = np.arange(2, runset_net_mixmets_srt[0].members_n * (5*2+1) +2, barw * 5*2+1)
    bart2_x = np.arange(2.5, runset_net_mixmets_srt[0].members_n * (5*2+1) + 2.5, barw * 5*2+1)

    barv3_x = np.arange(4, runset_net_mixmets_srt[0].members_n * (5*2+1) + 4, barw * 5*2+1)
    bart3_x = np.arange(4.5, runset_net_mixmets_srt[0].members_n * (5*2+1) + 4.5, barw * 5*2+1)

    barv4_x = np.arange(6, runset_net_mixmets_srt[0].members_n * (5*2+1) + 6, barw * 5*2+1)
    bart4_x = np.arange(6.5, runset_net_mixmets_srt[0].members_n * (5*2+1) + 6.5, barw * 5*2+1)

    barv5_x = np.arange(8, runset_net_mixmets_srt[0].members_n * (5*2+1) + 8, barw * 5*2+1)
    bart5_x = np.arange(8.5, runset_net_mixmets_srt[0].members_n * (5*2+1) + 8.5, barw * 5*2+1)

    ptcsep_lines = np.arange(9.5,runset_net_mixmets_srt[0].members_n*(5*2+1)+9.5, barw * 5*2+1)
    pipetick_lines = np.concatenate([barv1_x,barv2_x,barv3_x,barv4_x,barv5_x])
    pipetick_lines.sort()

    pipetick_labels = np.asarray(['NoICA', 'RunICA_MARA', 'RunICA_ICLABEL', 'AMICA_MARA', 'AMICA_ICLABEL'])
    pipetick_labels = np.tile(pipetick_labels,runset_net_mixmets_srt[0].members_n)



    rmdf_bestfval = rmdf_bestvals[networkidx]
    rmdf_test = rmdf_ttf1s[networkidx]

    barv1_y = rmdf_bestfval[0:runset_net_mixmets_srt[0].members_n]
    barv2_y = rmdf_bestfval[1*runset_net_mixmets_srt[0].members_n:2*runset_net_mixmets_srt[0].members_n]
    barv3_y = rmdf_bestfval[2*runset_net_mixmets_srt[0].members_n:3*runset_net_mixmets_srt[0].members_n]
    barv4_y = rmdf_bestfval[3*runset_net_mixmets_srt[0].members_n:4*runset_net_mixmets_srt[0].members_n]
    barv5_y = rmdf_bestfval[4*runset_net_mixmets_srt[0].members_n:5*runset_net_mixmets_srt[0].members_n]

    bart1_y = rmdf_test[0:runset_net_mixmets_srt[0].members_n]
    bart2_y = rmdf_test[1*runset_net_mixmets_srt[0].members_n:2*runset_net_mixmets_srt[0].members_n]
    bart3_y = rmdf_test[2*runset_net_mixmets_srt[0].members_n:3*runset_net_mixmets_srt[0].members_n]
    bart4_y = rmdf_test[3*runset_net_mixmets_srt[0].members_n:4*runset_net_mixmets_srt[0].members_n]
    bart5_y = rmdf_test[4*runset_net_mixmets_srt[0].members_n:5*runset_net_mixmets_srt[0].members_n]


    bpv1 = netbar_ax.bar(barv1_x, barv1_y, width=barw, color=set3cm_colors[0], edgecolor='k')
    bpv2 = netbar_ax.bar(barv2_x, barv2_y, width=barw, color=set3cm_colors[0], edgecolor='k')
    bpv3 = netbar_ax.bar(barv3_x, barv3_y, width=barw, color=set3cm_colors[0], edgecolor='k')
    bpv4 = netbar_ax.bar(barv4_x, barv4_y, width=barw, color=set3cm_colors[0], edgecolor='k')
    bpv5 = netbar_ax.bar(barv5_x, barv5_y, width=barw, color=set3cm_colors[0], edgecolor='k')

    bpt1 = netbar_ax.bar(bart1_x, bart1_y, width=barw, color=set3cm_colors[1], edgecolor='k')
    bpt2 = netbar_ax.bar(bart2_x, bart2_y, width=barw, color=set3cm_colors[1], edgecolor='k')
    bpt3 = netbar_ax.bar(bart3_x, bart3_y, width=barw, color=set3cm_colors[1], edgecolor='k')
    bpt4 = netbar_ax.bar(bart4_x, bart4_y, width=barw, color=set3cm_colors[1], edgecolor='k')
    bpt5 = netbar_ax.bar(bart5_x, bart5_y, width=barw, color=set3cm_colors[1], edgecolor='k')

    # use vertical bars instead of minor tick for usage in ptc labeling
    netbar_ax.vlines(x=ptcsep_lines, ymin=0, ymax=1, colors='gray', ls='--', lw=1)
    #netbar_ax.set_xticks(ptcsep_lines, minor=True)

    netbar_ax.set_ylim([0.3, 1])
    netbar_ax.set_xlim([-1, runset_net_mixmets_srt[0].members_n*(5*2+1)-1.5])
    # netbar_ax.set(xticks=pipetick_lines,
    #             xticklabels=pipetick_labels)
    netbar_ax.set_xticks(pipetick_lines)
    netbar_ax.set_xticklabels(pipetick_labels, rotation=90)
    #netbar_ax.xticks(rotation=90)
    netbar_ax.set_xticks(barv3_x, minor=True) # for ptc labeling
    netbar_ax.set_ylabel('F1-Score')
    netbar_ax.set_title(f'{networkname}')
    netbar_ax.legend([bpv1, bpt1], ['Best Validation F1', 'Test F1'], ncol=2, loc='lower right')
    #netbar_ax.grid(which='minor')


# part 4? printout: latex format for all test f1s
for networkidx, networkname in enumerate(stat_config['networks']):
    net_testf1 = [val for sublist in
                  [rmet.metrics['all_test_f1'] for rmet in runset_net_mixmets_srt[networkidx * 5:(networkidx + 1) * 5]]
                  for val in sublist]
    net_finalepo = [val for sublist in
                    [rmet.metrics['all_final_epo'] for rmet in runset_net_mixmets_srt[networkidx * 5:(networkidx + 1) * 5]]
                        for val in sublist]
    net_bestvalf1 = [val for sublist in
                        [rmet.metrics['all_val_f1'] for rmet in
                        runset_net_mixmets_srt[networkidx * 5:(networkidx + 1) * 5]]
                        for val in sublist]
    net_simchance_high = [val for sublist in
                    [rmet.metrics['all_simchance_cihigh'] for rmet in
                     runset_net_mixmets_srt[networkidx * 5:(networkidx + 1) * 5]]
                    for val in sublist]

    ####
    print(f"testf1_table - {networkname}")
    tab_strs = f''
    for sbji in np.arange(runset_net_mixmets_srt[0].members_n):
        tab_str = f'{sbji+1}&'

        for pipei in np.arange(5):
            #if pipei < 4:
            tab_str = tab_str + f'${net_testf1[sbji+runset_net_mixmets_srt[0].members_n*(pipei)]:.3f}$&'
            # else:
            #     tab_str = tab_str + f'${net_testf1[sbji + runset_net_mixmets_srt[0].members_n * (pipei)]:.3f}$\\\\'
        sbji_values = net_testf1[sbji:sbji+runset_net_mixmets_srt[0].members_n*4:runset_net_mixmets_srt[0].members_n]
        tab_str = tab_str + f'${np.mean(sbji_values):.3f}(\pm{np.std(sbji_values):.3f})$\\\\ \hline'
        #print(f'{sbji+1}\&{net_testf1[sbji:sbji+4*runset_net_mixmets_srt[0].members_n]}')
        print(tab_str)

    tab_str = f'\\textbf{{avg.(SD)}}&'
    for pipei in np.arange(len(stat_config['runsets'])):
        pipe_values = net_testf1[pipei*runset_net_mixmets_srt[0].members_n:(pipei+1)*runset_net_mixmets_srt[0].members_n]
        tab_str = tab_str + f'$\\pmb{{{np.mean(pipe_values):.3f}(\pm{np.std(pipe_values):.3f})}}$&'


    tab_str = tab_str + f'$\\pmb{{{np.mean(net_testf1):.3f}({np.std(net_testf1):.3f})}}$\\\\ \\hline'
    print(tab_str)
    #####


    ####
    print(f"finalepo - {networkname}")
    tab_strs = f''
    for sbji in np.arange(runset_net_mixmets_srt[0].members_n):
        tab_str = f'{sbji+1}&'

        for pipei in np.arange(5):
            #if pipei < 4:
            tab_str = tab_str + f'${net_finalepo[sbji+runset_net_mixmets_srt[0].members_n*(pipei)]}$&'
            # else:
            #     tab_str = tab_str + f'${net_testf1[sbji + runset_net_mixmets_srt[0].members_n * (pipei)]:.3f}$\\\\'
        sbji_values = net_finalepo[sbji:sbji+runset_net_mixmets_srt[0].members_n*4:runset_net_mixmets_srt[0].members_n]
        tab_str = tab_str + f'${np.mean(sbji_values):.1f}(\pm{np.std(sbji_values):.1f})$\\\\ \hline'
        #print(f'{sbji+1}\&{net_testf1[sbji:sbji+4*runset_net_mixmets_srt[0].members_n]}')
        print(tab_str)

    tab_str = f'\\textbf{{avg.(SD)}}&'
    for pipei in np.arange(len(stat_config['runsets'])):
        pipe_values = net_finalepo[pipei*runset_net_mixmets_srt[0].members_n:(pipei+1)*runset_net_mixmets_srt[0].members_n]
        tab_str = tab_str + f'$\\pmb{{{np.mean(pipe_values):.1f}(\pm{np.std(pipe_values):.1f})}}$&'


    tab_str = tab_str + f'$\\pmb{{{np.mean(net_finalepo):.1f}({np.std(net_finalepo):.1f})}}$\\\\ \\hline'
    print(tab_str)
    #####

    ####
    print(f"bestvalf1_table - {networkname}")
    tab_strs = f''
    for sbji in np.arange(runset_net_mixmets_srt[0].members_n):
        tab_str = f'{sbji+1}&'

        for pipei in np.arange(5):
            #if pipei < 4:
            tab_str = tab_str + f'${net_bestvalf1[sbji+runset_net_mixmets_srt[0].members_n*(pipei)]:.3f}$&'
            # else:
            #     tab_str = tab_str + f'${net_testf1[sbji + runset_net_mixmets_srt[0].members_n * (pipei)]:.3f}$\\\\'
        sbji_values = net_bestvalf1[sbji:sbji+runset_net_mixmets_srt[0].members_n*4:runset_net_mixmets_srt[0].members_n]
        tab_str = tab_str + f'${np.mean(sbji_values):.3f}(\pm{np.std(sbji_values):.3f})$\\\\ \hline'
        #print(f'{sbji+1}\&{net_testf1[sbji:sbji+4*runset_net_mixmets_srt[0].members_n]}')
        print(tab_str)

    tab_str = f'\\textbf{{avg.(SD)}}&'
    for pipei in np.arange(len(stat_config['runsets'])):
        pipe_values = net_bestvalf1[pipei*runset_net_mixmets_srt[0].members_n:(pipei+1)*runset_net_mixmets_srt[0].members_n]
        tab_str = tab_str + f'$\\pmb{{{np.mean(pipe_values):.3f}(\pm{np.std(pipe_values):.3f})}}$&'


    tab_str = tab_str + f'$\\pmb{{{np.mean(net_bestvalf1):.3f}({np.std(net_bestvalf1):.3f})}}$\\\\ \\hline'
    print(tab_str)
    #####

# part 4-2 print Test-F1 correlations

for networkidx, networkname in enumerate(stat_config['networks']):
    print(f"testF1 Corr - {networkname}")
    tab_strs = ' '
    for pipei, pipename in enumerate(stat_config['runsets']):
        rowstr = [f'{tbst:.3f}&' for tbst in rmdf_corrs[networkidx][pipei,:].tolist()]
        rowstr_s = ''.join(rowstr)
        tab_strs = tab_strs + rowstr_s
        tab_strs = tab_strs + '\\\\ \\hline'

    print(tab_strs)

print('posthoc tttest rows')
for tbrow in np.arange(10):
    tb_hypothesis = [rmdf1_ttests_nets[0][tbrow][1]<0.005,rmdf1_ttests_nets[1][tbrow][1]<0.005, rmdf1_ttests_nets[2][tbrow][1]<0.005]
    tabpost_strs = f'{rmdf1_ttests_nets[0][tbrow][0]:.3f}&{rmdf1_ttests_nets[0][tbrow][1]:.3f}&{tb_hypothesis[0]}&' \
               f'{rmdf1_ttests_nets[1][tbrow][0]:.3f}&{rmdf1_ttests_nets[1][tbrow][1]:.3f}&{tb_hypothesis[1]}&' \
               f'{rmdf1_ttests_nets[2][tbrow][0]:.3f}&{rmdf1_ttests_nets[2][tbrow][1]:.3f}&{tb_hypothesis[2]}\\\\'
    print(tabpost_strs)



#plt.rcParams['figure.constrained_layout.use'] = True
plt.show(block=False)