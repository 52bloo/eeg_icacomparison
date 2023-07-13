import matplotlib.pyplot as plt

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
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
from sklearn import preprocessing
import re
from matplotlib.collections import PatchCollection

matplotlib.use('Qt5Agg') # Pycharm crashes the plots otherwise!! Win64 issue possibly, may not be compatible with linux

def get_fname_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    filename_num = re.match(r"([a-z]+)([0-9]+)", filename, re.I).groups()[1]

    return int(filename_num)




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
# plotting after everything is aggregated (dataset-wise plotting)


'''
Circos-Section below
'''

# circos-plot
# 1. multi-model, single dataset, ptc-individual, separated by pipeline
# outermost : subject test-f1 by chance level (each column becomes subject+model)
# layer2 : barplots of subject test-f1
# layer3 : barplots of subject val-acc/f1 (all 5 folds)
# layer4 :
# innermost : chord plot of inter-pipeline subject anova/tukey results
# 2. single-model, single dataset, ptc-individual, separated by pipeline
# outermost :


#ptc_colormap =
pipe_colormap = plt.cm.get_cmap('tab20')
pipe_heat_colormap = ["Purples"]
n1_cmap = plt.cm.get_cmap('Greens')._resample(50)(np.linspace(0,1,50))[10:]
n2_cmap = plt.cm.get_cmap('PuRd')._resample(50)(np.linspace(0,1,50))[10:]
n3_cmap = plt.cm.get_cmap('Greys')._resample(50)(np.linspace(0,1,50))[10:]
n1_cmap = matplotlib.colors.ListedColormap(n1_cmap)
n2_cmap = matplotlib.colors.ListedColormap(n2_cmap)
n3_cmap = matplotlib.colors.ListedColormap(n3_cmap)
n_cmaps = [n1_cmap, n2_cmap, n3_cmap]
n_mixmap = np.concatenate([n1_cmap.colors[:10:2,:], n2_cmap.colors[:10:2,:], n3_cmap.colors[:10:2,:]])


# colormap bases to be used for plotting later
red_cm = matplotlib.colormaps['Reds']._resample(200)
purple_cm = matplotlib.colormaps['Blues']._resample(200)
set3_ptc_cm = matplotlib.colormaps['Set3']._resample(12)
accent_ptc_cm = matplotlib.colormaps['Accent']._resample(8)

redcm_colors = red_cm(np.linspace(0, 1, 200))
purplecm_colors = purple_cm(np.linspace(0, 1, 200))

set3cm_colors = np.zeros((45,4))
set3cm_colors[:12, : ] = set3_ptc_cm(np.linspace(0,1,12))
set3cm_colors[12:20, :] = accent_ptc_cm(np.linspace(0,1,8))

set3cm_cm = matplotlib.colors.ListedColormap(set3cm_colors)

cc_pink_orangecont = np.array([0.12156863, 0.46666667, 0.70588235, 1])

cc_pink = np.array([248/256, 24/256, 148/256, 1])
cc_gray = np.array([0.166, 0.166, 0.166, 1])
cc_gray_l = np.array([0.766, 0.766, 0.766, 1])

cc_allgray = np.zeros((45,4))
cc_allgray[0:,:] = cc_gray_l

ccp_red = redcm_colors
ccp_purple = purplecm_colors


#ccp_red[:50, :] = cc_gray
#ccpink_cm = matplotlib.colors.ListedColormap(cc_pink)
ccpink_cm = matplotlib.colors.ListedColormap(cc_pink)
red_chance_cm = matplotlib.colors.ListedColormap(ccp_red)
purple_chance_cm = matplotlib.colors.ListedColormap(ccp_purple)
allgray_cm = matplotlib.colors.ListedColormap(cc_allgray)


#circofig = plt.figure()
cg1_arc = pycircos.Garc()
cg1_circ = pycircos.Gcircle()

# run for loops to add different runset groups as arc
# for runsetsidx, runset in enumerate(runset_mets):
#     run_arc = pycircos.Garc(arc_id=runset[0].container_info['runset'], label_visible=True) # size is optional
#     cg1_circ.add_garc(run_arc)



for runsetsidx, runset in enumerate(runset_net_mixmets_srt):
    run_arc = pycircos.Garc(arc_id=f'{runset.container_info["runset"]}_{runset.container_info["network_config"]["network_type"]}',
                            label_visible=True, raxis_range=(950, 1000), labelsize=8,
                            facecolor=n_mixmap[runsetsidx])  # size is optional
    cg1_circ.add_garc(run_arc)

    # start with barplot
cg1_circ.set_garcs()


# for runsetsidx, runset in enumerate(runset_mets):
#     # collect values first
#     testf1_data = runset[0].collect_metric_values_for_plotting('test_f1')
#     gridvaldata = runset[0].collect_cvres_value_for_plotting('mean_test_score')
#     gridvaldata = np.array(gridvaldata)
#     cg1_circ.barplot(runset[0].container_info['runset'], data=testf1_data, raxis_range=(500, 700))
#     heat_min, heat_max = np.min(gridvaldata[:]), np.max(gridvaldata[:])
#     heat_min, heat_max = np.min(gridvaldata[0]), np.max(gridvaldata[0])
#     plot_cmap = plt.cm.get_cmap("Purples", gridvaldata[0].shape[0])
#     #cg1_circ.heatmap(runset[0].container_info['runset'], data=gridvaldata, cmap=plt.cm.viridis, vmin=heat_min, vmax=heat_max)
#     cg1_circ.heatmap(runset[0].container_info['runset'], data=gridvaldata[0], cmap=plot_cmap, raxis_range=(200,300), vmin=heat_min, vmax=heat_max, linewidth=0.03)
#


# perform significance for each participant, for each network
nmc_results = []
anov_results = []
all_pvalues = []
icasbj_tukeys = []



for networkidx, networkset in enumerate(stat_config['networks']):
    for sbjidx in np.arange(runset_net_mixmets_srt[0].members_n):
        #target_data = [x.members[sbjidx].cv_result['mean_test_score'] for x in runset_net_mixmets_srt[networkidx::len(stat_config['networks'])]]
        target_data = [x.members[sbjidx].cv_result['mean_test_score'] for x in
                       runset_net_mixmets_srt[networkidx*len(stat_config['runsets']):(networkidx+1)*len(stat_config['runsets'])]]
        #.collect_cvres_value_for_plotting('mean_test_score')
        nmc_result_bag = []
        for runsetidx in np.arange(len(target_data)):
            # test for distribution
            nmc_stat, nmc_p = scipy.stats.normaltest(target_data[runsetidx])
            #nmc_results.append([nmc_stat, nmc_p])
            nmc_result_bag.append([nmc_stat, nmc_p])
        nmc_results.append(nmc_result_bag)
        # anova
        #
        anov = stats.f_oneway(*target_data)
        # do multiple comparsion correction here
        # as well as setting zero pvalue into something slightly larger
        anov._replace(pvalue = anov.pvalue*bonferroni_value)#*= bonferroni_value
        if anov.pvalue > 1:
            anov._replace(pvalue = 1) #pvalue = 1
        if anov.pvalue == 0:
            anov._replace(pvalue = 10 ** -100)#pvalue = 10 ** -100
        anov_results.append(anov)
        all_pvalues.append(anov.pvalue)
        rmdf_pc = np.asarray(target_data)
        rmdf = np.asarray(target_data).reshape([-1])
        rmdf_cond = np.asarray(stat_config['runsets']).repeat(rmdf_pc.shape[1])
        sbj_row = np.tile(np.arange(runset_net_mixmets_srt[0].members_n), 5)

        rmdf_pd = pd.DataFrame({'pipeline': rmdf_cond, 'test_f1': rmdf})
        rmdf_tukey = pairwise_tukeyhsd(endog=rmdf_pd['test_f1'], groups=rmdf_pd['pipeline'], alpha=0.05)
        icasbj_tukeys.append(rmdf_tukey)

        # also run posthoc here
            #nmc_stat, nmc_p = scipy.stats.normaltest(target_data)



for runsetsidx, runset in enumerate(runset_net_mixmets_srt):
    # collect values first
    arc_id = f'{runset.container_info["runset"]}_{runset.container_info["network_config"]["network_type"]}'
    testf1_data = runset.collect_metric_values_for_plotting('test_f1')
    bestvalf1_data = runset.collect_metric_values_for_plotting('val_f1')

    gridvaldata = runset.collect_cvres_value_for_plotting('mean_test_score')
    gridvaldata = np.array(gridvaldata)

    # bar plot for test f1 accuracy
    cg1_circ.barplot(arc_id, data=testf1_data, raxis_range=(850, 950), facecolor=cg1_circ._garc_dict[arc_id].facecolor,
                     rlim=(0.3,1), linewidth=0.01)
    #heat_min, heat_max = np.min(gridvaldata[:]), np.max(gridvaldata[:])
    heat_min, heat_max = 0, 1
    #heat_min, heat_max = np.min(gridvaldata[0]), np.max(gridvaldata[0])
    plot_cmap_val = plt.cm.get_cmap("Purples", 200)
    plot_cmap_test = plt.cm.get_cmap("Reds", 200)
    #cg1_circ.heatmap(runset[0].container_info['runset'], data=gridvaldata, cmap=plt.cm.viridis, vmin=heat_min, vmax=heat_max)
    param_raxis_start = 450
    param_raxis_step = 400/gridvaldata.shape[1]

    sbj_xpos = np.asarray([0]).astype(float);
    sbjx_stepsize = cg1_circ._garc_dict[arc_id].size / runset_net_mixmets_srt[0].members_n
    for sbjidx in np.arange(runset_net_mixmets_srt[0].members_n):
        param_raxis_start = 450
        ptch_min = np.min(gridvaldata[sbjidx, :])
        ptch_max = np.max(gridvaldata[sbjidx, :])
        ptch_min = 0
        ptch_max = 1

        ccp_purpleptc = purplecm_colors
        highpoint_int = np.ceil(np.max(gridvaldata[sbjidx, :])*100).astype(int)
        #ccp_purpleptc[highpoint_int, :] = cc_pink

        #red_chance_cm = matplotlib.colors.ListedColormap(ccp_red)
        purpleptc_chance_cm = matplotlib.colors.ListedColormap(ccp_purpleptc)

        for paramspace in np.arange(10): #np.arange(gridvaldata.shape[1]): #np.arange(10)

            if paramspace == runset.members[sbjidx].container_info['best_param_idx']:
                heat_cm = ccpink_cm
                heat_rstep = param_raxis_step*2
            else:
                heat_cm = purpleptc_chance_cm
                heat_rstep = param_raxis_step

            cg1_circ.heatmap(arc_id, data=[gridvaldata[sbjidx, paramspace]],
                             positions=sbj_xpos,
                             width=sbjx_stepsize,
                             cmap=heat_cm,
                             raxis_range=(param_raxis_start, param_raxis_start + heat_rstep),
                             vmin=ptch_min, vmax=ptch_max) # , linewidth=0
            param_raxis_start += param_raxis_step
        sbj_xpos += sbjx_stepsize

    # param space chance level plots
    # for paramspace in np.arange(30): #np.arange(gridvaldata.shape[1]): #np.arange(10)
    #     # edit: change it to accomdate participant-wise colormaps\
    #     sbj_xpos = np.asarray([0]).astype(float);
    #     sbjx_stepsize = cg1_circ._garc_dict[arc_id].size / runset_net_mixmets[0].members_n
    #     for sbjidx in np.arange(runset_net_mixmets[0].members_n):
    #         ptch_min = np.min(gridvaldata[sbjidx,:])
    #         ptch_max = np.max(gridvaldata[sbjidx,:])
    #
    #         cg1_circ.heatmap(arc_id, data=[gridvaldata[sbjidx, paramspace]],
    #                          positions=sbj_xpos,
    #                          width=sbjx_stepsize,
    #                          cmap=plot_cmap_val,
    #                          raxis_range=(param_raxis_start, param_raxis_start + param_raxis_step),
    #                          vmin=heat_min, vmax=heat_max, linewidth=0.01)
    #         sbj_xpos+=sbjx_stepsize
    #     # cg1_circ.heatmap(arc_id, data=gridvaldata[:,paramspace], cmap=plot_cmap_val, raxis_range=(param_raxis_start, param_raxis_start+param_raxis_step),
    #     #                  vmin=heat_min, vmax=heat_max, linewidth=0.01)
    #     param_raxis_start += param_raxis_step

    cg1_circ.heatmap(arc_id, data=bestvalf1_data, vmin=heat_min, vmax=heat_max, cmap=plot_cmap_val, raxis_range=(425, 450), linewidth=0.03)
    cg1_circ.heatmap(arc_id, data=testf1_data, vmin=heat_min, vmax=heat_max, cmap=plot_cmap_test, raxis_range=(400, 425), linewidth=0.03)

    # draw connections for each significant difference

    # source = ('amica_iclabel_shallownet', 50, 100, 180)
    # destination = ('noica_shallownet', 50, 100, 180)
    # cg1_circ.chord_plot(source, destination)

#circos chord plot
#all_pvalues[5]=0.0 #temporary measure due to nan

# correct pvalus for multiple comparison
# in each ica-comp set there are 5x4/2 = 10 comparisons
#all_pvalues_np = np.asarray(all_pvalues)*10
# instead of normalizing put logarithms on p values
# this means we need to adjust pvalue==0 to 1e-100 or something like that
# done above
#all_pvalues_np[np.where(all_pvalues_np>1)[0]]=1
#all_pvalues_np[np.where(all_pvalues_np==0)[0]]=10**-100
pval_log = np.log(np.asarray(all_pvalues)) * -1
#pval_norm = preprocessing.normalize(pval_log.reshape(-1,1))

connection_colorbags = [(0.58, 0, 0.58), (0.32, 0.23, 0.56), (0, 0.25, 0.63)]
plot_cmap_conn = plt.cm.get_cmap("Set3", runset_net_mixmets_srt[0].members_n)

for networkidx, networkset in enumerate(stat_config['networks']):
    for sbjidx in np.arange(runset_net_mixmets_srt[0].members_n):
        current_anovarr_id = networkidx*runset_net_mixmets_srt[0].members_n+sbjidx
        target_pvalue = anov_results[networkidx*runset_net_mixmets_srt[0].members_n+sbjidx].pvalue
        if anov_results[networkidx*runset_net_mixmets_srt[0].members_n+sbjidx].pvalue<0.05:
            startpoint = sbjidx*1000/runset_net_mixmets_srt[0].members_n
            endpoint = (sbjidx+1)*1000/runset_net_mixmets_srt[0].members_n
            #all_pipes = list(cg1_circ._garc_dict.keys())[networkidx::len(stat_config['networks'])]
            all_pipes = list(cg1_circ._garc_dict.keys())[networkidx*len(stat_config['runsets']):(networkidx+1)*len(stat_config['runsets'])]
            connections = list(itertools.combinations(all_pipes,2))
            for connection in connections:
                source = (connection[0], startpoint, endpoint, 400)
                destination = (connection[1], startpoint, endpoint, 400)
                #cg1_circ.chord_plot(source, destination, facecolor=(0.58*pval_norm[current_anovarr_id].item(), 0.0, .827*pval_norm[current_anovarr_id].item(), 0.6),linewidth=0.005)
                # cg1_circ.chord_plot(source, destination, facecolor=(
                # connection_colorbags[networkidx][0] * pval_norm[current_anovarr_id].item(), connection_colorbags[networkidx][1], connection_colorbags[networkidx][2] * pval_norm[current_anovarr_id].item(), 0.5),
                #                     linewidth=0.008)
                cg1_circ.chord_plot(source, destination, facecolor=(n_cmaps[networkidx].colors[sbjidx][0],
                                                                    n_cmaps[networkidx].colors[sbjidx][1],
                                                                    n_cmaps[networkidx].colors[sbjidx][2], 0.8),
                                    linewidth=0.002)

# 3. tukey_individual plot
tukey_figs = []
xlabel = [j[0].data + ' & ' + j[1].data for j in icasbj_tukeys[0]._results_table][1:]

for networkidx, networkset in enumerate(stat_config['networks']):

    # net_tukey_ps = [
    #     tukres[3].data for tukres in
    #     [tuk for tuklist in
    #      [indivtuk._results_table[1:] for indivtuk in icasbj_tukeys[networkidx*runset_net_mixmets_srt[0].members_n:(networkidx+1)*runset_net_mixmets_srt[0].members_n]]
    #     for tuk in tuklist]
    # ]
    net_tukey_ps = [
        indivtuk.pvalues for indivtuk in icasbj_tukeys[networkidx*runset_net_mixmets_srt[0].members_n:(networkidx+1)*runset_net_mixmets_srt[0].members_n]
    ]
    net_tukey_h0 = [
        indivtuk.reject for indivtuk in icasbj_tukeys[
                                         networkidx * runset_net_mixmets_srt[0].members_n:(networkidx + 1) *
                                                                                          runset_net_mixmets_srt[
                                                                                              0].members_n]
    ]
    net_tukey_diff = [
        indivtuk.meandiffs for indivtuk in icasbj_tukeys[
                                         networkidx * runset_net_mixmets_srt[0].members_n:(networkidx + 1) *
                                                                                          runset_net_mixmets_srt[
                                                                                              0].members_n]
    ]

    net_tukey_psnp = np.asarray(net_tukey_ps).flatten()
    net_tukey_psnp[net_tukey_psnp==0]=10**-100
    net_tukey_diffnp = np.asarray(net_tukey_diff).flatten()
    net_tukey_psnplog = -np.log10(net_tukey_psnp)
    net_tukey_h0np = np.asarray(net_tukey_h0).flatten()


    # net_tukey_h0 = [
    #     tukres[6].data for tukres in
    #     [tuk for tuklist in
    #      [indivtuk._results_table[1:] for indivtuk in icasbj_tukeys[networkidx*runset_net_mixmets_srt[0].members_n:(networkidx+1)*runset_net_mixmets_srt[0].members_n]]
    #     for tuk in tuklist]
    # ]

    tukey_fig = plt.subplots(1,1)
    tukey_ax = tukey_fig[1]

    ylabels = np.arange(runset_net_mixmets_srt[networkidx * 5].members_n) # rows
    xlabels = xlabel #cols


    cirx, ciry = np.meshgrid(np.arange(len(xlabel)), np.arange(ylabels.shape[0])) #col, row mesh
    #cirr = net_tukey_psnplog/net_tukey_psnplog.max()/2 # circle radius
    #cirr = net_tukey_diffnp.shape[0] / net_tukey_diffnp.max() / 2  # circle radius
    cirr = np.ones(net_tukey_diffnp.shape[0])*0.3  # circle radius

    # draw in the significant ones
    sigs = np.where(net_tukey_h0np==True)[0]
    nonsigs = np.where(net_tukey_h0np==False)[0]
    tukcircles = [plt.Circle((j,i), radius=r) for r,j,i in zip(cirr[sigs], cirx.flat[sigs], ciry.flat[sigs])]
    tukcol = PatchCollection(tukcircles, array=net_tukey_psnplog[sigs], cmap='RdYlGn')
    tukey_ax.add_collection(tukcol)
    tukcircles_n = [plt.Circle((j, i), radius=r) for r, j, i in zip(cirr[nonsigs], cirx.flat[nonsigs], ciry.flat[nonsigs])]
    tukcol_n = PatchCollection(tukcircles_n, array=net_tukey_psnplog[nonsigs], cmap=allgray_cm)
    tukey_ax.add_collection(tukcol_n)

    tukey_ax.set_xlim([0-0.5,len(xlabel)-.5])
    tukey_ax.set_ylim([0-0.5,ylabels.shape[0]-0.5])
    tukey_ax.set(xticks=np.arange(len(xlabel)), yticks=np.arange(ylabels.shape[0]),
                 xticklabels=xlabel, yticklabels=ylabels)
    tukey_ax.set_xticks(np.arange(len(xlabel) + 1) - 0.5, minor=True)
    tukey_ax.set_yticks(np.arange(ylabels.shape[0]) - 0.5, minor=True)
    tukey_ax.set_ylabel('Ptc.num')
    plt.xticks(rotation=90)

    tukey_fig[0].colorbar(tukcol, label='-log10(P), p<0.05')
    tukey_ax.grid(which='minor')
    tukey_ax.set_title(f'Tukey HSD P, {dataset["dsetname"]}, {networkset}')

    plt.show()




    #draw the non significant ones







# These could be participant specific


ccp_red = redcm_colors
ccp_purple = purplecm_colors


red_chance_cm = matplotlib.colors.ListedColormap(ccp_red)
purple_chance_cm = matplotlib.colors.ListedColormap(ccp_purple)

#newcolors[:25, :] = pink
#newcmp = plt.cm.ListedColormap(newcolors)



#plt.pause()
plt.show(block=False)
#print('ha')
