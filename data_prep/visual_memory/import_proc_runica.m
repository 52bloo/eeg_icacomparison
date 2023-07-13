
clear;
close all;
startup;
eeglab;
addpath('../common/');
addpath('../common/MARA/');
%%

% variables : eeg_chaninfo in EEG.chaninfo, eeg_chanloc_info in EEG.chanlocs
eeglab_chaninfo_file = ['./eeglab_chaninfo.mat'];

% pipeline flow : import files in EEGLAB format
% this is a no-ica run, so ICA/component creation is unnecessary, only do
% preprocessing steps beforehand
% apply PREP 

%{
dsf

%}

% importing raw data from eeglab format
data_root = ['d:/data/cogsysbci/'];
root_folder =  ['d:/data/cogsysbci/prep_ready/visual_memory'];
ica_root = [data_root 'runica/visual_memory'];
output_iclabel_root = [data_root 'runica_iclabel_ready/visual_memory'];
output_mara_root = [data_root 'runica_mara_ready/visual_memory'];
amica_out_root = [data_root 'runica_out/visual_memory'];

%creating folders if they don't exist

check_create_folder(ica_root);
check_create_folder(output_iclabel_root);
check_create_folder(output_mara_root);
check_create_folder(amica_out_root);


eventcodes = {'2040', '1129', '1040', '1049', '1041', '1030', '1031', ...
    '1039', '1110', '1111', '1119', '1120', '1121', '2030', '2031', '2039,'...
    '2041', '2049', '2091', '2110', '2111', '2119', '2120', '2121', '2129'};


sbj_size = 33;


skip_sbj = [];

for ptcidx = 1:sbj_size
    %skip if subject idx is in skip list
    if find(skip_sbj==ptcidx)
        continue;
    end
    x_data = [];
    y_scene= [];
    y_old = [];
    y_behavior = [];
    y_memory= [];
    eeg_preped = [];

    disp(['working on subj' num2str(ptcidx)]);
    %load('./fieldtrip_layout_allchannels.mat'); % variable name ftdata_layout
    %eegdata = pop_loadset(['EMP' num2str(ptcidx, '%02d') '.set'], dataroot);
    load([root_folder '/subj_' num2str(ptcidx) '.mat']); % loaded variable will be eegdata_epo

    mrkfile = ['./mrkfiles/ftmrk_ind_' num2str(ptcidx) '.mat'];
    load(mrkfile);
    
    %mrkfile = ['./mrkfiles/ftmrk_ind_' num2str(ptcidx) '.mat'];
    %load(mrkfile);


    %step 1 . apply binica (Inofmax ICA, runica's faster version) 
    % EEGLAB data needs to be flattened into 2 dimensions, because that's how
    % amica accepts information
%     eeglabcb = reshape(eegdata_epo.data, length(eegdata_epo.chanlocs), []);
%     am_numprocs = 1; % number of process 
%     % (qsub, for some reason anything higher than 1 is not working on my pc: multiprocessing not supported?)
%     am_maxthread = 6; % maximum number of thread per 
%     am_nmodel = 1; % we only wnat to do component rejection instead of trial rejection, so only a single model is necessary
%     am_numchan = length(eegdata_epo.chanlocs);
%     am_max_iter = 200;
%     %am_save_path = '/../../../../data/sbj01/eegmanypipelines/amica';
%     am_save_path = [amica_out_root '/sbj' num2str(ptcidx)];
%     check_create_folder(am_save_path);
%     %am_save_path = 'amica_results/sbj_01';
%     
%     [am_weight, am_sphere, am_mod] = runamica15(eeglabcb, 'num_models', am_nmodel, ...
%         'numprocs', am_numprocs, 'max_threads', am_maxthread, 'max_iter', am_max_iter, ...
%         'outdir', am_save_path, 'num_chans', am_numchan);
    ncomps_to_calc = length(eegdata_epo.chanlocs) - length(eegdata_epo.etc.noiseDetection.interpolatedChannelNumbers) ...
     - length(eegdata_epo.etc.noiseDetection.stillNoisyChannelNumbers)-1;
    %eegdata_ft = eeglab2fieldtrip(eegdata_epo, 'preprocessing');
%     cfg        = [];
%       cfg.method = 'runica';
%       cfg.numcomponent = ncomps_to_calc;
%       ft_comp = ft_componentanalysis(cfg, eegdata_ft);
    
 
    %eeg_icarun = pop_runica(eegdata_epo, 'icatype', 'runica', 'chanind', [1:length(eegdata_epo.chanlocs)]);
    eeg_icarun = pop_runica(eegdata_epo, 'icatype', 'runica', 'chanind', [1:length(eegdata_epo.chanlocs)], 'pca', ncomps_to_calc);
    %eeg_icarun = pop_runica(eegdata_epo, 'icatype', 'runica', 'chanind', [1:length(eegdata_epo.chanlocs)], 'ncomps', ncomps_to_calc);
%     [eegdata_epo.weight, eegdata_epo.sphere] = runica(reshape(eegdata_epo.data, [size(eegdata_epo.data, 1), size(eegdata_epo.data, 2)*size(eegdata_epo.data, 3)]), ...
%     'ncomps', ncomps_to_calc);
    %eeg_icarun = pop_runica(eegdata_epo, 'icatype', 'binica', 'chanind', [1:length(eegdata_epo.chanlocs)], 'ncomps', ncomps_to_calc);


    
    % now we're ready to make the rejections and reconstruction
    % apply icalabel
    iclami_eeg = iclabel(eeg_icarun);
    % set rejection threshold for eye and muscle component probability
    rej_threshold = .8;
    rej_comp = find(iclami_eeg.etc.ic_classification.ICLabel.classifications(:,2)>rej_threshold ...
        | iclami_eeg.etc.ic_classification.ICLabel.classifications(:,3)>rej_threshold); %%SAVE
    
    % information we need to save here for later:
    % classification probabilities for each component
    % predicted labels for each component(whcih artifact?)
    % rejected component indices
    iclabel_probabilities = iclami_eeg.etc.ic_classification.ICLabel.classifications; %%SAVE
    iclabel_labels = iclami_eeg.etc.ic_classification.ICLabel.classes;  %% SAVE
    
    % reject components and reconstruct eeg from the remaining components
    iclami_eeg_rej = pop_subcomp(iclami_eeg, rej_comp);
    %iclami_eeg_rej.epoch = epoeegdata.epoch;
    %iclami_eeg_rej.event = epoeegdata.event;
    %iclami_eeg_rej.urevent = epoeegdata.urevent;

   
    % save iclabel result train ready format
    % noica, so just assign relevant markers and call it a day
    %note : train ready data should ideally be N x chan x time 
    % previous exports of train ready were time x chan x N
    % where as current state of eeglab data is in chan x time x trial

    
   x_data = iclami_eeg_rej.data;
   x_data = permute(x_data, [3 2 1]);
   time_zero_point = 20; % based on 0-index
   time_arr = eegdata_epo.times;
   d_chan_info = {eegdata_epo.chanlocs.labels};
   y_tgcode = [ftmrk_ar.tgcode];
   y_scene = [ftmrk_ar.lb_scene];
   y_old = [ftmrk_ar.lb_old];
   y_behavior = [ftmrk_ar.lb_behavior];
   y_memory = [ftmrk_ar.lb_memory];
    

% save after all sessions in participant is done
output_path = [output_iclabel_root '/subj_' num2str(ptcidx) '.mat'];
save(output_path, '-v7.3', 'x_data', 'time_zero_point', 'time_arr', 'd_chan_info', 'y_tgcode', 'y_scene', 'y_old', 'y_behavior', 'y_memory');


% now do MARA
[mara_rej_comp, mara_info] =  MARA(iclami_eeg);
%eegcomp = eeg_getica(iclami_eeg);
%eegcomp_bbci = convert_eeglab_to_bbci_amica(iclami_eeg, eegcomp);
%[mara_goodcomp, mara_info] = proc_mara_nolas(eegcomp_bbci, {iclami_eeg.chanlocs.labels}, eegdata_epo.icaweights); % input is data, channels, and mixing array
    
    % based on selection we need to reject components, do it here:
    % no need to reconvert to ft for now? up to this point we kept the
    % original comp format. YUP.
    
    
    % seshepo is bbci, but ft_rejectcomponent requires ft format. better
    % convert them before merging
    %seshepo_ft = 
    
    
    % reforge data based on everything:
    mara_eeg_rej = pop_subcomp(iclami_eeg, mara_rej_comp);


     % save component info
    rejcomp_save_path = [amica_out_root '/subj_' num2str(ptcidx) '.mat'];
    runica_info = eeg_icarun.etc;
    save(rejcomp_save_path, '-v7.3', 'iclami_eeg', 'iclabel_probabilities', ...
        'iclabel_labels', 'rej_comp', 'rej_threshold', 'runica_info', 'mara_rej_comp', 'mara_info');


    % save iclabel result train ready format
    % noica, so just assign relevant markers and call it a day
    %note : train ready data should ideally be N x chan x time 
    % previous exports of train ready were time x chan x N
    % where as current state of eeglab data is in chan x time x trial

    
   x_data = mara_eeg_rej.data;
   x_data = permute(x_data, [3 2 1]);
   time_zero_point = 20; % based on 0-index
   time_arr = eegdata_epo.times;
   d_chan_info = {eegdata_epo.chanlocs.labels};
   y_tgcode = [ftmrk_ar.tgcode];
   y_scene = [ftmrk_ar.lb_scene];
   y_old = [ftmrk_ar.lb_old];
   y_behavior = [ftmrk_ar.lb_behavior];
   y_memory = [ftmrk_ar.lb_memory];
    
    output_path = [output_mara_root '/subj_' num2str(ptcidx) '.mat'];
save(output_path, '-v7.3', 'x_data', 'time_zero_point', 'time_arr', 'd_chan_info', 'y_tgcode', 'y_scene', 'y_old', 'y_behavior', 'y_memory');


end %endof subject loop

