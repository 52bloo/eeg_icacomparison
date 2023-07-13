
clear;
close all;
startup;
eeglab;
addpath('../common/');
addpath('../common/MARA/');
%%

% variables : eeg_chaninfo in EEG.chaninfo, eeg_chanloc_info in EEG.chanlocs
%eeglab_chaninfo_file = ['./eeglab_chaninfo.mat'];

% pipeline flow : import files in EEGLAB format
% this is a no-ica run, so ICA/component creation is unnecessary, only do
% preprocessing steps beforehand
% apply PREP 

%{
dsf

%}

ftmrk_root = './ftmrks/';
% load ftmrk to use in epoching later
load([ftmrk_root 'ftmrk_askrw_nextday.mat']);
ftmrk_rw_nextday = ftmrk_askrw_nextday;
% importing raw data from eeglab format
data_root = ['d:/data/cogsysbci/'];
root_folder =  ['d:/data/cogsysbci/prep_ready/language_memory'];
ica_root = [data_root 'runica/language_memory'];
output_iclabel_root = [data_root 'runica_iclabel_ready/language_memory'];
output_mara_root = [data_root 'runica_mara_ready/language_memory'];
amica_out_root = [data_root 'runica_out/language_memory'];

%creating folders if they don't exist

check_create_folder(ica_root);
check_create_folder(output_iclabel_root);
check_create_folder(output_mara_root);
check_create_folder(amica_out_root);


sbj_size = 15;
sesh_size = 15;
skip_sbj = [];

for ptcidx = 1:sbj_size
    %skip if subject idx is in skip list
    if find(skip_sbj==ptcidx)
        continue;
    end
    x_data_ic = [];
    x_data_mara = [];
    y_nextday = [];
    eeg_preped = [];

    rejections_mara = [];
    rejections_iclabel = [];

     

  
    %load('./fieldtrip_layout_allchannels.mat'); % variable name ftdata_layout
    %eegdata = pop_loadset(['EMP' num2str(ptcidx, '%02d') '.set'], dataroot);
    load([root_folder '/subj_' num2str(ptcidx) '.mat']); % loaded variable will be eegdata_epo

    for seshidx=1:sesh_size
        disp(['working on subj' num2str(ptcidx) 'sesh' num2str(seshidx)]);
        eegdata_epo = prepdata_cell{seshidx};
        

        ncomps_to_calc = length(eegdata_epo.chanlocs) - length(eegdata_epo.etc.noiseDetection.interpolatedChannelNumbers) ...
        - length(eegdata_epo.etc.noiseDetection.stillNoisyChannelNumbers) -1;
        eeg_icarun = pop_runica(eegdata_epo, 'icatype', 'runica', 'chanind', [1:length(eegdata_epo.chanlocs)], 'pca', ncomps_to_calc);


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
    x_datasesh = iclami_eeg_rej.data;
   x_datasesh = permute(x_datasesh, [3 2 1]);
   y_nextdaysesh = [ftmrk_rw_nextday{seshidx, ptcidx}.label];
   x_data_ic = cat(1, x_data_ic, x_datasesh);
   y_nextday = [y_nextday y_nextdaysesh]; 



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
    rejcomp_save_path = [amica_out_root '/subj_' num2str(ptcidx) '_sesh' num2str(seshidx) '.mat'];
    runica_info = eeg_icarun.etc;
    save(rejcomp_save_path, '-v7.3', 'iclami_eeg', 'iclabel_probabilities', ...
        'iclabel_labels', 'rej_comp', 'rej_threshold', 'runica_info', 'mara_rej_comp', 'mara_info');

     x_datasesh = mara_eeg_rej.data;
   x_datasesh = permute(x_datasesh, [3 2 1]);
   x_data_mara = cat(1, x_data_mara, x_datasesh);
   % share the y data with iclabel result, because they don't change from
   % different ic rejection methods

    end % end of seshloop


    
    
   time_zero_point = 20; % based on 0-index
   time_arr = eegdata_epo.times;
   d_chan_info = {eegdata_epo.chanlocs.labels};
   
    
    x_data = x_data_ic;
    % save after all sessions in participant is done
    output_path = [output_iclabel_root '/subj_' num2str(ptcidx) '.mat'];
    save(output_path, '-v7.3', 'x_data', 'time_zero_point', 'time_arr', 'd_chan_info', 'y_nextday', 'rejections_iclabel');

   time_zero_point = 20; % based on 0-index
   time_arr = eegdata_epo.times;
   d_chan_info = {eegdata_epo.chanlocs.labels};
   
    x_data = x_data_mara;
    output_path = [output_mara_root '/subj_' num2str(ptcidx) '.mat'];
    save(output_path, '-v7.3', 'x_data', 'time_zero_point', 'time_arr', 'd_chan_info', 'y_nextday', 'rejections_mara');

end %endof subject loop

