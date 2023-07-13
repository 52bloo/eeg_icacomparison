
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

ftmrk_root = './ftmrks/';
% load ftmrk to use in epoching later
load([ftmrk_root 'ftmrk_askrw_nextday.mat']);
ftmrk_rw_nextday = ftmrk_askrw_nextday;

%{
dsf

%}

% importing raw data from eeglab format
data_root = ['d:/data/cogsysbci/'];
root_folder =  ['d:/data/cogsysbci/prep_ready/language_memory'];
ica_root = [data_root 'amica/language_memory'];
output_noica_root = [data_root 'noica/language_memory'];
%output_mara_root = [data_root 'amica_mara_ready/language_memory'];
%amica_out_root = [data_root 'amica_out/language_memory'];

%creating folders if they don't exist

check_create_folder(ica_root);
check_create_folder(output_noica_root);




sbj_size = 15;
sesh_size = 15;

skip_sbj = [];

for ptcidx = 1:sbj_size
    %skip if subject idx is in skip list
    if find(skip_sbj==ptcidx)
        continue;
    end
    x_data = [];
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
        % run ica and rejections once for each session (makes it tough yea)

        % for memory we also need to collate the label data (because the
        % other train_ready file used trial rejections, whereas we won't
        % here, we can't just reuse them from the previous runs)


         %step 1 . apply amica
        % EEGLAB data needs to be flattened into 2 dimensions, because that's how
        % amica accepts information

        %iclami_eeg_rej.epoch = epoeegdata.epoch;
        %iclami_eeg_rej.event = epoeegdata.event;
        %iclami_eeg_rej.urevent = epoeegdata.urevent;
    
       
        % save iclabel result train ready format
        % noica, so just assign relevant markers and call it a day
        %note : train ready data should ideally be N x chan x time 
        % previous exports of train ready were time x chan x N
        % where as current state of eeglab data is in chan x time x trial
    
        
       x_datasesh = eegdata_epo.data;
       x_datasesh = permute(x_datasesh, [3 2 1]);
       y_nextdaysesh = [ftmrk_rw_nextday{seshidx, ptcidx}.label];
       x_data = cat(1, x_data, x_datasesh);
       y_nextday = [y_nextday y_nextdaysesh]; 


    end


    
   %these are constant throughout sessions, so no need to do them in every
   %session
   
   time_zero_point = 20; % based on 0-index
   time_arr = eegdata_epo.times;
   d_chan_info = {eegdata_epo.chanlocs.labels};
   
    
    %x_data = x_data_ic;
    % save after all sessions in participant is done
    output_path = [output_noica_root '/subj_' num2str(ptcidx) '.mat'];
    save(output_path, '-v7.3', 'x_data', 'time_zero_point', 'time_arr', 'd_chan_info', 'y_nextday');



end %endof subject loop

