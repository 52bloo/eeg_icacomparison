clear;
close all;
startup;
eeglab;
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
root_folder =  ['d:/data/cogsysbci/prep_ready/visual_memory'];
output_root = 'd:/data/cogsysbci/noica/visual_memory';
if ~exist(output_root, 'dir')
  [parentdir, newdir]=fileparts(output_root);
  [status,msg]= mkdir(parentdir, newdir);
  if status~=1
    error(msg);
  end
end

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
    load('./fieldtrip_layout_allchannels.mat'); % variable name ftdata_layout
    %eegdata = pop_loadset(['EMP' num2str(ptcidx, '%02d') '.set'], dataroot);
    load([root_folder '/subj_' num2str(ptcidx) '.mat']); % loaded variable will be eegdata_epo
    
    mrkfile = ['./mrkfiles/ftmrk_ind_' num2str(ptcidx) '.mat'];
    load(mrkfile);

    % noica, so just assign relevant markers and call it a day
    %note : train ready data should ideally be N x chan x time 
    % previous exports of train ready were time x chan x N
    % where as current state of eeglab data is in chan x time x trial

    
   x_data = eegdata_epo.data;
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
output_path = [output_root '/subj_' num2str(ptcidx) '.mat'];
save(output_path, '-v7.3', 'x_data', 'time_zero_point', 'time_arr', 'd_chan_info', 'y_tgcode', 'y_scene', 'y_old', 'y_behavior', 'y_memory');

end %endof subject loop

