
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

ftmrk_root = './ftmrks/';
% load ftmrk to use in epoching later
%load([ftmrk_root 'ftmrk_rw_nextday.mat']);

%{
dsf

%}

% importing raw data from eeglab format
data_root = ['d:/data/cogsysbci/'];
root_folder =  ['d:/data/cogsysbci/prep_ready/bcic42a'];
ica_root = [data_root 'amica/bcic42a'];
output_iclabel_root = [data_root 'amica_iclabel_ready/bcic42a'];
output_mara_root = [data_root 'amica_mara_ready/bcic42a'];
amica_out_root = [data_root 'amica_out/bcic42a'];

%creating folders if they don't exist

check_create_folder(ica_root);
check_create_folder(output_iclabel_root);
check_create_folder(output_mara_root);
check_create_folder(amica_out_root);



sbj_size = 9;
sesh_size = 2;
session_types = {'E', 'T'};
skip_sbj = [];

for ptcidx = 1:sbj_size
    %skip if subject idx is in skip list
    if find(skip_sbj==ptcidx)
        continue;
    end
    x_data_ic = [];
    x_data_mara = [];
    x_train_ic = [];
    x_test_ic = [];
    
    y_train = [];
    y_test = [];


    x_train_mara = [];
    x_test_mara = [];

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
        sesh_mrk = truelabel_cell{seshidx};
        % for memory we also need to collate the label data (because the
        % other train_ready file used trial rejections, whereas we won't
        % here, we can't just reuse them from the previous runs)


         %step 1 . apply amica
        % EEGLAB data needs to be flattened into 2 dimensions, because that's how
        % amica accepts information
        eeglabcb = reshape(eegdata_epo.data, length(eegdata_epo.chanlocs), []);
        am_numprocs = 1; % number of process 
        % (qsub, for some reason anything higher than 1 is not working on my pc: multiprocessing not supported?)
        am_maxthread = 6; % maximum number of thread per 
        am_nmodel = 1; % we only wnat to do component rejection instead of trial rejection, so only a single model is necessary
        am_numchan = length(eegdata_epo.chanlocs);
        am_max_iter = 1000;
        %am_save_path = '/../../../../data/sbj01/eegmanypipelines/amica';
        am_save_path = [amica_out_root '/sbj' num2str(ptcidx) '/sesh' num2str(seshidx)];
        check_create_folder(am_save_path);
        %am_save_path = 'amica_results/sbj_01';
        
        [am_weight, am_sphere, am_mod] = runamica15(eeglabcb, 'num_models', am_nmodel, ...
            'numprocs', am_numprocs, 'max_threads', am_maxthread, 'max_iter', am_max_iter, ...
            'outdir', am_save_path, 'num_chans', am_numchan);
        
        % to construct individual trial components, we msut run eeg_getica on top
        % of this.
        % eeg_getica requires the eeglab dataformat to contain the ica results
        % (spheres and mixing weight matrices), so we need to put it in there
        eegdata_epo.etc.amica = am_mod;
        eegdata_epo.icaweights = am_weight;
        eegdata_epo.icasphere = am_sphere;
        eegdata_epo.icawinv = am_mod.A;
        eegdata_epo.icachansind = 1:eegdata_epo.nbchan;
    
        
        % now we're ready to make the rejections and reconstruction
        % apply icalabel
        iclami_eeg = iclabel(eegdata_epo);
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
       if seshidx ==1
        y_test = truelabel_cell{seshidx}';
        x_test_ic = x_datasesh;
       else
        y_train = truelabel_cell{seshidx}';
        x_train_ic = x_datasesh;
       end
       %y_nextdaysesh = [ftmrk_rw_nextday{seshidx, ptcidx}.label];
       %x_data_ic = cat(1, x_data_ic, x_datasesh);
       %y_nextday = [y_nextday y_nextdaysesh]; 

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

        rejections_iclabel{seshidx} = rej_comp;
        rejections_mara{seshidx} = mara_rej_comp;
    
    
         % save component info
        rejcomp_save_path = [amica_out_root '/subj_' num2str(ptcidx) '_sesh' num2str(seshidx) '.mat'];
        save(rejcomp_save_path, '-v7.3', 'iclami_eeg', 'iclabel_probabilities', ...
            'iclabel_labels', 'rej_comp', 'rej_threshold', 'am_weight', 'am_mod', 'am_sphere', 'mara_rej_comp', 'mara_info');
    
    
        % save iclabel result train ready format
        % noica, so just assign relevant markers and call it a day
        %note : train ready data should ideally be N x chan x time 
        % previous exports of train ready were time x chan x N
        % where as current state of eeglab data is in chan x time x trial
    
        
       x_datasesh = mara_eeg_rej.data;
       x_datasesh = permute(x_datasesh, [3 2 1]);
       %x_data_mara = cat(1, x_data_mara, x_datasesh);
       % share the y data with iclabel result, because they don't change from
       % different ic rejection methods

       if seshidx ==1
        
        x_test_mara = x_datasesh;
       else
        
        x_train_mara = x_datasesh;
       end


    end


    
   %these are constant throughout sessions, so no need to do them in every
   %session
   
   time_zero_point = 125; % based on 0-index
   time_arr = eegdata_epo.times;
   d_chan_info = {eegdata_epo.chanlocs.labels};
   
    
    x_train = x_train_ic;
    x_test = x_test_ic;
    % save after all sessions in participant is done
    output_path = [output_iclabel_root '/subj_' num2str(ptcidx) '.mat'];
    save(output_path, '-v7.3', 'x_train', 'x_test', 'time_zero_point', 'time_arr', 'd_chan_info', 'y_train', 'y_test', 'rejections_iclabel');

    x_train = x_train_mara;
    x_test = x_test_mara;
    output_path = [output_mara_root '/subj_' num2str(ptcidx) '.mat'];
    save(output_path, '-v7.3', 'x_train', 'x_test', 'time_zero_point', 'time_arr', 'd_chan_info', 'y_train', 'y_test', 'rejections_mara');


end %endof subject loop

