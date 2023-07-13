clear;
startup;
addpath(genpath('../common/eegprep'));
eeglab


root_folder =  ['e:/data/eegmanypipelines/']; % for cntlab
% root_folder =  ['d:/data/eegmanypipelines/']; % for vrpc
dataroot = [root_folder 'eeglab_format/'];

eventcodes = {'2040', '1129', '1040', '1049', '1041', '1030', '1031', ...
    '1039', '1110', '1111', '1119', '1120', '1121', '2030', '2031', '2039,'...
    '2041', '2049', '2091', '2110', '2111', '2119', '2120', '2121', '2129', ...
    '2090', '2091', '2099', '2190', '2191', '2199', ...
    '1090', '1091', '1190', '1191','1199', '1099', };

sbj_size = 33;


output_root = 'd:/data/cogsysbci/prep_ready/visual_memory';
if ~exist(output_root, 'dir')
  [parentdir, newdir]=fileparts(output_root);
  [status,msg]= mkdir(parentdir, newdir);
  if status~=1
    error(msg);
  end
end


skip_sbj = [1:3];

for ptcidx = 1:sbj_size
    %skip if subject idx is in skip list
    if find(skip_sbj==ptcidx)
        continue;
    end
    
    eeg_preped = [];

    disp(['working on subj' num2str(ptcidx)]);
    load('./fieldtrip_layout_allchannels.mat'); % variable name ftdata_layout
    eegdata = pop_loadset(['EMP' num2str(ptcidx, '%02d') '.set'], dataroot);
    
    mrkfile = ['./mrkfiles/ftmrk_ind_' num2str(ptcidx) '.mat'];
    load(mrkfile);
    
   

    
% remove non eeg/eog channels 
eegdata_rmchan = pop_select(eegdata, 'nochannel', {'POz', 'IO1', 'IO2', ...
    'M1', 'M2', 'Afp9', 'Afp10'});


params = {};
params.lineFrequencies = [50, 100, 150, 200]; % data was collected in europe, so power line is 50Hz
params.detrendType = 'high pass';
params.detrendCutoff = 1;
params.referenceType = 'robust';
params.keepFiltered = false;
%params.removeInterpolatedChannels = true;

params.ignoreBoundaryEvents = true; %likely there will be breaks in the middle of recording
% use params to exclude non EEG channels here
% specficially, referenceChannels, evaluationChannels.
% that should help with excluding those channels
params.referenceChannels = 1:63; % exclude EOG! (64, 65) dataset specific
params.evaluationChannels = 1:63; % exclude EOG! (64, 65) dataset specific
params.rereferencedChannels = 1:65;
params.detrendChannels = 1:65;
params.lineNoiseChannels = 1:65;       


%fill in channel location info from template file provided by eeglab
%eegdata_rmchan.chanlocs = get_chan_locs(eegdata_rmchan.chanlocs, true);
% not necessary because eegmanypipeline data already has channel locations


[eeg_preped, test_params, test_computationtime] = prepPipeline(eegdata_rmchan, params);
%the specific rejected channels should be in
%testeeg_done.etc.noiseDetection.interpolatedChannelnumbers
% use a length() on it to reduce the signal rank count 

% post processing (filtering to 1 100hz, downsampling, epoching) has to be done
% afterwards
% lets do filtering on eeglab to rid of the inconvenience

eeg_preped = pop_eegfiltnew(eeg_preped, 1, 100);
eeg_preped = pop_resample(eeg_preped, 100);

eegdata_epo = pop_epoch(eeg_preped, eventcodes, [-0.2 1]);

% save after all sessions in participant is done
output_path = [output_root '/subj_' num2str(ptcidx) '.mat'];
save(output_path, 'eegdata_epo');

end %endof subject loop
