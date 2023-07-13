clear;
close all;
startup;
eeglab;

addpath('../common/');
addpath(genpath('../common/eegprep'));

root_folder =  ['E:/data_backup/cogsysbci_download/bruner_bci42a_2014/']; % for cntlab
root_gdf_folder = 'D:/data/bncI_datasets/bcic_2a/';
% root_folder =  ['d:/data/eegmanypipelines/']; % for vrpc
dataroot = [root_folder];



sbj_size = 9;
session_types = {'E', 'T'};

ptcidx = 1;
seshidx = 1;
output_root = 'd:/data/cogsysbci/prep_ready/bcic42a';


%channel labels are missing in thefiles, so we fill them out on our own
chname_list = {'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', ...
    'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', ...
    'POz', 'EOG-1', 'EOG-2', 'EOG-3'};
%and then insert the chanloc locations based on the reference channel
%location file from EEGLAB

eventcodes = {783}; %event label is unknown at this point, but we get it from a different file later


%load('./fieldtrip_layout_allchannels.mat'); % variable name ftdata_layout
%eegdata = pop_loadset(['EMP' num2str(ptcidx, '%02d') '.set'], dataroot);
%loadfile = [root_folder 'A' num2str(ptcidx, '%02d') session_types{seshidx} '.mat'];
%load(loadfile);

%load gdf using biosig
gdfl = pop_biosig([root_gdf_folder 'A01E.gdf']);


%mrkfile = ['./mrkfiles/ftmrk_ind_' num2str(ptcidx) '.mat'];
%load(mrkfile);


for chanidx=1:length(gdfl.chanlocs)
    gdfl.chanlocs(chanidx).labels = chname_list{chanidx};
end %endof chanidx loop

%set channel locations
gdfl.chanlocs = get_chan_locs(gdfl.chanlocs, true);

% apply PREP (for consistency sake, although line noise is already removed
% here)
% remove non eeg/eog channels 
% eegdata_rmchan = pop_select(eegdata, 'nochannel', {'POz', 'IO1', 'IO2', ...
%     'M1', 'M2', 'Afp9', 'Afp10'});
% 

% not sure if it's because the data is old, but the event code column for data.events has a
% different column name than recognized by pop_epoch later on
% so we do this:
for epidx = 1:length(gdfl.event)
    gdfl.event(epidx).type = gdfl.event(epidx).edftype;
end
%gdfl.event.type = {gdfl.event.edftype};

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
params.referenceChannels = 1:22; % exclude EOG! (23, 24, 25) dataset specific
params.evaluationChannels = 1:22; % exclude EOG! (23, 24, 25) dataset specific
params.rereferencedChannels = 1:25;
params.detrendChannels = 1:25;
params.lineNoiseChannels = 1:25;       


%fill in channel location info from template file provided by eeglab
%eegdata_rmchan.chanlocs = get_chan_locs(eegdata_rmchan.chanlocs, true);
% not necessary because eegmanypipeline data already has channel locations


[eeg_preped, test_params, test_computationtime] = prepPipeline(gdfl, params);
%the specific rejected channels should be in
%testeeg_done.etc.noiseDetection.interpolatedChannelnumbers
% use a length() on it to reduce the signal rank count 

% post processing (filtering to 1 100hz, downsampling, epoching) has to be done
% afterwards
% lets do filtering on eeglab to rid of the inconvenience

% note: BCIC42a data is already filtered between [0.5 100hz]
% keep the 250hz sampling frequency
%eeg_preped = pop_eegfiltnew(eeg_preped, 1, 100);
%eeg_preped = pop_resample(eeg_preped, 100);


%build a big epoch for now and cut it down later because PREP takes a lot
%longer than any ica related operations or further epoching
eegdata_epo = pop_epoch(eeg_preped, eventcodes, [-0.5 5]);


