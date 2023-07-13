clear;
startup;
eeglab;
addpath('../common/');
addpath(genpath('../common/eegprep/'));

ftmrk_root = './ftmrks/';
% load ftmrk to use in epoching later
load([ftmrk_root 'ftmrk_askrw_nextday.mat']);
ftmrk_rw_nextday = ftmrk_askrw_nextday;

session_size = 15;
sbj_size = 15;
seshlist_file = './sessionlists/sessionlist';
sslist = textscan(fopen(seshlist_file), '%s');
sslist = sslist{1};

data_root = 'D:/data/bbciRaw/';
output_root = 'd:/data/cogsysbci/prep_ready/language_memory';
if ~exist(output_root, 'dir')
  [parentdir, newdir]=fileparts(output_root);
  [status,msg]= mkdir(parentdir, newdir);
  if status~=1
    error(msg);
  end
end


skip_sbj = [1:14];

for ptcidx = 1:sbj_size
    disp(['running subject ' num2str(ptcidx) ]);
    %skip if subject idx is in skip list
    if find(skip_sbj==ptcidx)
        continue;
    end

    prepdata_cell = [];

    for seshidx = 1:session_size
        disp(['running sbj ' num2str(ptcidx) 'session ' num2str(seshidx)]);
        seshfile = sslist{(ptcidx-1)*session_size+seshidx};
        seshpath = [data_root '/' seshfile]
        dfile = [seshpath '.eeg'];
        hdfile = [seshpath '.vhdr'];
        [hdfile_path hdfile_name hdfile_ext] = fileparts(hdfile);
        eegdata_bvio = pop_loadbv(hdfile_path, [hdfile_name hdfile_ext]);

        params = {};
        params.lineFrequencies = [60, 120, 180, 210]; % korea's power line noise is at 60Hz
        params.detrendType = 'high pass';
        params.detrendCutoff = 1;
        params.referenceType = 'robust';
        params.keepFiltered = false;
        %params.removeInterpolatedChannels = true;
        
        params.ignoreBoundaryEvents = true; %likely there will be breaks in the middle of recording
        % use params to exclude non EEG channels here
        % specficially, referenceChannels, evaluationChannels.
        % that should help with excluding those channels
        params.referenceChannels = 1:62; % exclude EOG! (63) dataset specific
        params.evaluationChannels = 1:62; % exclude EOG! (63) dataset specific
        params.rereferencedChannels = 1:63;
        params.detrendChannels = 1:63;
        params.lineNoiseChannels = 1:63;       


        %fill in channel location info from template file provided by eeglab
        eegdata_bvio.chanlocs = get_chan_locs(eegdata_bvio.chanlocs, true);
        
        [eeg_preped, test_params, test_computationtime] = prepPipeline(eegdata_bvio, params);
        %eeg_preped = eegdata_bvio;
        %the specific rejected channels should be in
        %testeeg_done.etc.noiseDetection.interpolatedChannelnumbers
        % use a length() on it to reduce the signal rank count 
        
        % post processing (filtering to 1 100hz, downsampling, epoching) has to be done
        % afterwards
        % lets do filtering on eeglab to rid of the inconvenience
        
        eeg_preped = pop_eegfiltnew(eeg_preped, 1, 100);
        eeg_preped = pop_resample(eeg_preped, 100);
        
        
        %%%MEMORY ONLY : use predefined marker file (possibly for other public datasets as well) %%
        % override data.events that contain all sorts of other markers (the ftmrk's
        % time data matches data.events time markers, so this can be done without issue)
        target_ftmrk = ftmrk_rw_nextday{seshidx,ptcidx};
        event_new = {};
        for evidx=1:length(target_ftmrk)
            % data has been resampled from 1000 to 100hz... need to readjust marker samples
            event_new(evidx).latency = round(target_ftmrk(evidx).sample./10); 
            event_new(evidx).duration = 1;
            event_new(evidx).channel = 0;
            event_new(evidx).bvtime = [];
            event_new(evidx).bvmknum = evidx;
            event_new(evidx).visible = [];
            event_new(evidx).type = '161'; %event specific, but we only have one event so 
            event_new(evidx).code = 'Stimulus';
            event_new(evidx).urevent = evidx;
        
        end %end of ftmrk looop
        
        % apply new event data
        eeg_preped.event = event_new;
        eeg_preped.urevent = rmfield(event_new, 'urevent');
        
        % epoch the data
        eegdata_epo = pop_epoch(eeg_preped, {'161'}, [-0.2 1]);


        
        prepdata_cell{seshidx} = eegdata_epo;
    end %endof sesh loop

    % save after all sessions in participant is done
    output_path = [output_root '/subj_' num2str(ptcidx) '.mat'];
    save(output_path, 'prepdata_cell', '-v7.3');

end %endof subject loop
