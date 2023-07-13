function [epo] = convert_eeglab_to_bbci_amica(eegdata, eegdata_comp)
%CONVERT_EEGLAB_TO_BBCI Summary of this function goes here
%   Detailed explanation goes here

epo = [];
epo.clab = {eegdata.chanlocs.labels};
epo.fs = eegdata.srate;
epo.file = eegdata.filename;
epo.t = eegdata.times;
%trials_x = cellfun(@transpose,ftdata.trial,'un',0); % T C N % CTN TO TCN
epo.x = permute(eegdata_comp, [2 1 3]);
epo.mix = eegdata.etc.amica.A;
epo.unmix = eegdata.etc.amica.W;
%epo.className = classnames;
%epo.y = y_onehot;
%epo.indexedByEpochs={'time'};



end

