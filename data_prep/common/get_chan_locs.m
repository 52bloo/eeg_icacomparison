function [data_chan] = get_chan_locs(data_chan, use_defaultloc, ref_filepath)
%GET_CHAN_LOCS Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3 % use default path
    ref_filepath = 'C:/toolboxes/eeglab2021.1/functions/supportfiles/Standard-10-5-Cap385.sfp';
end

ref_chan = readlocs(ref_filepath);

for chanidx=1:length(data_chan)
    % run for each channel 
    datchan_name = data_chan(chanidx).labels;
    refidx = find(strcmp({ref_chan.labels}, datchan_name));

    if isempty(refidx)
        if use_defaultloc
            refidx = 1;

            %chan_old_name = data_chan(chanidx).labels;
            %data_chan(chanidx) = ref_chan(1);
            %data_chan(chanidx).labels = chan_old_name;
        else
            continue;
        end
        
        % just leave it empty, the pipeline (PREP) will throw an error from
        % an empty channel location info anyways
        %OR, they could be EOG/nonEEG channels, so just put some random
        %number in it?
   %else
       
    end % end of if checking for refidx found
    % assign channel's location info based on reference location data
    %data_chan(chanidx).ref = ref_chan(refidx).  % leave reference as empty
    data_chan(chanidx).theta = ref_chan(refidx).theta;
    data_chan(chanidx).radius = ref_chan(refidx).radius;
    data_chan(chanidx).sph_theta = ref_chan(refidx).sph_theta;
    data_chan(chanidx).sph_phi = ref_chan(refidx).sph_phi;
    data_chan(chanidx).sph_radius = ref_chan(refidx).sph_radius;
    data_chan(chanidx).X = ref_chan(refidx).X;
    data_chan(chanidx).Y = ref_chan(refidx).Y;
    data_chan(chanidx).Z = ref_chan(refidx).Z;
    %data_chan(chanidx).ffff = ref_chan(refidx).ffff;
    %data_chan(chanidx) = ref_chan(refidx);
    %ref theta radius X Y Z sph_theta sph_phi shp_radius type urchan

end % end of channel forloop 

end

    