clear;
close all;
startup;
data_root = ['d:/data/cogsysbci/'];

root_folder =  ['d:/data/cogsysbci/noica/visual_memory'];

alls =0;
behavs = 0;
mems = 0;

allsind = zeros(33,1);
behavsind = zeros(33,1);
memsind = zeros(33,1);



for sbji=1:33

    %load data
    load([root_folder '/subj_' num2str(sbji) '.mat']);

    y_behavior = y_behavior(y_behavior~=2);
    y_memory = y_memory(y_memory~=2);

    alls = alls + length(y_scene);
    behavs = behavs + length(y_behavior);
    mems = mems + length(y_memory);
    allsind(sbji) = length(y_scene);
    behavsind(sbji) = length(y_behavior);
    memsind(sbji) = length(y_memory);

end
