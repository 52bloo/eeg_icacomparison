function [] = check_create_folder(path_str)
%CHECK_CREATE_FOLDER Summary of this function goes here
%   Detailed explanation goes here

if ~exist(path_str, 'dir')
  [parentdir, newdir]=fileparts(path_str);
  [status,msg]= mkdir(parentdir, newdir);
  if status~=1
    error(msg);
  end
end

end

