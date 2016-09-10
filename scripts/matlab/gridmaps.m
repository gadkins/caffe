% This script creates labelmaps for the PASCAL-Part dataset where each 
% pixel label corresponds to its respective grid number. Grids are of size
% mxn. The following are available object classes and ids are:
%
% aeroplane     -   1
% bicycle       -   2
% bird          -   3
% boat          -   4
% bottle        -   5
% bus           -   6
% car           -   7
% cat           -   8
% chair         -   9
% cow           -   10
% diningtable   -   11
% dog           -   12
% horse         -   13
% motorbike     -   14    
% person        -   15
% pottedplant   -   16
% sheep         -   17
% sofa          -   18
% train         -   19
% tvmonitor     -   20

dataRoot = '/home/cv/hdl/caffe/data';
pascal = fullfile(dataRoot, '/pascal');
addpath(genpath(pascal));

classKeys = {'aeroplane','bicycle','bird','boat','bottle','bus','car',...
    'cat','chair','cow','diningtable','dog','horse','motorbike','person',...
    'pottedplant','sheep','sofa','train','tvmonitor'};
classValues = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
classMap = containers.Map(classKeys,classValues);
pimap = part2ind();     % part index mapping

%% Set className and partName per your intended use
% See part2ind() for part names and pids
className = 'person';
classID = classMap(className);
partName = 'all_parts';
desired_pid = 0;        % Set to 0 use all parts

close all
% Input
% N.B. size of Annotations_Part and JPEGImages dirs are not equal.
% Only a subset (10,103) of JPEGImages are annotated by pascal parts

anno_dir = fullfile(pascal, '/pascal-part/Annotations_Part/');
anno_files = dir(strcat(anno_dir,'*.mat'));
img_path = fullfile(pascal, '/VOC/VOC2010/JPEGImages');
cmap = VOClabelcolormap();

% Output
outputRoot = fullfile(pascal, '/grid');
outputImDir = fullfile(outputRoot,'images',className,partName);
outputSegDir = fullfile(outputRoot,'segmentations',className,partName);

% Grid size, mxn
m_grid = 6;
n_grid = 4;
gridMap = cell(m_grid, n_grid);
blockSize = 50; % dxd

if ~exist(outputImDir, 'dir')
  mkdir(outputImDir);
  fileattrib(outputImDir,'+w','u');
end
if ~exist(outputSegDir, 'dir')
  mkdir(outputSegDir);
  fileattrib(outputSegDir,'+w','u');
end

for ii = 1:numel(anno_files)
    matname = anno_files(ii).name;
    load(strcat(anno_dir,matname));
    imname = strcat(matname(1:end-4),'.jpg');
    img = imread([img_path, '/', imname]);
    % cell array of objects matching our desired class only
    objects = get_class_obj(anno, classID);
    if(isempty(objects))
        continue;
    end
    for oo = 1:size(objects,2)
        [~,inst_mask,parts_mask,parts] = part_mat2map(objects{oo}, img, pimap, desired_pid);
        if (sum(part_mask(:)) == 0)
           continue;
        end
        [croppedRGB,~,croppedPartsMask] = cropMask(img, inst_mask, parts_mask);
        for pp = 1:size(parts,1)
            mask = croppedPartsMask == pimap{classID}(parts{pp});
            if parts{pp} == 'head'
                % apply grid over
            imshow(mask,cmap);
        end
    end
end