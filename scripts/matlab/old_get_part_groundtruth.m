% This script uses the PASCAL-Context dataset to get the groundtruth 
% (segmentations) of a given part. The following object classes and ids are:
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

classKeys = {'aeroplane','bicycle','bird','boat','bottle','bus','car',...
    'cat','chair','cow','diningtable','dog','horse','motorbike','person',...
    'pottedplant','sheep','sofa','train','tvmonitor'};
classValues = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
classMap = containers.Map(classKeys,classValues);
pimap = part2ind();     % part index mapping

% Set className and partName per your intended use
className = 'person';
classID = classMap(className);
partName = 'head+torso';

close all
% Input
anno_dir = '../Annotations_Part/';
anno_files = dir(strcat(anno_dir,'*.mat'));
img_path = '../../VOC/VOC2010/JPEGImages';
cmap = VOClabelcolormap();

% Output
outputRoot = '../';
outputImDir = fullfile(outputRoot,'images',className,partName);
outputSegDir = fullfile(outputRoot,'segmentations',className,partName);

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
    % load annotation -- anno
    objects = get_class_obj(anno, classID);
    if(isempty(objects))
        continue;
    end
    
    for oo = 1:size(objects,2)
        %parts = get_parts(objects{oo}, partName);
        if (~isempty(parts))
            [cls_mask, inst_mask, part_mask] = ...
                part_mat2map(img, pimap, objects, parts);
            
            segfile = fullfile(outputSegDir,imname);
            
            [~,basename,ext] = fileparts(segfile);
            multiInstanceName = strcat(basename,'_',num2str(oo),ext);
            segfile = fullfile(outputSegDir,multiInstanceName);
            imfile = fullfile(outputImDir,multiInstanceName);
            imwrite(img,imfile);
            imwrite(part_mask,segfile);
        end
    end
end