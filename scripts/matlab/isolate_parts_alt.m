% This script uses the PASCAL-Part dataset to isolate individual object
% parts given an object class id. The following object classes and ids are:
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

% Select a class and part
className = 'person';
classID = classMap(className);
partName = 'torso';

close all
% Input
anno_files = './Annotations_Part/%s.mat';
examples_path = './examples';
examples_imgs = dir([examples_path, '/', '*.jpg']);
cmap = VOClabelcolormap();

% Output
outputRoot = pwd;

outputImDir = fullfile(outputRoot,'images',className,partName);
outputSegDir = fullfile(outputRoot,'segmentations',className./,partName);
if ~exist(outputImDir, 'dir')
  mkdir(outputImDir);
  fileattrib(outputImDir,'+w','u');
end
if ~exist(outputSegDir, 'dir')
  mkdir(outputSegDir);
  fileattrib(outputSegDir,'+w','u');
end

for ii = 1:numel(examples_imgs)
    imname = examples_imgs(ii).name;
    img = imread([examples_path, '/', imname]);
    % load annotation -- anno
    load(sprintf(anno_files, imname(1:end-4)));
    objects = get_class_obj(anno, classID);
    if(isempty(objects))
        continue;
    end
    
    for oo = 1:size(objects,2)
        parts = get_parts(objects{oo}, partName);
        if (~isempty(parts))
            [cls_mask, inst_mask, part_mask] = ...
                part_mat2map(img, pimap, objects, parts);
            
            [croppedRGB,croppedMask] = cropPart(img,part_mask,[250,250]);
            
            % N.B. '_2.jpg' means there are at least two instances of
            % the object class, however, not necessarily two instances of
            % the part

            imfile = fullfile(outputImDir,imname);
            [~,basename,ext] = fileparts(imfile);
            multiInstanceName = strcat(basename,'_',num2str(oo),ext);
            imfile = fullfile(outputImDir,multiInstanceName);
            imwrite(croppedRGB,imfile);

            segfile = fullfile(outputSegDir,imname);
            [~,basename,ext] = fileparts(segfile);
            multiInstanceName = strcat(basename,'_',num2str(oo),ext);
            segfile = fullfile(outputSegDir,multiInstanceName);
            imwrite(croppedMask,segfile);

        end
    end
end