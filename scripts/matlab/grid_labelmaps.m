% This script creates labelmaps for the PASCAL-Part dataset where each 
% pixel label corresponds to its respective grid number. Grids are of size
% mxn. Total grid size is a function of input image size.
%         n
%    __ __ __ __
%   |__|__|__|__|
%   |__|__|__|__|
%   |__|__|__|__|
%   |__|__|__|__|   m    e.g. 8x4
%   |__|__|__|__|
%   |__|__|__|__|
%   |__|__|__|__|
%   |__|__|__|__|

%       Label Mask            Instance Mask            Label Map
%   =================     ===================     ===================
%   | 1 | 2 | 3 | 4 |     |      |-_-|      |     |      |2|3|      |
%   | 5 | 6 | 7 | 8 |     |   _ _|_ _|_ _   |     |   _ _|6|7|_ _   |
%   | 9 |10 |11 |12 |     |  / |       | \  |     |  / |10 | 11| \  |
%   |13 |14 |15 |16 |  +  | / /|       |\ \ |  =  | /13|14 | 15|16\ |
%   |17 |18 |19 |20 |     |/_/ |_ _ _ _| \_\|     |/_/ |18_|_19| \_\|
%   |21 |22 |23 |24 |     |    |       |    |     |    |22 | 23|    |
%   |25 |26 |27 |28 |     |    |       |    |     |    |26 | 27|    |
%   |29 |30 |31 |32 |     |    |  /\   |    |     |    |30 /\31|    |
%   =================     ===================     ===================

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
if ~exist(outputImDir, 'dir')
  mkdir(outputImDir);
  fileattrib(outputImDir,'+w','u');
end
if ~exist(outputSegDir, 'dir')
  mkdir(outputSegDir);
  fileattrib(outputSegDir,'+w','u');
end

% Grid size, mxn
m_grid = 8;
n_grid = 4;
gridMap = cell(m_grid, n_grid);

%% For each image
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
    % For each object instance
    for oo = 1:size(objects,2)
        [~,inst_mask,parts_mask,parts] = part_mat2map(objects{oo}, img, pimap, desired_pid);
        if (sum(parts_mask(:)) == 0)
           continue;
        end
        [croppedRGB,~,croppedPartsMask] = cropMask(img, inst_mask, parts_mask);
        labelMap = zeros(size(croppedPartsMask,1), size(croppedPartsMask,2), 'uint8');
        % For each object part
        for pp = 1:size(parts,1)
            binaryIm = uint8(croppedPartsMask == pimap{classID}(parts{pp}));
%             binary(binary == 1) = pimap{classID}(parts{pp});
            if (sum(binaryIm(:)) == 0)
                continue;
            end
            partBBox = boundingBox(binaryIm);
            % mxn grid, blocks 2, 3, 6 & 7 (ie rows 1 & 2, cols 2 & 3)
            if (strcmp(parts{pp}, 'head'))
                labelMask = maskGrid(partBBox, m_grid, n_grid, 1:2, 2:3);
            % mxn grid, blocks 10, 11, 14, 15, 18 & 19 (i.e. rows 3-5, cols 2 & 3)
            elseif (strcmp(parts{pp}, 'torso'))
                labelMask = maskGrid(partBBox, m_grid, n_grid, 3:5, 2:3);
            elseif (~isempty(regexp(parts{pp}, 'r[ul]arm', 'once')))
                labelMask = maskGrid(partBBox, m_grid, n_grid, 3:5, 1);
            elseif (~isempty(regexp(parts{pp}, 'l[ul]arm', 'once')))
                labelMask = maskGrid(partBBox, m_grid, n_grid, 3:5, 4);
            elseif (~isempty(regexp(parts{pp}, 'r[ul]leg', 'once')))
                labelMask = maskGrid(partBBox, m_grid, n_grid, 6:8, 2);
            elseif (~isempty(regexp(parts{pp}, 'l[ul]leg', 'once')))
                labelMask = maskGrid(partBBox, m_grid, n_grid, 6:8, 3);
            else
                % Do nothing; only considering 7 body parts
            end
            [r,c] = convertBoundingBox(partBBox, labelMap);
            labelMap(r, c) = labelMap(r,c) + binaryIm(r, c).*labelMask...
                                .*uint8(~labelMap(r,c));
        end
        % Save images
        segfile = fullfile(outputSegDir,imname);
        [~,basename,~] = fileparts(segfile);
        multiInstanceNameSeg = strcat(basename,'_',num2str(oo),'.png');
        multiInstanceNameRGB = strcat(basename,'_',num2str(oo),'.jpg');
        segfile = fullfile(outputSegDir,multiInstanceNameSeg);
        rgbfile = fullfile(outputImDir,multiInstanceNameRGB);
        imwrite(labelMap, cmap, segfile);
        %imwrite(croppedRGB, rgbfile);
    end
end