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
outputRoot = fullfile(pascal, '/pascal-part');
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
imageCount = 0;
standardSize = [333 333];
gridSize = 3;
grid = cell(gridSize);
sumImage = cell(gridSize);
meanImage = cell(gridSize);
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
        [~,inst_mask,part_mask,parts] = part_mat2map(objects{oo}, img, pimap, desired_pid);
        if isempty(parts)
            continue
        end
        [croppedRGB,~,croppedPartMask] = cropMask(img, inst_mask, part_mask);
        partRows = cellstr(char('head', 'torso', '[lr][ul]leg'));
%         rowCount = 0;
        for p=1:size(partRows,1)
            if ~isempty(find(~cellfun(@isempty,regexp(parts,partRows(p,:)))))
                rowCount = p;
            else
                continue
            end
        end
%         if rowCount <= 0
%             continue
%         end
        % A 3x3 block grid cell array is allocated for each person instance. 
        % If the instance has just a head and no torso or legs, only 1x3 blocks 
        % are added to grid. If person has head and torso, 2x3 blocks are 
        % added to grid. If head, torso and leg(s) are present, all nine 
        % blocks are added to 3x3 grid.
        imageCount = imageCount + 1;
        resizedRGB = imresize(croppedRGB, standardSize);
        [nRows,nCols,~] = size(resizedRGB);
        for i=1:rowCount
            for j=1:gridSize
                grid{i,j} = resizedRGB(1+(i-1)*nRows/gridSize:i*nRows/gridSize,1+(j-1)*nCols/gridSize:j*nCols/gridSize,:);
                if imageCount == 1
                    sumImage{i,j} = double(grid{i,j});
                else
                    sumImage{i,j} = double(sumImage{i,j}) + double(grid{i,j});
                end
            end
        end
%         figure
        subplot(3,3,1);imshow(grid{1,1});subplot(3,3,2);imshow(grid{1,2});subplot(3,3,3);imshow(grid{1,3});
        subplot(3,3,4);imshow(grid{2,1});subplot(3,3,5);imshow(grid{2,2});subplot(3,3,6);imshow(grid{2,3});
        subplot(3,3,7);imshow(grid{3,1});subplot(3,3,8);imshow(grid{3,2});subplot(3,3,9);imshow(grid{3,3});

%         subplot(3,3,1);imshow(sumImage{1,1});subplot(3,3,2);imshow(sumImage{1,2});subplot(3,3,3);imshow(sumImage{1,3});
%         subplot(3,3,4);imshow(sumImage{2,1});subplot(3,3,5);imshow(sumImage{2,2});subplot(3,3,6);imshow(sumImage{2,3});
%         subplot(3,3,7);imshow(sumImage{3,1});subplot(3,3,8);imshow(sumImage{3,2});subplot(3,3,9);imshow(sumImage{3,3});
%         fprintf('(paused) Press any key to continue\n');
%         pause
%         close all
        
    end
end

for i=1:3
    for j=1:3
        meanImage{i,j} = mat2gray(double(sumImage{i,j}) / imageCount);
    end
end
combinedIm = [meanImage{1,1},meanImage{1,2},meanImage{1,3};meanImage{2,1},meanImage{2,2},meanImage{2,3};meanImage{3,1},meanImage{3,2},meanImage{3,3}];
figure
subplot(3,3,1);imshow(meanImage{1,1});subplot(3,3,2);imshow(meanImage{1,2});subplot(3,3,3);imshow(meanImage{1,3});
subplot(3,3,4);imshow(meanImage{2,1});subplot(3,3,5);imshow(meanImage{2,2});subplot(3,3,6);imshow(meanImage{2,3});
subplot(3,3,7);imshow(meanImage{3,1});subplot(3,3,8);imshow(meanImage{3,2});subplot(3,3,9);imshow(meanImage{3,3});
figure
imshow(combinedIm);