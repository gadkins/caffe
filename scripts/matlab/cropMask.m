function [croppedRGB,croppedInstMask,croppedPartMask] = cropMask(im, inst_mask, part_mask, outputSize)
% Given an 2D rgb image and a binary object instance mask, return an 
% cropped rgb image corresponding to the instance mask location and 
% outputSize is 2D vector of nonnegative integers

if nargin < 4
    outputSize = size(inst_mask);
end

properties = regionprops('table', inst_mask,'BoundingBox', 'Centroid');
param = properties.BoundingBox;
[~,row] =  max(param(:,3));         % 3rd column corresponds to width
rect = param(row,:);

for ii = 1:numel(rect)
   rect(ii) = fix(rect(ii)); 
end

croppedRGB = imcrop(im, rect);
croppedRGB = imresize(croppedRGB,outputSize);
croppedInstMask = imcrop(inst_mask,rect);
croppedInstMask = imresize(croppedInstMask,outputSize);
croppedPartMask = imcrop(part_mask,rect);
croppedPartMask = imresize(croppedPartMask,outputSize);

end

