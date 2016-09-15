function [croppedRGB,croppedInstMask,croppedPartMask] = cropMask(im, inst_mask, part_mask, outputSize)
% Given an 2D rgb image and a binary object instance mask, return an 
% cropped rgb image corresponding to the instance mask location and 
% outputSize is 2D vector of nonnegative integers

rect = boundingBox(inst_mask);
croppedRGB = imcrop(im, rect);
croppedInstMask = imcrop(inst_mask,rect);
croppedPartMask = imcrop(part_mask,rect);

end

