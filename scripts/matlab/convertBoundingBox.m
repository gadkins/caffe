function [r,c] = convertBoundingBox(bbox, labelMap)

r = bbox(2):bbox(2)+bbox(4)-1;
c = bbox(1):bbox(1)+bbox(3)-1;

% Ceck that bounding box does not exceed labelMap dimensions
if (max(r) > size(labelMap,1))
   r = r - (max(r) - size(labelMap,1));
end
if (max(c) > size(labelMap,2))
    c = c - (max(c) - size(labelMap,2));
end
if (min(r) == 0)
    r = 1+(r(1):r(end));
end
if (min(c) == 0)
    c = 1+(c(1):c(end));
end