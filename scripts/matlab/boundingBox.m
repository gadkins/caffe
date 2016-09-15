function rect = boundingBox(mask)

properties = regionprops('table', mask,'BoundingBox', 'Centroid');
param = properties.BoundingBox;
[~,row] =  max(param(:,3));         % 3rd column corresponds to width
rect = param(row,:);

for ii = 1:numel(rect)
   rect(ii) = fix(rect(ii)); 
end