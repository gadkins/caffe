function [cls_mask, inst_mask, part_mask] = part_mat2map(img, pimap, objects, parts)
% Read the annotation and present it in terms of 3 segmentation mask maps (
% i.e., the class maks, instance maks and part mask). pimap defines a
% mapping between part name and index (See part2ind.m).


cls_mask = zeros(size(img,1), size(img,2), 'uint8');
inst_mask = zeros(size(img,1), size(img,2), 'uint8');
part_mask = zeros(size(img,1), size(img,2), 'uint8');

for oo = 1:size(objects,2)
    % The objects are ordered such that later objects can occlude previous
    % objects
    obj = objects{oo};
    class_ind = obj.class_ind;
    silh = obj.mask;            % the silhouette mask of the object
    assert(size(silh,1) == size(img,1) && size(silh,2) == size(img,2));
    inst_mask(silh) = oo; 
    cls_mask(silh) = class_ind;
    
    for pp = 1:size(parts,2)
        part_name = parts{pp}.part_name;
        assert(isKey(pimap{class_ind}, part_name));     % must define part index for every part
%         assert(all(silh(parts{pp}.mask)));          % all part region is a subregion of the object
        pid = pimap{class_ind}(part_name);
        part_mask(parts{pp}.mask) = pid;
    end
end