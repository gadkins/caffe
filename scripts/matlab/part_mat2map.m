function [cls_mask, inst_mask, part_mask] = part_mat2map(obj, img, pimap, desired_pid)
% obj is expected to be a single object from anno.objects()

cls_mask = zeros(size(img,1), size(img,2), 'uint8');
inst_mask = zeros(size(img,1), size(img,2), 'uint8');
part_mask = zeros(size(img,1), size(img,2), 'uint8');

class_ind = obj.class_ind;
silh = obj.mask;            % the silhouette mask of the object
assert(size(silh,1) == size(img,1) && size(silh,2) == size(img,2));
inst_mask(silh) = oo; 
cls_mask(silh) = class_ind;
    
for pp = 1:numel(obj.parts)
    part_name = obj.parts(pp).part_name;
    assert(isKey(pimap{class_ind}, part_name));     % must define part index for every part
    assert(all(silh(obj.parts(pp).mask)));          % all part region is a subregion of the object
    pid = pimap{class_ind}(part_name);
    if (pid == desired_pid)
        part_mask(obj.parts(pp).mask) = pid;
    end
end

end