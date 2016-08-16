function objects = get_class_obj(anno, desired_class)
% Given struct anno, return all struct objects matching a given class
% If a given class of objects do not exist in anno, return empty objects

if nargin < 2
    desired_class = 0;      % all classes
else
    assert(desired_class>=1 && desired_class<=20);
end
objects = {};

num_obj = numel(anno.objects);

if (desired_class == 0)
    for oo = 1:num_obj
        obj = anno.objects(oo);
        objects{oo} = obj;
    end
else
    for oo = 1:num_obj
        if (desired_class == anno.objects(oo).class_ind)
            objects{oo} = anno.objects(oo);
        end
    end
    objects = objects(~cellfun('isempty',objects)); 
end