 function parts = get_parts(obj, desired_part)
% Given an object, find all parts matching
% string desired_part. Returns a one-dimensional array of parts

if nargin < 2
    desired_part = '';      % all parts
end

num_parts = numel(obj.parts);
parts = {};

if (isempty(desired_part))
    for pp = 1:num_parts
        parts{pp} = obj.parts(pp);
    end
else
    for pp = 1:num_parts
        part = obj.parts(pp);
        if (strcmp(desired_part, part.part_name))
            parts{pp} = part;
        end
    end
    parts = parts(~cellfun('isempty',parts));
end
    