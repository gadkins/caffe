import os

def remove_label(s):
    return s[0:-2]

def get_base(s):
    values = s.split(' ')
    return values[0]

# select only the positive/neutral samples for the obj class (i.e. gt 1 or 0)
def check_gt(s):
    values = s.split(' ')
    if values[-1] == '-1':
        return False
    else:
        return values[0]

def create_obj_imgset(split_f, imgset):
    ind = open(split_f, 'r').read().splitlines()
    imgset_h = open(imgset, 'w')
    for line in ind:
        label = check_gt(line)
	if label is not False:
	    imgset_h.write("%s\n" % label)
    imgset_h.close()

def create_part_imgset(obj_imgset_f, seg_dir, part_imgset_f):
    split_ids = open(obj_imgset_f, 'r').read().splitlines()
    images = [f for f in os.listdir(seg_dir)]
    imgset_f = open(part_imgset_f, 'w')
    for i in split_ids:
	for j in images:
	    if i in j:
		imgset_f.write("%s\n" % j[:-4])
    imgset_f.close()
