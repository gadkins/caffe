part = 'head'
obj = 'person'
split = 'val'
voc_dir = '../../data/pascal/VOC/VOC2010'
obj_imgset_f = '{}/ImageSets/{}/{}.txt'.format(voc_dir, obj, split)
img_dir = '../../data/pascal/pascal-part/segmentations/{}/{}/'.format(obj,part)

