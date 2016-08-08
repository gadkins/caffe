import caffe

import os
import numpy as np
from PIL import Image
import scipy.io
import imagesets

import random

class PASCALPartSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL-Part
    one-at-a-time while reshaping the net to preserve dimensions.

    The labels follow the 21 class parts defined by

        Xianjie Chen, Roozbeh Mottaghi, Xiaobai Liu, Sanja Fidler, 
	Raquel Urtasun, Alan Yuille. Detect What You Can: Detecting
	and Representing Object using Holistic Models and Body Parts"
	(CVPR), 2014 
    
    This script focuses on the 'person' class only
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC dir (must contain 2010)
        - part_dir: path to PASCAL-Part annotations
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for PASCAL-Part semantic segmentation.

        example: params = dict(voc_dir="/path/to/PASCAL", split="val")
        """
        # config
        params = eval(self.param_str)
        self.voc_dir = params['voc_dir'] + '/VOC2010'
        self.part_dir = params['part_dir']
	self.obj_cls = params['obj_cls']
	self.part = params['part']
        self.split = params['split']
        self.mean = np.array((104.007, 116.669, 122.679), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # load labels and resolve inconsistencies by mapping to full 400 labels
        #self.labels_400 = [label.replace(' ','') for idx, label in np.genfromtxt(self.part_dir + '/labels.txt', delimiter=':', dtype=None)]
        #self.labels_59 = [label.replace(' ','') for idx, label in np.genfromtxt(self.part_dir + '/59_labels.txt', delimiter=':', dtype=None)]
        #for main_label, task_label in zip(('table', 'bedclothes', 'cloth'), ('diningtable', 'bedcloth', 'clothes')):
            #self.labels_59[self.labels_59.index(task_label)] = main_label

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

	#####################################################################
	"""
	This block of code selects the class-specific ImageSets, finds
	which of these samples have instances of the specified sub-part, and
	creates new part-specific, multi-instance imagesets. For example, 
	given a 'person' class and part 'torso', the code selects all 
	images containing instances of a torso by cross-referencing with 
	multi-instance image annotations (generated from Matlab scripts)  
	from segmentations/<obj_cls>/<part>/
	"""

        # load indices for class-specific images and labels
        split_f  = '{}/ImageSets/Main/{}_{}.txt'.format(self.voc_dir, self.obj_cls, self.split)

	# destination of train/val multi-instance ids
	obj_imgset_dir = '{}/ImageSets/{}'.format(self.voc_dir, self.obj_cls)
        if not os.path.exists(obj_imgset_dir):
    		os.makedirs(obj_imgset_dir)
        imgset_f = '{}/{}_{}.txt'.format(obj_imgset_dir, self.obj_cls, self.split)

	imagesets.create_obj_imgset(split_f, imgset_f)	# removes labels

	# path to multi-instance segmentation images
	multi_inst_dir = '{}/segmentations/{}/{}'.format(self.part_dir, self.obj_cls, self.part)
	# whole object imageset (e.g. 'person')
	obj_imgset_f = '{}/ImageSets/{}/{}_{}.txt'.format(self.voc_dir, 
		self.obj_cls, self.obj_cls, self.split)
	# single-part imageset (e.g. 'torso')
	destination = '{}/ImageSets/{}'.format(self.voc_dir, self.obj_cls)
	if not os.path.exists(destination):
    		os.makedirs(destination)
	part_imgset_f = '{}/{}_{}.txt'.format(destination, self.part, self.split)

	imagesets.create_part_imgset(obj_imgset_f, multi_inst_dir, part_imgset_f)

	#####################################################################

        self.indices = open(part_imgset_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
	"""
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
	"""
        #im = Image.open('{}/JPEGImages/{}.jpg'.format(self.voc_dir, idx))
	im = Image.open('{}/images/{}/{}/{}.jpg'.format(self.part_dir, self.obj_cls, 
		self.part, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_

    def load_label(self, idx):
	"""
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        The full 400 labels are translated to the 59 class task labels.
	"""
	"""
        label_400 = scipy.io.loadmat('{}/trainval/{}.mat'.format(self.part_dir, idx))['LabelMap']
        label = np.zeros_like(label_400, dtype=np.uint8)
        for idx, l in enumerate(self.labels_59):
            #idx_400 = self.labels_400.index(l) + 1
            #label[label_400 == idx_400] = idx + 1
        label = label[np.newaxis, ...]
	"""
	seg_im = Image.open('{}/segmentations/{}/{}/{}.jpg'.format(self.part_dir, self.obj_cls, 
		self.part, idx))
        label = np.array(seg_im, dtype=np.uint8)
        #label = label[:,:,::-1]
        #label -= self.mean
        #label = label.transpose((2,0,1))
	label = label[np.newaxis, ...]
        return label
