Before training, be sure to have images and segmetations appropriately separated.
See <caffe-root>/data/pascal/pascal-part/matlab/get_segmentations
 
To begin training, run init.sh which will exectue
<caffe-root>/models/pascalpart-fcn32s/person/<part-name>/net.py for initializing
which will create the train.prototxt and val.prototxt files

Once initialized, you should be able to begin training with train.sh (see script 
for usage details)
