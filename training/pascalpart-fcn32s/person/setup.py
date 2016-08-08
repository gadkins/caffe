import os

def training_setup(models, parts):
    for p in parts:
	seg_results = '{}/segmentation_results'.format(p)
	snapshot = '{}/snapshot'.format(p)
	model = '{}/model'.format(p)
	loss = '{}/loss'.format(p)
	log = '{}/log'.format(p)

	# setup useful directories
	if os.path.exists(seg_results):
	    raise Exception("Directory '%s' already exists." % seg_results)

	if not os.path.exists(snapshot):
	    os.symlink('{}/pascalpart-fcn32s/person/{}'.format(models, snapshot), snapshot)
	    os.symlink('{}/pascalpart-fcn32s/person/{}'.format(models, p), model)
	    os.mkdir(loss)
	    os.mkdir(log)

# initialize prototxt networks
def model_setup(parts):
    for p in parts:
        execfile("{}/model/net.py".format(p))
