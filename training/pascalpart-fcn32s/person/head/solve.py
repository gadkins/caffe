# Solve network from scratch or resume solving from specified iteration
# Script is called from train.sh (see usage note in that script)
import caffe
import surgery, score

import numpy as np
import os, sys
import time
import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))
import errno
import time

t = time.time()

caffe_root = '/home/cv/hdl/caffe'
models = '{}/models'.format(caffe_root)
part = 'head'

device = sys.argv[1]
if len(sys.argv) > 2:
	is_resume = sys.argv[2] == '-resume' and int(sys.argv[3])%4000 == 0
	if is_resume:
		iteration = int(sys.argv[3])

timestr = time.strftime("%Y%m%d-%H%M%S")
log = 'log'
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('{}/accuracy.log.{}'.format(log, timestr), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

# setup
sys.stdout = Logger()
seg_results = 'segmentation_results'
if os.path.exists(seg_results):
    raise Exception("Directory '%s' already exists." % seg_results)

snapshot = 'snapshot'
if not os.path.exists(snapshot):
    print 'Creating snapshot directory...\n'
    os.symlink('{}/pascalpart-fcn32s/person/{}/{}'.format(models, part, snapshot), snapshot)

model = 'model'
if not os.path.exists(model):
    print 'Creating model directory...\n'
    os.symlink('{}/pascalpart-fcn32s/person/{}'.format(models, part), model)

loss = 'loss'
if not os.path.exists(loss):
    print 'Creating loss directory...\n'
    os.mkdir(loss)

if not os.path.exists(log):
    print 'Creating training_log directory...\n'
    os.mkdir(log)

# init
caffe.set_device(int(device))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('{}/pascalpart-fcn32s/person/{}/solver.prototxt'.format(models, part))
if is_resume:
    solver.net.copy_from('{}/train_iter_{}.caffemodel'.format(snapshot, iteration))
    solver.restore('{}/train_iter_{}.solverstate'.format(snapshot, iteration))
else:
    solver.net.copy_from('../vgg_no_bilinear_vgg16fc.caffemodel')

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('{}/data/pascal/VOC/VOC2010/ImageSets/person/{}_val.txt'.format(caffe_root, part), dtype=str)
seg_results = '{}/models/pascalpart-fcn32s/person/{}/segmentation_results'.format(caffe_root, part)
for _ in range(10):
    solver.step(8000)
    score.seg_tests(solver, False, val, layer='score')

elapsed = time.time() - t
print 'Time elapse: %ds' % elapsed
