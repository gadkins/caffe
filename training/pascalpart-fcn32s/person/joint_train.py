import numpy as np
from PIL import Image
import caffe
import setproctitle
import os, sys
import surgery, score
import tools
from copy import copy
import time
import setup

setproctitle.setproctitle(os.path.basename(os.getcwd()))

caffe_root = '/home/cv/hdl/caffe'
models = '{}/models'.format(caffe_root)
voc_dir = '{}/data/pascal/VOC/VOC2010'.format(caffe_root)
snapshot = 'snapshot'
part1 = 'head'
part2 = 'torso'
joint_parts = 'head+torso'
parts = [part1, part2, joint_parts]
weights = 'vgg16fc.caffemodel'
classes = np.asarray(['background', 'head', 'torso', 'head+torso', 'left arm', 'right arm', 'arms', 'left leg', 'right leg', 'legs', 'person'])

device = sys.argv[1]
if len(sys.argv) > 2:
	is_resume = sys.argv[2] == '-resume' and int(sys.argv[3])%4000 == 0
	if is_resume:
		iteration = int(sys.argv[3])

timestr = time.strftime("%Y%m%d-%H%M%S")
#log = 'training_log'
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('accuracy.log.{}'.format(timestr), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

def get_caffemodel(part, iteration):
    return '{}/training_log/train_iter_{}'.format(part, iteration)

def get_solver(part):
    return caffe.SGDSolver('{}/model/solver.prototxt'.format(part))

def load_test_net(part, caffemodel):
    return caffe.Net('{}/model/deploy.prototxt'.format(part), caffemodel, caffe.TEST)

def do_surgery(solver):
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    surgery.interp(solver.net, interp_layers)

def resume_training(solvers, iteration):
    for p in parts:
        solver.net.copy_from('{}/{}/train_iter_{}.caffemodel'.format(p, snapshot, iteration))
        solver.restore('{}/{}/train_iter_{}.solverstate'.format(p, snapshot, iteration))
    
# expects 'params' to be a list of layer names
# expects source and destination network architectures to be identical
def copy_params(src_net, dest_net, params):
    # src_params = {layer_name: (weights, biases)}  
    src_params = {pr: (src_net.params[pr][0].data, src_net.params[pr][1].data) for pr in params}
    # dest_params = {layer_name: (weights, biases)}
    dest_params = {pr: (dest_net.params[pr][0].data, dest_net.params[pr][1].data) for pr in params}
    # copy
    for pr in params:
        dest_params[pr][0].flat = src_params[pr][0].flat  # flat unrolls the arrays
        dest_params[pr][1][...] = src_params[pr][1]
        
# expects network architectures to be identical
def compute_joint_avg(net1, net2, net3):
    layers = [l for l in net1.params.keys() if 'up' not in l]
    net1_params = {l: (net1.params[l][0].data, net1.params[l][1].data) for l in layers}
    net2_params = {l: (net2.params[l][0].data, net2.params[l][1].data) for l in layers}
    net3_params = {l: (net3.params[l][0].data, net3.params[l][1].data) for l in layers}
    # 'up' layers only have one param (i.e. net.params['upscore'][0].data)
    up_layers = [l for l in net1.params.keys() if 'up' in l]
#     for up in up_layers:
#         net1_params[up] = (net1.params[up][0].data,)
#         net2_params[up] = (net2.params[up][0].data,)
#         net3_params[up] = (net3.params[up][0].data,)
    for l in layers:
        combined_weights = np.concatenate((net1_params[l][0].flat, net2_params[l][0].flat, 
                                       net3_params[l][0].flat), axis=0).reshape([-1, np.size(net1_params[l][0])])
        net3_params[l][0].flat = np.mean(combined_weights, axis=0)
        combined_bias = np.concatenate((net1_params[l][1][...], net2_params[l][1][...], 
                                       net3_params[l][1][...]), axis=0).reshape([-1, np.size(net1_params[l][1][...])])
        net3_params[l][1][...] = np.mean(combined_bias, axis=0)
#     for up in up_layers:
#         combined_upscore = np.concatenate((net1_params[up][0].flat, net2_params[up][0].flat, net3_params[up][0].flat),
#                                          axis=0).reshape([-1, np.size(net1_params[up][0])])
#         net3_params[up][0].flat = np.mean(combined_upscore, axis=0)


# setup
sys.stdout = Logger()

# setup
t = time.time()
setup.training_setup(models, parts)
setup.model_setup(parts)

# init
caffe.set_device(0)
caffe.set_mode_gpu()
    
# load solvers
net_1_solver = get_solver(part1)
net_2_solver = get_solver(part2)
net_joint_solver = get_solver(joint_parts)

# initialize weights from VGG
# N.B. weights and biases will not be visibile when 'print net.params['conv1_1'][0].data' is called
# Not sure why this is atm. Only appear when caffe.Net('deploy.prototxt', 'vgg16fc.caffemodel', caffe.TEST) is
# used
net_1_solver.net.copy_from(weights)
net_2_solver.net.copy_from(weights)
net_joint_solver.net.copy_from(weights)

net_1_solver.test_nets[0].share_with(net_1_solver.net)
net_2_solver.test_nets[0].share_with(net_2_solver.net)
net_joint_solver.test_nets[0].share_with(net_joint_solver.net)

val_joint = np.loadtxt('{}/ImageSets/person/{}_val.txt'.format(voc_dir, joint_parts), dtype=str)
seg_results = '{}/model/segmentation_results'.format(joint_parts)

for _ in range(800):
    net_1_solver.step(100)
    net_2_solver.step(100)
    net_joint_solver.step(100)
    compute_joint_avg(net_1, net_2, net_joint)
#     score.seg_tests(net_joint_solver, False, val_joint, layer='score')

elapsed = time.time() - t
print 'Time elapse: %ds' % elapsed
