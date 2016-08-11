# Helpful functions for training

import caffe 
import numpy as np
import surgery

def get_caffemodel(part, iteration):
    return '{}/snapshot/train_iter_{}.caffemodel'.format(part, iteration)

def get_net(part):
    return caffe.Net('{}/train.prototxt'.format(part), caffe.TEST)

def get_solver(part):
    return caffe.SGDSolver('{}/model/solver.prototxt'.format(part))

def load_test_net(part, caffemodel):
    return caffe.Net('{}/model/deploy.prototxt'.format(part), caffemodel, caffe.TEST)

def do_surgery(solver):
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    surgery.interp(solver.net, interp_layers)

def resume_training(part, solver, iteration):
#     for p in parts:
#         solver.net.copy_from('{}/{}/train_iter_{}.caffemodel'.format(p, snapshot, iteration))
#         solver.restore('{}/{}/train_iter_{}.solverstate'.format(p, snapshot, iteration))
    solver.net.copy_from('{}/{}/train_iter_{}.caffemodel'.format(part, snapshot, iteration))
    solver.restore('{}/{}/train_iter_{}.solverstate'.format(part, snapshot, iteration))
    
# expects 'params' to be a list of layer names
# expects source and destination network architectures to be identical and the following in dict forms:
# src_params = {layer_name: (weights, biases)}  
# dest_params = {layer_name: (weights, biases)}
def copy_params(src_params, dest_params, params):
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
#     up_layers = [l for l in net1.params.keys() if 'up' in l]
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

def max_params(net1, net2, layers):
    net1_params = {l: (net1.params[l][0].data, net1.params[l][1].data) for l in layers}
    net2_params = {l: (net2.params[l][0].data, net2.params[l][1].data) for l in layers}
    max_params = {}
    for l in layers:
        max_weights = np.maximum(net1_params[l][0].flat, net2_params[l][0].flat)
        # choose biases corresponding to max weight, not max bias
#         max_net = np.concatenate((net1_params[l][0].flat, net2_params[l][0].flat), 
#                                axis=0).reshape([-1, np.size(net1_params[l][0])]).argmax(axis=0)
        max_biases = np.maximum(net1_params[l][1][...], net2_params[l][1][...])
#         for m in max_net:
#             if m is 0:
#                 biases = net1_params[l][1][...]
#             else:
#                 biases = net2_params[l][1][...]
        max_params[l] = (max_weights, max_biases)
    return max_params
