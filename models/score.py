from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
import matplotlib as plt

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, seg_dir, conf_dir, sm_dir, dataset, layer='score', gt='label'):
    # N.B. channels refers to num_output pages for FCN
    n_cl = net.blobs[layer].channels
    if seg_dir and not os.path.exists(seg_dir):    
    	os.mkdir(seg_dir)
    if conf_dir and not os.path.exists(conf_dir): 
	os.mkdir(conf_dir)
    if sm_dir and not os.path.exists(sm_dir): 
	os.mkdir(sm_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if seg_dir:
            #im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            #im.save(os.path.join(seg_dir, idx + '.png'))
	    plt.image.imsave(os.path.join(seg_dir, idx + '.png'), net.blobs[layer].data[0].argmax(0).astype(np.uint8))
	if conf_dir:
	    plt.image.imsave(os.path.join(conf_dir, idx + '.png'), net.blobs[layer].data[0].max(axis=0).astype(np.uint8))
	#if sm_dir:
	    #plt.image.imsave(os.path.join(sm_dir, idx + '.png'), net.blobs['loss'].data[0].max(axis=0))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)

def seg_tests(solver, save_format, conf_format, sm_format, dataset, layer='score', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, conf_format, sm_format, dataset, layer, gt)

def do_seg_tests(net, iter, save_format, conf_format, sm_format, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    if conf_format:
        conf_format = conf_format.format(iter)
    if sm_format:
        sm_format = sm_format.format(iter)
    hist, loss = compute_hist(net, save_format, conf_format, sm_format, dataset, layer, gt)
    # mean loss
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    return hist
