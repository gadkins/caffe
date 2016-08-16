from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image

def fast_hist(a, b, n):
    # Select only positive samples less than # of classes (i.e. values 0 - 10 for 11-class fcn)
    k = (a >= 0) & (a < n) # a boolean index array 
    # Build an nxn confusion matrix where diagonal corresponds to correctly predicted class
    # Confusion matrix is indexed sequentially as [0,1,2,...n**2], thus every nth index corresponds
    # to the end of a row (hence the n *). The a[k] term is the gt while the b[k] value gives you 
    # a error (i.e. actual (y-axis) vs predicted (x-axis))
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    # N.B. channels refers to num_output pages for FCN
    n_cl = net.blobs[layer].channels
    if save_dir:    
    	os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))    # aka, confusion matrix
    loss = 0
    for idx in dataset:
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            im.save(os.path.join(save_dir, idx + '.png'))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, layer='score', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt)

def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    # mean loss
    print '>>>', datetime.now(), 'Iteration', iter, 'mean loss', loss
    # overall accuracy
    # hist.sum() should equal image height x width (i.e. 187,500 for a 375x500 pixel image)
    # as long as no gt are negative or outside the range of classes (see fast_hist)
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    # hist.sum(1) means to sum the 2nd dimension (i.e. columns, i.e. sum all cols of first row, sum all col 
    # of second row, etc.)
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    return hist
