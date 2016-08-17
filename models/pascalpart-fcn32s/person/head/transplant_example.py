import caffe
import surgery

import numpy as np

context_proto = '../../../pascalcontext-fcn32s/train.prototxt'
context_weights = '../../../pascalcontext-fcn32s/pascalcontext-fcn32s-heavy.caffemodel'

solver = caffe.SGDSolver('solver.prototxt')

context_net = caffe.Net(context_proto, context_weights, caffe.TEST)
surgery.transplant(solver.net, context_net, suffix='part')
del context_net
