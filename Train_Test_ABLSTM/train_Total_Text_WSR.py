from __future__ import division
import numpy as np
import sys, os
from os.path import isfile, join
sys.path.insert(0, './caffe')
sys.path.insert(0, './caffe/python')
sys.path.insert(0, './model')
sys.path.insert(0, './lib')
import caffe

if not isfile('./model/TD_Total_Text_WSR_train.pt') or not isfile('./model/TD_Total_Text_WSR_test.pt') or not isfile('./model/TD_Total_Text_WSR_solver.pt'):
  from TD_Total_Text_WSR import make_all
  make_all()

# base_weights = 'model/TD_EKMV3_12S_iter_310000.caffemodel'
base_weights = './model/vgg16convs.caffemodel'
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('./model/TD_Total_Text_WSR_solver.pt')
solver.net.copy_from(base_weights)
for p in solver.net.params:
  param = solver.net.params[p]
  for i in range(len(param)):
    print p, "param[%d]: mean=%.5f, std=%.5f"%(i, solver.net.params[p][0].data.mean(), \
    solver.net.params[p][0].data.mean())
solver.solve()
