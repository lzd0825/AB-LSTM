# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import os, sys, argparse
from os.path import isfile, join
sys.path.insert(0, '../caffe')
sys.path.insert(0, '../caffe/python')
sys.path.insert(0, '../model')
sys.path.insert(0, '../lib')
import caffe

import scipy.misc
import cv2
import scipy.io
from os.path import join, splitext, split, isfile
import time 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math 
from math import sqrt

parser = argparse.ArgumentParser(description='Forward all testing images.')
parser.add_argument('--model', type=str, default='../snapshot/TD_ICDAR2013_TCB/TD_ICDAR2013_TCB_iter_50000.caffemodel')
parser.add_argument('--net', type=str, default='../model/TD_ICDAR2013_TCB_test.pt')
parser.add_argument('--output', type=str, default='sigmoid_output_fusion') # output field
parser.add_argument('--gpu', type=int, default= 2)
parser.add_argument('--ms', type=bool, default=False) # Using multiscale
parser.add_argument('--savemat', type=bool, default=False) # whether save .mat
args = parser.parse_args()

def forward(data):
  assert data.ndim == 3
  data -= np.array((104.00698793,116.66876762,122.67891434))
  data = data.transpose((2, 0, 1))
  net.blobs['data'].reshape(1, *data.shape)
  net.blobs['data'].data[...] = data
  return net.forward()
assert isfile(args.model) and isfile(args.net), 'file not exists'
caffe.set_mode_gpu()
caffe.set_device(args.gpu)

net = caffe.Net(args.net, args.model, caffe.TEST)
# test_dir = '../data/test_datasets/ICDAR2013/' # test images directory
test_dir = '../data/test_datasets/MSRA-TD500/' # test images directory
test_dir_lst = test_dir.split('/')


if args.ms:
  save_dir = join('../data/test_results/', splitext(split(args.model)[1])[0]) # directory to save results
  save_dir = save_dir + '_ms_'+str(len(scales01))+'_'+test_dir_lst[len(test_dir_lst)-2]

else:
  save_dir = join('../data/test_results/', splitext(split(args.model)[1])[0]+'_s_sigmoid_fuse'+test_dir_lst[len(test_dir_lst)-2]) # directory to save results

if not os.path.exists(save_dir):
    os.makedirs(save_dir)   

imgs = [i for i in os.listdir(test_dir) if '.jpg' in i or '.JPG' in i]
imgs.sort()

nimgs = len(imgs)
print "totally "+str(nimgs)+"images"

# scale = 326400  
scale = 406400 

sf = 1
sf_factor = True

for i in range(nimgs):
  img = imgs[i]
  img = cv2.imread(join(test_dir, img)).astype(np.float32)
  h, w, _ = img.shape
  img_size = h*w
  
  if sf_factor == True:
    lamda = h*1.0/w
    wx = int(sqrt(scale/lamda)*sf)
    # hx = int(lamda*wx)  
  else:    
    lamda = h*1.0/w  
    wx = int(sqrt(scale/lamda))
    # hx = int(lamda*wx)

  if h*w>=scale:
    img = cv2.resize(img, (wx, int(scale/wx)))
  else:
    if sf_factor==True:
      img = cv2.resize(img, (int(w*sf), int(img_size/(w*sf))))
    else:
      pass

  # if h*w>=scale and h>=w:
  #   img = cv2.resize(img, (int(img.shape[1] * max_side / img.shape[0]), max_side))
  # elif h*w>=scale and h<w:
  #   img = cv2.resize(img, (max_side, int(img.shape[0] * max_side / img.shape[1])))


  if img.ndim == 2:
    img = img[:, :, np.newaxis]
    img = np.repeat(img, 3, 2)
  h, w, _ = img.shape
  edge = np.zeros((h, w), np.float32)
  if args.ms:
    scales = [0.5, 1, 1.5]
  else:
    scales = [1]
  for s in scales:
    h1, w1 = int(s * h), int(s * w)
    img1 = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    edge1 = np.squeeze(forward(img1)[args.output][0, 0, :, :])
    edge += cv2.resize(edge1, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
  edge /= len(scales)
  fn, ext = splitext(imgs[i])  
  plt.imsave(join(save_dir, fn+'.png'), edge)

  print "Saving to '" + join(save_dir, imgs[i][0:-4]) + "', Processing %d of %d..."%(i + 1, nimgs)