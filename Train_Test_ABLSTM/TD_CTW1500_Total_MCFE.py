# -*- coding: utf-8 -*-
import sys, os

sys.path.append("./caffe/python")
sys.path.append("./caffe/python/caffe")

import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
import numpy as np
from math import ceil

# 腐蚀系数1, Channel为128的洞孔卷积
def conv_dilation1(bottom, nout=128, dila=1, ks=3, stride=1, pad=1, mult=[1,1,2,0]):
  conv = L.Convolution(bottom, kernel_size=ks, dilation=dila, stride=stride,
    num_output=nout, pad=pad, weight_filler=dict(type='xavier'), 
    param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])], engine=1)
  return conv

# 腐蚀系数3, Channel为128的洞孔卷积
def conv_dilation3(bottom, nout=128, dila=3, ks=3, stride=1, pad=3, mult=[1,1,2,0]):
  conv = L.Convolution(bottom, kernel_size=ks, dilation=dila, stride=stride,
    num_output=nout, pad=pad, weight_filler=dict(type='xavier'), 
    param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])], engine=1)
  return conv

# 腐蚀系数5, Channel为128的洞孔卷积
def conv_dilation5(bottom, nout=128, dila=5, ks=3, stride=1, pad=5, mult=[1,1,2,0]):
  conv = L.Convolution(bottom, kernel_size=ks, dilation=dila, stride=stride,
    num_output=nout, pad=pad, weight_filler=dict(type='xavier'), 
    param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])], engine=1)
  return conv

# 腐蚀系数7, Channel为128的洞孔卷积
def conv_dilation7(bottom, nout=128, dila=7, ks=3, stride=1, pad=7, mult=[1,1,2,0]):
  conv = L.Convolution(bottom, kernel_size=ks, dilation=dila, stride=stride,
    num_output=nout, pad=pad, weight_filler=dict(type='xavier'), 
    param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])], engine=1)
  return conv  


# 腐蚀系数1, Channel为64的洞孔卷积
def conv_dilation01(bottom, nout=64, dila=1, ks=3, stride=1, pad=1, mult=[1,1,2,0]):
  conv = L.Convolution(bottom, kernel_size=ks, dilation=dila, stride=stride,
    num_output=nout, pad=pad, weight_filler=dict(type='xavier'), 
    param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])], engine=1)
  return conv

# 腐蚀系数3, Channel为64的洞孔卷积
def conv_dilation03(bottom, nout=64, dila=3, ks=3, stride=1, pad=3, mult=[1,1,2,0]):
  conv = L.Convolution(bottom, kernel_size=ks, dilation=dila, stride=stride,
    num_output=nout, pad=pad, weight_filler=dict(type='xavier'), 
    param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])], engine=1)
  return conv

# 腐蚀系数5, Channel为64的洞孔卷积
def conv_dilation05(bottom, nout=64, dila=5, ks=3, stride=1, pad=5, mult=[1,1,2,0]):
  conv = L.Convolution(bottom, kernel_size=ks, dilation=dila, stride=stride,
    num_output=nout, pad=pad, weight_filler=dict(type='xavier'), 
    param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])], engine=1)
  return conv

# 腐蚀系数7, Channel为64的洞孔卷积
def conv_dilation07(bottom, nout=64, dila=7, ks=3, stride=1, pad=7, mult=[1,1,2,0]):
  conv = L.Convolution(bottom, kernel_size=ks, dilation=dila, stride=stride,
    num_output=nout, pad=pad, weight_filler=dict(type='xavier'), 
    param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])], engine=1)
  return conv  


def conv_relu(bottom, nout, ks=3, stride=1, pad=1, mult=[1,1,2,0]):
  conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
    num_output=nout, pad=pad, weight_filler=dict(type='xavier'), 
    param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])], engine=1)
  return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
  return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def conv1x1(bottom, lr=[0.01, 1, 0.02, 0], wf=dict(type="constant")):
  return L.Convolution(bottom, kernel_size=1,num_output=1, weight_filler=wf,
      param=[dict(lr_mult=lr[0], decay_mult=lr[1]), dict(lr_mult=lr[2], decay_mult=lr[3])], engine=1)

# 上采样
def upsample(bottom, stride):
  s, k, pad = stride, 2 * stride, int(ceil(stride-1)/2)
  name = "upsample%d"%s
  return L.Deconvolution(bottom, name=name, convolution_param=dict(num_output=1, 
    kernel_size=k, stride=s, pad=pad, weight_filler = dict(type="bilinear")),
      param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])


def net(split):
  n = caffe.NetSpec()
  loss_param = dict(normalize=False)
  if split=='train':
    data_params = dict(mean=(104.00699, 116.66877, 122.67892))
    # 图像与标签

    data_params['root'] = './datasets/CTW1500_Total_TCB'
    data_params['source'] = "CTW1500_Total_TCB.lst"

    data_params['shuffle'] = True
    data_params['ignore_label'] = -1
    n.data, n.label = L.Python(module='pylayer_old', layer='ImageLabelmapDataLayer', ntop=2, \
    param_str=str(data_params))
    if data_params.has_key('ignore_label'):
      loss_param['ignore_label'] = int(data_params['ignore_label'])
  elif split == 'test':
    n.data = L.Input(name = 'data', input_param=dict(shape=dict(dim=[1,3,500,500])))
  else:
    raise Exception("Invalid phase")

# The first conv stage
  n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=1)
  n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
# # ===================== prepare lstm inputs =====================
  n.pool1 = max_pool(n.relu1_2)



# The second conv stage  
  n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
  n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
# # ===================== prepare lstm inputs =====================
  n.pool2 = max_pool(n.relu2_2)

# The third conv stage
  n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
  n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
  n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)

# # ===================== prepare lstm inputs =====================
  n.im2col_conv33 = L.Im2col(n.relu3_3, convolution_param=dict(kernel_size=3, pad=1))
  n.im2col_transpose_conv33 = L.Transpose(n.im2col_conv33, transpose_param =dict(dim=[3,2,0,1]))
  n.lstm_input_conv33 = L.Reshape(n.im2col_transpose_conv33, reshape_param =dict(shape=dict(dim=-1), axis=1, num_axes=2))
  n.lstm_conv33 = L.Lstm(n.lstm_input_conv33,lstm_param =dict(num_output=128,weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant'), clipping_threshold=1))
# ===================== rlstm ===================
  n.rlstm_input_conv33 = L.Reverse(n.lstm_input_conv33, name='lstm_reverse1_conv33', reverse_param =dict(axis=0))
  n.rlstm_output_conv33= L.Lstm(n.rlstm_input_conv33, name='rlstm_conv33', lstm_param =dict(num_output=128))
  n.rlstm_conv33 = L.Reverse(n.rlstm_output_conv33, name='lstm_reverse2_conv33', reverse_param =dict(axis=0))
# ===================== merge lstm_conv33 and rlstm_conv33
  n.merge_lstm_rlstm_conv33 = L.Concat(n.lstm_conv33, n.rlstm_conv33, concat_param=dict(axis=2))
  n.lstm_output_reshape_conv33 = L.Reshape(n.merge_lstm_rlstm_conv33, reshape_param=dict(shape=dict(dim=[-1,1]), axis=1, num_axes=1))
# transpose size of output as (N, C, H, W)
  n.lstm_output_conv33 = L.Transpose(n.lstm_output_reshape_conv33,transpose_param=dict(dim=[2,3,1,0]))
  n.pool3 = max_pool(n.relu3_3)


# The fourth conv stage
  n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
  n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
  n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)

# # ===================== prepare lstm inputs =====================
  n.im2col_conv43 = L.Im2col(n.relu4_3, convolution_param=dict(kernel_size=3, pad=1))
  n.im2col_transpose_conv43 = L.Transpose(n.im2col_conv43, transpose_param =dict(dim=[3,2,0,1]))
  n.lstm_input_conv43 = L.Reshape(n.im2col_transpose_conv43, reshape_param =dict(shape=dict(dim=-1), axis=1, num_axes=2))
  n.lstm_conv43 = L.Lstm(n.lstm_input_conv43,lstm_param =dict(num_output=256,weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant'), clipping_threshold=1))
# ===================== rlstm ===================
  n.rlstm_input_conv43 = L.Reverse(n.lstm_input_conv43, name='lstm_reverse1_conv43', reverse_param =dict(axis=0))
  n.rlstm_output_conv43= L.Lstm(n.rlstm_input_conv43, name='rlstm_conv43', lstm_param =dict(num_output=256))
  n.rlstm_conv43 = L.Reverse(n.rlstm_output_conv43, name='lstm_reverse2_conv43', reverse_param =dict(axis=0))
# ===================== merge lstm_conv43 and rlstm_conv43
  n.merge_lstm_rlstm_conv43 = L.Concat(n.lstm_conv43, n.rlstm_conv43, concat_param=dict(axis=2))
  n.lstm_output_reshape_conv43 = L.Reshape(n.merge_lstm_rlstm_conv43, reshape_param=dict(shape=dict(dim=[-1,1]), axis=1, num_axes=1))
# transpose size of output as (N, C, H, W)
  n.lstm_output_conv43 = L.Transpose(n.lstm_output_reshape_conv43,transpose_param=dict(dim=[2,3,1,0]))
  n.pool4 = max_pool(n.relu4_3)
  
  n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
  n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
  n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)

# The fiveth conv stage
# ===================== prepare lstm inputs =====================
  n.im2col_conv53 = L.Im2col(n.relu5_3, convolution_param=dict(kernel_size=3, pad=1))
  n.im2col_transpose_conv53 = L.Transpose(n.im2col_conv53, transpose_param =dict(dim=[3,2,0,1]))
  n.lstm_input_conv53 = L.Reshape(n.im2col_transpose_conv53, reshape_param =dict(shape=dict(dim=-1), axis=1, num_axes=2))
  n.lstm_conv53 = L.Lstm(n.lstm_input_conv53,lstm_param =dict(num_output=256,weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant'), clipping_threshold=1))
# ===================== rlstm ===================
  n.rlstm_input_conv53 = L.Reverse(n.lstm_input_conv53, name='lstm_reverse1_conv53', reverse_param =dict(axis=0))
  n.rlstm_output_conv53= L.Lstm(n.rlstm_input_conv53, name='rlstm_conv53', lstm_param =dict(num_output=256))
  n.rlstm_conv53 = L.Reverse(n.rlstm_output_conv53, name='lstm_reverse2_conv53', reverse_param =dict(axis=0))
# ===================== merge lstm_conv53 and rlstm_conv53
  n.merge_lstm_rlstm_conv53 = L.Concat(n.lstm_conv53, n.rlstm_conv53, concat_param=dict(axis=2))
  n.lstm_output_reshape_conv53 = L.Reshape(n.merge_lstm_rlstm_conv53, reshape_param=dict(shape=dict(dim=[-1,1]), axis=1, num_axes=1))
# transpose size of output as (N, C, H, W)
  n.lstm_output_conv53 = L.Transpose(n.lstm_output_reshape_conv53,transpose_param=dict(dim=[2,3,1,0]))

# # DSN1
# # DSN3
  n.score_dsn3 = conv1x1(n.lstm_output_conv33, lr=[0.01, 1, 0.02, 0], wf=dict(type='gaussian', std=0.01))
  n.score_dsn3_up = upsample(n.score_dsn3, stride=4)
  n.upscore_dsn3 = L.Crop(n.score_dsn3_up, n.data)

  if split=='train':
    n.loss3 = L.BalanceCrossEntropyLoss(n.upscore_dsn3, n.label, loss_param=loss_param)  
  if split=='test':
    n.sigmoid_dsn3 = L.Sigmoid(n.upscore_dsn3)  

# # DSN4
  n.score_dsn4 = conv1x1(n.lstm_output_conv43, lr=[0.01, 1, 0.02, 0], wf=dict(type='gaussian', std=0.01))
  n.score_dsn4_up = upsample(n.score_dsn4, stride=8)
  n.upscore_dsn4 = L.Crop(n.score_dsn4_up, n.data)

  if split=='train':
    n.loss4 = L.BalanceCrossEntropyLoss(n.upscore_dsn4, n.label, loss_param=loss_param)  
  if split=='test':
    n.sigmoid_dsn4 = L.Sigmoid(n.upscore_dsn4)

# DSN5
  n.score_dsn5 = conv1x1(n.lstm_output_conv53, lr=[0.01, 1, 0.02, 0], wf=dict(type='gaussian', std=0.01))
  n.score_dsn5_up = upsample(n.score_dsn5, stride=16)
  n.upscore_dsn5 = L.Crop(n.score_dsn5_up, n.data)


  if split=='train':
    n.loss5 = L.BalanceCrossEntropyLoss(n.upscore_dsn5, n.label, loss_param=loss_param)  
  if split=='test':
    n.sigmoid_dsn5 = L.Sigmoid(n.upscore_dsn5)    

# ############### concatenation and pass through attention model #########
  n.concat_upscore = L.Concat(n.upscore_dsn3,
                      n.upscore_dsn4,
                      n.upscore_dsn5,                      
                      name='concat', concat_param=dict({'concat_dim':1}))

  n.output_mask_product03 = L.Convolution(n.upscore_dsn3,
                 num_output=1, kernel_size=3,pad=1,
                 param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)], weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant'), engine=1)
  n.output_mask_product04 = L.Convolution(n.upscore_dsn4,
                 num_output=1, kernel_size=3,pad=1,
                 param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)], weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant'), engine=1)
  n.output_mask_product05 = L.Convolution(n.upscore_dsn5,
                 num_output=1, kernel_size=3,pad=1,
                 param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)], weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant'), engine=1)


### attention model
  n.att_conv1_mask_512 =  L.Convolution(n.concat_upscore,
                 num_output=512, kernel_size=3,pad=1,
                 param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)], engine=1)
  n.relu_att_conv1 = L.ReLU(n.att_conv1_mask_512, in_place=True)
  n.drop_att_conv1_mask = L.Dropout(n.relu_att_conv1, dropout_ratio=0.5, in_place=True)
  n.att_fc_mask_512 = L.Convolution(n.drop_att_conv1_mask,
                 num_output=3, kernel_size=1,
                 param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)], engine=1)
  n.attention = L.Softmax(n.att_fc_mask_512)
  n.attention3,n.attention4,n.attention5= L.Slice(n.attention, name='slice_attention', slice_param=dict(axis=1, slice_point=[1,2]), ntop=3)

# # ---- multiply attention weights ----
  n.output_mask3 = L.Eltwise(n.attention3, n.output_mask_product03,operation=P.Eltwise.PROD)
  n.output_mask4 = L.Eltwise(n.attention4, n.output_mask_product04,operation=P.Eltwise.PROD)
  n.output_mask5 = L.Eltwise(n.attention5, n.output_mask_product05,operation=P.Eltwise.PROD)  


  n.output_fusion = L.Eltwise(n.output_mask3, n.output_mask4, n.output_mask5, operation=P.Eltwise.SUM)
  n.upscore_fuse = L.Convolution(n.concat_upscore, name='new-score-weighting', 
                 num_output=1, kernel_size=1,
                 param=[dict(lr_mult=0.001, decay_mult=1), dict(lr_mult=0.002, decay_mult=0)],
                 weight_filler=dict(type='constant', value=0.2), engine=1)

  if split=='test':
    n.sigmoid_fuse = L.Sigmoid(n.upscore_fuse)
    n.sigmoid_output_fusion= L.Sigmoid(n.output_fusion)
  if split=='train':
    n.loss_fuse = L.BalanceCrossEntropyLoss(n.upscore_fuse, n.label, loss_param=loss_param)
    n.loss_output_fusion = L.BalanceCrossEntropyLoss(n.output_fusion, n.label, loss_param=loss_param) 
    # n.loss_fuse = L.BalanceCrossEntropyLoss(n.upscore_fuse, n.label, loss_param=loss_param)    
  return n.to_proto()

def make_net():
  with open('./model/TD_CTW1500_Total_MCFE_train.pt', 'w') as f:
    f.write(str(net('train')))
  with open('./model/TD_CTW1500_Total_MCFE_test.pt', 'w') as f:
    f.write(str(net('test')))
def make_solver():
  sp = {}
  sp['net'] = '"./model/TD_CTW1500_Total_MCFE_train.pt"'
  sp['base_lr'] = '0.000000001'
  sp['lr_policy'] = '"step"'
  sp['momentum'] = '0.9'
  sp['weight_decay'] = '0.0002'
  sp['iter_size'] = '5'
  sp['stepsize'] = '150000'
  sp['display'] = '10'
  sp['snapshot'] = '5000'
  sp['snapshot_prefix'] = '"./snapshot/TD_CTW1500_Total_MCFE/TD_CTW1500_Total_MCFE"'
  sp['gamma'] = '0.1'
  sp['max_iter'] = '400000'
  sp['solver_mode'] = 'GPU'
  f = open('./model/TD_CTW1500_Total_MCFE_solver.pt', 'w')
  for k, v in sorted(sp.items()):
      if not(type(v) is str):
          raise TypeError('All solver parameters must be strings')
      f.write('%s: %s\n'%(k, v))
  f.close()

def make_all():
  make_net()
  make_solver()

if __name__ == '__main__':
  make_all()
