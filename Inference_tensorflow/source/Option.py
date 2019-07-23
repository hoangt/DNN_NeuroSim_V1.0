import time
import tensorflow as tf
import sys
import os
import os, sys, optparse

#==== Network architecture
network = 'vgg7'

#==== Quantization option
bitsW = 8   # bit width of weights
bitsA = 8   # bit width of activations
bitsG = 16  # bit width of gradients
bitsE = 16  # bit width of errors
bitsR = 16  # bit width of randomizer

#==== Training/Inference parameters
dataSet   = 'CIFAR10'
batchSize = 256
epoches   = 300

#==== Adaptive learning rate
lr          = tf.Variable(initial_value=0., trainable=False, name='lr', dtype=tf.float32)
lr_schedule = [0, 8, 200, 1, 250, 1./8, 300, 0]
L2          = 0

#==== Loss function
lossFunc   = 'SSE'
#lossFunc  = tf.losses.softmax_cross_entropy

#==== Optimizer
#optimizer  = tf.train.GradientDescentOptimizer(1)  # lr is controlled in Quantize.G
optimizer = tf.compat.v1.train.GradientDescentOptimizer(1)

#optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

#==== Misc
debug = False
Time  = time.strftime('%Y-%m-%d', time.localtime())
GPU   = [0]

seed    = 213123
#sess   = None
W_scale = []

modelName = network + '_{:d}_{:d}_{:d}_{:d}'.format(bitsW, bitsA, bitsG, bitsE)
modelDir  = os.path.join('../model', modelName)

loadModel = None
saveModel = os.path.join(modelDir, 'model.cpkt')
traceDir  = os.path.join('traces', modelName)

os.system('mkdir -p ' + modelDir)
os.system('mkdir -p ' + traceDir)

