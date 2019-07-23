import numpy as np
import time
import tensorflow as tf
import NN
import Option
import Log
import getData
import Quantize
from tqdm import tqdm
from hardware_estimation import hardware_estimation

#==== for single GPU quanzation
def quantizeGrads(Grad_and_vars):
    if Quantize.bitsG <= 16:
        grads = []
        for grad_and_vars in Grad_and_vars:
            grads.append([Quantize.G(grad_and_vars[0]), grad_and_vars[1]])
        return grads
    return Grad_and_vars

#====
def showVariable(keywords=None, debug=False):
    Vars = tf.global_variables()
    Vars_key = []
    for var in Vars:
        if debug:
            print('>>> Variables')
            print('  - devide  ' + str(var.device))
            print('  - name    ' + str(var.name))
            print('  - shape   ' + str(var.shape))
            print('  - dtype   ' + str(var.dtype))

        print var.device,var.name,var.shape,var.dtype
        if keywords is not None:
            if var.name.lower().find(keywords) > -1:
                Vars_key.append(var)
        else:
            Vars_key.append(var)
    return Vars_key

#====
def main():
  # get Option
    batchSize = Option.batchSize
    pathLog = os.path.join('../log', Option.modelName)
    os.system('mkdir -p ' + pathLog)
    logFile = os.path.join(pathLog, 'train.txt')
    fLog = open(logFile, 'w+')

    sys.stdout = Log.Log(fLog, sys.stdout) # set log file
    print('>>> Path to log file ' + pathLog)
    print('>>> Start at ' + time.strftime('%Y-%m-%d %X', time.localtime()) + '\n')
    open('Option.py').read()

    # get data
    numThread = 4*len(GPU)
    assert batchSize % len(GPU) == 0, ('batchSize must be divisible by number of GPUs')

    with tf.device('/cpu:0'):
      batchTrainX,batchTrainY,batchTestX,batchTestY,numTrain,numTest,label =\
          getData.loadData(Option.dataSet,batchSize,numThread)

    batchNumTrain = numTrain / batchSize
    batchNumTest = numTest / 100

    optimizer = Option.optimizer
    global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
    Net = []


    # on my machine, alexnet does not fit multi-GPU training
    # for single GPU
    with tf.device('/gpu:%d' % GPU[0]):
        Net.append(NN.NN(batchTestX, batchTestY, training=False))
        _, errorTestBatch = Net[-1].build_graph()

    showVariable(debug=Option.debug)

    # Build an initialization operation to run below.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = Option.sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    def getErrorTest():
        errorTest = 0.
        for i in tqdm(xrange(batchNumTest),desc = 'Test', leave=False):
            errorTest += sess.run([errorTestBatch])[0]
        errorTest /= batchNumTest
        return errorTest
    
    if Option.loadModel is not None:
        print('Loading model from %s ...' % Option.loadModel, end='')
        saver.restore(sess, Option.loadModel)
        print(' finished', end='')
        errorTestBest = getErrorTest()
        print('>>> Test error  %.3f' % errorTestBest)
        H, W = sess.run([Net[0].input_array, Net[0].W_b])
        hardware_estimation(H,W,Option.bitsW,Option.bitsA)

    else:
      print "No saved model"

if __name__ == '__main__':
  main()

