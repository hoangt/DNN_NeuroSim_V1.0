import os, sys
import tensorflow as tf
import numpy as np
import Option

#====
def preprocess(x, 
               train=False):
    dataSet = Option.dataSet
    if dataSet == 'CIFAR10':
        if train:
            x = tf.image.resize_image_with_crop_or_pad(x, 40, 40)
            x = tf.random_crop(x, [32, 32, 3])
            x = tf.image.random_flip_left_right(x)
    else:
        print('>>> Unknown dataset ',dataSet,' no preprocess')
    x = tf.transpose(x, [2, 0, 1]) # from HWC to CHW
    return x

#====
def loadData(dataSet, batchSize, numThreads, seed=None):
    pathNPZ = '../dataSet/' + dataSet + '.npz'
    if os.path.isfile(pathNPZ):
        print('>>> Load dataset ' + pathNPZ)
    else:
        print('>>> Dataset ' + pathNPZ + ' is not found')
        sys.exit(1)

    numpyTrainX, numpyTrainY, numpyTestX, numpyTestY, label = loadNPZ(pathNPZ)
    numTrain = numpyTrainX.shape[0]
    numTest  = numpyTestX.shape[0]

    #trainX, trainY = data2Queue(numpyTrainX, numpyTrainY, batchSize,numThread, True,True)
    #testX, testY = data2Queue(numpyTestX, numpyTestY, 100, 1, False,False)
    trainX, trainY = data2Queue(dataX      = numpyTrainX, 
                                dataY      = numpyTrainY, 
                                batchSize  = batchSize, 
                                numThreads = numThreads, 
                                shuffle    = True,
                                isTraining = True, 
                                seed       = seed)

    testX, testY   = data2Queue(dataX      = numpyTestX,
                                dataY      = numpyTestY,  
                                batchSize  = 100,       
                                numThreads = numThreads,           
                                shuffle    = False,  
                                isTraining = False)

    print('>>> Dataset size')
    print('  - trainX   %s' % str(np.shape(trainX)))
    print('  - trainY   %s' % str(np.shape(trainY)))

    return trainX, trainY, testX, testY, numTrain, numTest, label

#====
# get dataset from NPZ files
def loadNPZ(pathNPZ):
    data   = np.load(pathNPZ)
    trainX = data['trainX']
    trainY = data['trainY']
    testX  = data['testX']
    testY  = data['testY']
    label  = data['label']
    return trainX, trainY, testX, testY, label

#====
def data2Queue(dataX, dataY, batchSize, numThreads, 
               shuffle=False, isTraining=True, seed=None):

    q = tf.FIFOQueue(capacity = dataX.shape[0], 
                     dtypes   = [dataX.dtype, dataY.dtype],
                     shapes   = [dataX.shape[1:],dataY.shape[1:]])

    enqueue_op = q.enqueue_many([dataX, dataY])
    sampleX, sampleY = q.dequeue()
    qRunner = tf.train.QueueRunner(q, [enqueue_op])
    tf.train.add_queue_runner(qRunner)
    sampleX_ = preprocess(sampleX, isTraining)

    #----
    if shuffle:
        batchX, batchY = tf.train.shuffle_batch([sampleX_, sampleY],
                                                batch_size        = batchSize,
                                                num_threads       = numThreads, 
                                                capacity          = int(dataX.shape[0]),
                                                min_after_dequeue = int(dataX.shape[0] / 2),
                                                seed              = seed)
    else:
        batchX, batchY = tf.train.batch([sampleX_, sampleY],
                                        batch_size  = batchSize,
                                        num_threads = numThreads,
                                        capacity    = int(dataX.shape[0]))

    return batchX, batchY


