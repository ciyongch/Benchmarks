import theano
theano.config.floatX='float32'
import sys
import os
import time
import timeit
import theano.tensor as T
import numpy as np
import cPickle as pickle
from googlenet_theano import googlenet, compile_models, set_learning_rate
from params import get_params, load_net_state, load_params, save_net_state, save_figure
from read_lmdb import read_lmdb

def googlenet_train(batch_size=256, image_size=(3, 224, 224), n_epochs=1,mkldnn=False):

    #mean_path = '/home/2T/caffe/data/ilsvrc12/imagenet_mean.binaryproto'
    train_lmdb_path = '/data/imagenet/ilsvrc2012/lmdb/ilsvrc12_train_lmdb'
    val_lmdb_path = '/data/imagenet/ilsvrc2012/lmdb/lsvrc12_val_lmdb'

    input_shape = (batch_size,) + image_size
    model = googlenet(input_shape,mkldnn)

    #####read lmdb
    train_lmdb_iterator = read_lmdb(batch_size, train_lmdb_path)
    train_data_size = train_lmdb_iterator.total_number
    n_train_batches = train_data_size / batch_size
    print('n_train_batches = '+ str(n_train_batches))

    val_lmdb_iterator = read_lmdb(batch_size, val_lmdb_path)
    val_data_size = val_lmdb_iterator.total_number
    n_val_batches = val_data_size / batch_size
    print('n_val_batches = '+ str(n_val_batches))
    
    ## COMPILE FUNCTIONS ##
    (train_model, validate_model, train_error,
        shared_x, shared_y, shared_lr) = compile_models(model, batch_size = batch_size)

    all_costs = []
    all_errors = []

    ####load net state
    net_params = load_net_state()
    if True:
        '''
	load_params(model.params, net_params['model_params'])
        train_lmdb_iterator.set_cursor(net_params['minibatch_index'])
        all_costs = net_params['all_costs']
        all_errors = net_params['all_errors']
        epoch = net_params['epoch']
        minibatch_index = net_params['minibatch_index']
        '''
        epoch = 0
        minibatch_index = 0
        infp =file("mkldnn_googlenet_v1_bdw_iter_9600.pkl","rb")
        weight = pickle.load(infp)
        infp.close()
        # set weight
        print len(model.params)
        print len(weight)
        for i in xrange(0, len(model.params)):
            #print model.params[i].type
            #print weight[i].shape
            model.params[i].set_value(weight[i])
            #print 'set ',i
    else:
        all_costs = []
        all_errors = []
        epoch = 0
        minibatch_index = 0

    #model.set_dropout_off()
    cost_log_name='cost_log_mkl_'+str(mkldnn)
    print 'cost log ',cost_log_name
    cost_file = open(cost_log_name, 'w', 0)
    print('... training')
    while(epoch < n_epochs):
        minibatch_index = 0
        while(minibatch_index < n_train_batches):
	        ####training
            idx = epoch * n_train_batches + minibatch_index
            train_data, train_label = train_lmdb_iterator.next()
            shared_x.set_value(train_data)
            shared_y.set_value(train_label)
            set_learning_rate(shared_lr, idx)
            #begin_time = time.time()
            cost_ij = train_model()
            if idx % 40 == 0:
                cost_file.write(str(cost_ij)+'\n')
            print('iter %d, cost %f' %(idx, cost_ij))
            #end_time = time.time()
            #print('A iteration costs %fs' %(end_time-begin_time))
            minibatch_index += 1
        epoch = epoch + 1
    cost_file.close()

if __name__ =='__main__':
    googlenet_train(batch_size=32,mkldnn=True)
