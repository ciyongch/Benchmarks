import theano
import sys
import os
import time
import theano.tensor as T
import numpy as np
import cPickle as pickle
from googlenet_theano import googlenet, compile_models, set_learning_rate
from params import get_params, load_net_state, load_params, save_net_state, save_figure
from read_lmdb import read_lmdb

def googlenet_train(batch_size=256, image_size=(3, 224, 224), n_epochs=1, mkldnn=False):

    train_lmdb_path = '/path/to/your/imagenet/ilsvrc2012/ilsvrc12_train_lmdb'
    val_lmdb_path = '/path/to/your/imagenet/ilsvrc2012/ilsvrc12_val_lmdb'

    input_shape = (batch_size,) + image_size
    model = googlenet(input_shape, mkldnn)

    ##### get training and validation data from lmdb file
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
    if net_params:
        load_params(model.params, net_params['model_params'])
        train_lmdb_iterator.set_cursor(net_params['minibatch_index'])
        all_errors = net_params['all_errors']
        all_costs = net_params['all_costs']
        epoch = net_params['epoch']
        minibatch_index = net_params['minibatch_index']
    else:
        all_costs = []
        all_errors = []
        epoch = 0
        minibatch_index = 0

    print('... training')
    while(epoch < n_epochs):
        minibatch_index = 0
        count = 0
        while(minibatch_index < n_train_batches):
            count = count + 1
            if count == 1:
                s = time.time()
            if count == 20:
                e = time.time()
                print "time per 20 iter:", (e - s)

	        ####training
            idx = epoch * n_train_batches + minibatch_index
            train_data, train_label = train_lmdb_iterator.next()
            shared_x.set_value(train_data)
            shared_y.set_value(train_label)
            set_learning_rate(shared_lr, idx)
            cost_ij = train_model()
            if idx % 40 == 0:
                print('iter %d, cost %f' %(idx, cost_ij))
            minibatch_index += 1
        epoch = epoch + 1

if __name__ =='__main__':
    googlenet_train(batch_size=32,mkldnn=True)
