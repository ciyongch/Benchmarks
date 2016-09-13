import sys
import time
from multiprocessing import Process, Queue

import yaml
import numpy as np
import zmq
import theano
import theano.tensor as T
from alex_net import AlexNet, compile_models
import logging
from datetime import datetime

from train_funcs import (unpack_configs, adjust_learning_rate,
                         get_val_error_loss, get_rand3d, train_model_wrap,
                         proc_configs)
sys.path.append('./lib')


def time_theano_run(func, info_string):
    num_batches = 50
    num_steps_burn_in = 10
    durations = []
    for i in xrange(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = func()
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print('%s: Iteration %d, %s, time: %.2f ms' %
                      (datetime.now(), i - num_steps_burn_in, info_string, duration*1000))
            durations.append(duration)
    durations = np.array(durations)
    print('%s: Average %s pass: %.2f ms ' %
          (datetime.now(), info_string, durations.mean()*1000))


def train_net(config):
    theano.config.on_unused_input = 'warn'

    image_sz = 227

    x = T.ftensor4('x')
    y = T.lvector('y')

    ## BUILD NETWORK ##
    model = AlexNet(config)
    layers = model.layers
    batch_size = model.batch_size
    errors = model.errors

    ## COMPILE FUNCTIONS ##
    (train_model, validate_model, train_error, learning_rate,
        shared_x, shared_y, rand_arr, vels) = compile_models(model, config, flag_top_5=True)

    images = np.random.rand(batch_size, 3, image_sz, image_sz).astype(np.float32)
    labels = np.random.randint(0, 1000, size=batch_size).astype(np.int32)

    shared_x.set_value(images)
    shared_y.set_value(labels)

    time_theano_run(validate_model, 'Forward')
    time_theano_run(train_model, 'Forward-Backward')



if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    with open('spec.yaml', 'r') as f:
        config = dict(config.items() + yaml.load(f).items())
        
    config = proc_configs(config)

    train_net(config)
