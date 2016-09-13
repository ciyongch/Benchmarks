import theano
theano.config.floatX='float32'
import theano.tensor as T
import numpy as np
from googlenet_theano import googlenet, compile_models, set_learning_rate
import time
from datetime import datetime

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


def googlenet_train(batch_size=256, image_size=(3, 224, 224), n_epochs=60, mkldnn=False):
    
    input_shape = (batch_size,) + image_size
    model = googlenet(input_shape, mkldnn)

    (train_model, validate_model, train_error,
        shared_x, shared_y, shared_lr) = compile_models(model, batch_size = batch_size)

    images = np.random.random_integers(0, 255, input_shape).astype('float32')
    labels = np.random.random_integers(0, 999, batch_size).astype('int32')
    shared_x.set_value(images)
    shared_y.set_value(labels)
    iter = 0
    set_learning_rate(shared_lr, iter)

    model.set_dropout_off()
    time_theano_run(validate_model, 'Forward')
    model.set_dropout_on()
    time_theano_run(train_model, 'Forward-Backward')

if __name__ =='__main__':
    googlenet_train(batch_size=32, mkldnn=True)
