import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
    	pass

def get_batch(X,y,batch_size,index):
    batch_flag=0
    assert X.shape[0]==y.shape[0]
    n_batches = int(X.shape[0]/batch_size)

    return X[index*batch_size:(index+1)*batch_size],y[index*batch_size:(index+1)*batch_size]
