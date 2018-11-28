""" Using convolutional net on CIFAR10 dataset
One conv layer and one fc layer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.keras.datasets.cifar10 import *
from tensorflow.keras.utils import to_categorical
import utils
from utils import get_batch
import tensorflow as tf

N_CLASSES = 10  # there are only 10 classes in cifar10 dataset

# Step 1: Read in data
# using TF Learn's built in function to load Cifar10 data to the folder data/mnist
(X_train, y_train), (X_test, y_test) = load_data()

# One-hot encode the label for input
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 2: Define parameters for the model
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 10

# Step 3: create placeholders for features and labels
# each image in the CIFAR10 dataset is of shape 32*32*3 = 1024*3 = 3072
# therefore, each image is represented with a [1,32,32,3] tensor
# We'll be doing dropout for hidden layer so we'll need a placeholder
# for the dropout probability too
# Use None for shape so we can change the batch_size once we've built the graph

with tf.name_scope('Input_data'):
    X = tf.placeholder(tf.float32, [None, 32, 32, 3], 'X_placeholder')
    Y = tf.placeholder(tf.float32, [None, 10], name="Y_placeholder")
dropout = tf.placeholder(tf.float32, name='dropout')

# Step 4 + 5: create weights + do inference
"""
Step 4&5 : Create weights and do inference
Model: conv1 -> relu -> pool -> fully connected -> softmax
Loss: cross entropy and L2 loss
"""

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

with tf.variable_scope('conv1') as scope:
    # first, reshape the image to [BATCH_SIZE, 28, 28, 1] to make it work with tf.nn.conv2d
    tf.summary.image(name='input_image', tensor=X, max_outputs=6)
    kernel = tf.get_variable('kernel', [5, 5, 3, 32],
                             initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable('biases', [32],
                             initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(X, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv + biases, name=scope.name)

    # output is of dimension BATCH_SIZE x 28 x 28 x 32
    # conv1 = layers.conv2d(X, 32, 5, 3, activation_fn=tf.nn.relu, padding='SAME')

with tf.variable_scope('pool1') as scope:
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

with tf.variable_scope('fc') as scope:  # use weight of dimension 16*16*32 x 1024
    input_features = 16 * 16 * 32
    w = tf.get_variable('weights', [input_features, 1024],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [1024],
                        initializer=tf.constant_initializer(0.0))
    reshaped_pool1 = tf.reshape(pool1, [-1, input_features])
    fc = tf.nn.relu(tf.add(tf.matmul(reshaped_pool1, w), b, name='Wx_plus_b'), name='relu')
    fc = tf.nn.dropout(fc, dropout, name='relu_dropout')

with tf.variable_scope('softmax_linear') as scope:
    w = tf.get_variable('weights', [1024, N_CLASSES],
                        initializer=tf.truncated_normal_initializer())

    b = tf.get_variable('biases', [N_CLASSES],
                        initializer=tf.random_normal_initializer())

    logits = tf.matmul(fc, w) + b

with tf.variable_scope('prediction_while_training'):
    train_preds_op = tf.argmax(logits, axis=1)

with tf.name_scope('cross_entropy_loss'):
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits)
    cross_entropy_loss = tf.reduce_mean(entropy, name='cross_entropy_loss')

with tf.name_scope('l2_loss'):
    pass

with tf.name_scope('summaries'):
    # This if for showing the test output image
    output_image = tf.slice(input_=conv1, begin=[0, 0, 0, 0], size=[-1, -1, -1, 3], name='slice')
    tf.summary.image('output_image', output_image,
                     max_outputs=6)
    summary_op = tf.summary.merge_all()

# Define training operation
# using gradient descent with learning rate of 0.001 to minimize cost
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=cross_entropy_loss,
                                                                         global_step=global_step)
# ckpts dir
utils.make_dir('checkpoints')
utils.make_dir('checkpoints/cifar10')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # To visualize using  TensorBoard
    writer = tf.summary.FileWriter('./graphs/cifar10', sess.graph)
    # to start the tensorboard
    # " tensorboard --logdir=graphs/convnet/ " at terminal

    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/cifar10/checkpoint'))

    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    initial_step = global_step.eval()  # global_step.eval() is "0"

    start_time = time.time()

    n_batches = int(X_train.shape[0] / BATCH_SIZE)
    batch_flag = 0

    total_loss = 0.0

    for index in range(initial_step, n_batches * N_EPOCHS):  # train the model N_EPOCH times

        X_batch, Y_batch = get_batch(X_train, y_train, batch_size=BATCH_SIZE, index=batch_flag)

        # control the data reader
        if batch_flag >= n_batches:
            batch_flag = 0
        batch_flag += 1

        _, loss_batch, summary = sess.run([optimizer, cross_entropy_loss, summary_op],
                                          feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT})

        writer.add_summary(summary,global_step=index)
        total_loss += loss_batch

        if (index+1)%SKIP_STEP ==0:
            print("Average loss at step {}:{:5.1f}".format(index+1,total_loss/SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/cifar10/cifar10-convnet', index)

    print("Optimization Finished")#
    print("Total time: {0} seconds".format(time.time()-start_time))

