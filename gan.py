import numpy as np
import tensorflow as tf 
import pdb

def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # input : [bs, 1, 1, noise_size=100]
        x = tf.reshape(x, shape=[-1, 1, 1, 100])
        net=tf.layers.conv2d_transpose(x, 512, 4, strides=(1,1),use_bias=False)
        net=tf.layers.batch_normalization(net)
        net=tf.nn.relu(net)
        # output : [bs, 4, 4, 512]

        # input : [bs, 4, 4, 512]
        net=tf.layers.conv2d_transpose(net, 256, 4, strides=(2,2),padding='same',use_bias=False)
        net=tf.layers.batch_normalization(net)
        net=tf.nn.relu(net)
        # output : [bs, 8, 8, 256]

        # input : [bs, 8, 8, 256]
        net=tf.layers.conv2d_transpose(net, 128, 4, strides=(2,2),padding='same',use_bias=False)
        net=tf.layers.batch_normalization(net)
        net=tf.nn.relu(net)
        # output : [bs, 16, 16, 128]

        # input : [bs, 16, 16, 128]
        net=tf.layers.conv2d_transpose(net, 64, 4, strides=(2,2),padding='same',use_bias=False)
        net=tf.layers.batch_normalization(net)
        net=tf.nn.relu(net)
        # output : [bs, 32, 32, 64]

        # input : [bs, 32, 32, 64]
        net=tf.layers.conv2d_transpose(net, 3, 4, strides=(2,2),padding='same',use_bias=False)
        net=tf.nn.tanh(net)

        return net


def generator_vanilla(x, reuse=False):
    with tf.variable_scope('Generator_Vanilla', reuse=reuse):
        # input : [bx, nois_size=100]
        net=tf.contrib.layers.fully_connected(x,128, activation_fn=None)
        net=tf.layers.batch_normalization(net)
        net=tf.nn.leaky_relu(net, alpha=0.2)
        # output : [bs, 128]

        # input : [bs, 128]
        net=tf.contrib.layers.fully_connected(net, 256, activation_fn=None)
        net=tf.layers.batch_normalization(net)
        net=tf.nn.leaky_relu(net, alpha=0.2)
        # output : [bs, 256]

        # input : [bs, 256]
        net=tf.contrib.layers.fully_connected(net, 512, activation_fn=None)
        net=tf.layers.batch_normalization(net)
        net=tf.nn.leaky_relu(net, alpha=0.2)
        # output : [bs, 512]

        # input : [bs, 512]
        net=tf.contrib.layers.fully_connected(net, 1024, activation_fn=None)
        net=tf.layers.batch_normalization(net)
        net=tf.nn.leaky_relu(net, alpha=0.2)
        # output

        # input : [bs, 1024]
        net=tf.contrib.layers.fully_connected(net, 2048, activation_fn=None)
        net=tf.layers.batch_normalization(net)
        net=tf.nn.leaky_relu(net, alpha=0.2)
        # output : [bs, 2048]

        # input : [bs, 2048]
        net=tf.contrib.layers.fully_connected(net, 4096, activation_fn=None)
        net=tf.layers.batch_normalization(net)
        net=tf.nn.leaky_relu(net, alpha=0.2)
        # output : [bs, 4096]

        #input : [bs, 4096]
        net=tf.contrib.layers.fully_connected(net, 12288, activation_fn=None)
        net=tf.nn.tanh(net)

        net = tf.reshape(net, shape=[-1, 64, 64, 3])
        return net


    



def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):

        # input : [bs, 64, 64, 3]
        net=tf.layers.conv2d(x, 64, 4, strides=(2,2),padding='same',use_bias=False)
        net=tf.nn.leaky_relu(net, alpha=0.2)
        # output : [bs, 32, 32, 64]

        # intput : [bs, 32, 32, 64]
        net=tf.layers.conv2d(net, 128, 4, strides=(2,2),padding='same',use_bias=False)
        net=tf.layers.batch_normalization(net)
        net=tf.nn.leaky_relu(net, alpha=0.2)
        # output : [bs, 16, 16, 128]

        # intput : [bs, 16, 16, 128]
        net=tf.layers.conv2d(net, 256, 4, strides=(2,2),padding='same',use_bias=False)
        net=tf.layers.batch_normalization(net)
        net=tf.nn.leaky_relu(net, alpha=0.2)
        # output : [bs, 8, 8, 256]

        # intput : [bs, 8, 8, 256]
        net=tf.layers.conv2d(net, 512, 4, strides=(2,2),padding='same',use_bias=False)
        net=tf.layers.batch_normalization(net)
        net=tf.nn.leaky_relu(net, alpha=0.2)
        # output : [bs, 4, 4, 512]

        # input : [bs, 4, 4, 512]
        net=tf.layers.conv2d(net, 1, 4, strides=(1,1),use_bias=False)
        #net=tf.nn.sigmoid(net)
        return net



