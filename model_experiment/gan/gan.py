import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.gridspec as gridspec

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'

get_ipython().magic('matplotlib inline')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./datasets/MNIST_data', one_hot=False)

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

def leaky_relu(x , rate=0.01):
    return tf.maximum(x , x*rate)

def sample_noise(batch_size, dim):
    #return tf.random_uniform([batch_size,dim], minval=-1, maxval=1, dtype=tf.float32)
    return np.random.uniform(low=-1 , high=1 , size=[batch_size,dim])

def generator(z):
    with tf.variable_scope('generator' , reuse=tf.AUTO_REUSE):
        #z~[batch_szie,96]
        #out~[batch_size,784]
        h1 = tf.layers.dense(inputs=z , units=1024 , activation=tf.nn.relu , use_bias=True)
        h2 = tf.layers.dense(inputs=h1 , units=1024 , activation=tf.nn.relu , use_bias=True)
        h3 = tf.layers.dense(inputs=h2 , units=784 , use_bias=True)

        return tf.tanh(h3)

def discriminator(x):
    with tf.variable_scope('discriminator' , reuse=tf.AUTO_REUSE):
        #x~[batch_size,784]
        h1 = tf.layers.dense(inputs=x , units=256 , use_bias=True)
        relu1 = leaky_relu(h1 , 0.01)
        
        h2 = tf.layers.dense(inputs=relu1 , units=256 , use_bias=True)
        relu2 = leaky_relu(h2 , 0.01)
        
        h3 = tf.layers.dense(inputs=relu2 , units=1 , use_bias=True)
        
        return h3

def gan_loss(fake_p , real_p):
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_p) , logits=fake_p)
    g_loss_ = tf.reduce_mean(g_loss)
    
    d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_p) , logits=real_p)
    d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_p) , logits=fake_p)
    d_loss_ = tf.reduce_mean(d_loss_real + d_loss_fake)
    
    return g_loss_ , d_loss_

batch_size = 128

x_train = tf.placeholder(dtype=tf.float32 , shape=[None,784])
x_train = 2*x_train-1.0
z_train = tf.placeholder(dtype=tf.float32 , shape=[None,96])

real_img = x_train
fake_img = generator(z_train)

real_p = discriminator(real_img)
fake_p = discriminator(fake_img)

g_loss , d_loss = gan_loss(fake_p , real_p)

d_opti = tf.train.AdamOptimizer(0.001 , beta1=0.5)
g_opti = tf.train.AdamOptimizer(0.001 , beta1=0.5)

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , 'discriminator')
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , 'generator')

d_train_step = d_opti.minimize(d_loss , var_list=d_vars)
g_train_step = g_opti.minimize(g_loss , var_list=g_vars)

sampler = generator(z_train)

num_iter = 100

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(num_iter):num_iter = 5000

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(4000):
        tx , ty = mnist.train.next_batch(batch_size)
        
        sess.run(d_train_step , feed_dict={x_train:tx,z_train:sample_noise(128,96)})
        sess.run(g_train_step , feed_dict={z_train:sample_noise(128,96)})
        if i % 50 == 0:
            print("g_loss is :"+str(sess.run(g_loss , feed_dict={z_train:sample_noise(128,96)})))
            print("d_loss is :"+str(sess.run(d_loss , feed_dict={x_train:tx,z_train:sample_noise(128,96)})))
    sample_img = sess.run(generator(tf.truncated_normal([batch_size,96])))
    show_images(sample_img[0:16])
    print(sample_img[0]-sample_img[1])