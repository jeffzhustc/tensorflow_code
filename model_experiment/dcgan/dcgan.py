
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
get_ipython().magic('matplotlib inline')


# In[2]:

def preprocess(x):
    return 2 * x - 1.0


# In[3]:

def leaky_relu(x , rate=0.2):
    return tf.maximum(x , x*rate)


# In[4]:

def batchnorm(data , name='bn'):
    with tf.variable_scope(name):
        out_dim = data.get_shape()[-1]
        beta = tf.get_variable('beta' , [out_dim] , initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma' , [out_dim] , initializer=tf.ones_initializer())
        
        if len(data.get_shape()) == 4:
            mean , var = tf.nn.moments(data , [0,1,2])
        elif len(data.get_shape()) == 2:
            mean , var = tf.nn.moments(data , [0])
        return tf.nn.batch_normalization(data , mean , var , beta , gamma , 1e-5)


# In[5]:

def dense(data , out_dim , name='dense' , stddev=0.02):
    with tf.variable_scope(name):
        shape = data.get_shape()
        W = tf.get_variable('W' , [shape[1] , out_dim] , initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b' , [out_dim] , initializer=tf.zeros_initializer())
        
        return tf.matmul(data , W)+b


# In[6]:

def deconv2d(data , out_dim , k_w=5 , k_h=5 , s_w=2 , s_h=2 , name='deconv2d' , stddev=0.2):
    with tf.variable_scope(name):
        shape = data.get_shape().as_list()
        W = tf.get_variable('filter' , [k_w , k_h , out_dim , shape[-1]] , initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias' , [out_dim] , initializer=tf.zeros_initializer())
        
        out_shape = [int(shape[0]) , int(shape[1] * s_w) , int(shape[2] * s_h) , out_dim]
        
        return tf.nn.conv2d_transpose(data , W , strides=[1,s_w,s_h,1] , output_shape=out_shape) + b


# In[7]:

def conv2d(data , out_dim , k_w=5 , k_h=5 , s_w=2 , s_h=2 , name='conved' , stddev=0.2):
    with tf.variable_scope(name):
        shape = data.get_shape()
        W = tf.get_variable('W' , [k_w,k_h,int(shape[-1]) , out_dim] , initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b' , [out_dim] , initializer=tf.zeros_initializer())
        
        return tf.nn.conv2d(data , W , strides=[1,s_w,s_h,1] , padding='SAME') + b


# In[8]:

def generator(z , batch_size=64):
    with tf.variable_scope('generator' , reuse=tf.AUTO_REUSE):
        h1 = dense(z , 1024 , name='g_den1')
        h1_ = tf.nn.relu(batchnorm(h1 , name='g_bn1'))
        
        h2 = dense(h1_ , 7*7*128 , name='g_den2')
        h2_ = tf.nn.relu(batchnorm(h2 , name='g_bn2'))
        
        data = tf.reshape(h2_ , [-1,7,7,128])
        
        h3 = deconv2d(data , 64 , name='g_deconv1')
        h3_ = tf.nn.relu(batchnorm(h3 , name='g_bn3'))
        
        h4 = deconv2d(h3_ , 1 , name='g_deconv2')
        out = tf.reshape(h4 , [-1,784])
        
        return tf.nn.tanh(out)


# In[9]:

def discriminator(x , batch_size=64):
    with tf.variable_scope('discriminator' , reuse=tf.AUTO_REUSE):
        #x~[batch_szie,784]
        x_ = tf.reshape(x , [-1,28,28,1])
        
        h1 = conv2d(x_ , 64 , name='d_conv1')
        h1_ = leaky_relu(batchnorm(h1 , 'd_bn1'))
        
        h2 = conv2d(h1_ , 128 , name='d_conv2')
        h2_ = leaky_relu(batchnorm(h2 , 'd_bn2'))
        
        data = tf.reshape(h2_ , [-1,256])
        data = leaky_relu(data)
        
        out = dense(data , 1 , name='d_dense1')
        return tf.nn.sigmoid(out)


# In[10]:

def gan_loss(fake_p , real_p):
    g_loss = tf.reduce_mean(-tf.log(fake_p))
    d_loss = tf.reduce_mean(tf.log(fake_p)-tf.log(real_p))
    
    return g_loss , d_loss


# In[11]:

batch_size = 64
noise_dim = 96

x_train = tf.placeholder(dtype=tf.float32 , shape=[batch_size,784])
z_train = tf.placeholder(dtype=tf.float32 , shape=[batch_size,noise_dim])

real_img = x_train
fake_img = generator(z_train , batch_size)

real_p = discriminator(real_img)
fake_p = discriminator(fake_img)

g_loss , d_loss = gan_loss(fake_p , real_p)


# In[12]:

d_opti = tf.train.AdamOptimizer(0.0001 , beta1=0.1)
g_opti = tf.train.AdamOptimizer(0.0002 , beta2=0.3)

d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS , 'discriminator')
g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS , 'generator')
d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , 'discriminator')
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , 'generator')

#with tf.control_dependencies(d_update_ops):
d_train_step = d_opti.minimize(d_loss , var_list=d_vars)
#with tf.control_dependencies(g_update_ops):
g_train_step = g_opti.minimize(g_loss , var_list=g_vars)


# In[13]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./datasets/MNIST_data', one_hot=False)


# In[14]:

def get_noise(batch_size=128 , noise_dim=96):
    return np.random.uniform(low=-1.0 , high=1.0 , size=[batch_size , noise_dim])


# In[15]:

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


# In[16]:

num_iter = 100

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(800):
        tx , ty = mnist.train.next_batch(batch_size)
        tx = preprocess(tx)
        tz = get_noise(batch_size , noise_dim)
        
        cur_g_loss , _ = sess.run([g_loss , g_train_step] , feed_dict={z_train:tz})
        cur_d_loss , _ = sess.run([d_loss , d_train_step] , feed_dict={x_train:tx,z_train:tz})
        
        sys.stdout.write("\r%d / %d: %f, %f" % (i,800, cur_d_loss, cur_g_loss))
        sys.stdout.flush()
            
    samples = sess.run(generator(tf.truncated_normal([batch_size,96])))
    show_images(samples[0:16])


# In[ ]:



