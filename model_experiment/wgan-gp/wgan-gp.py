
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
get_ipython().magic('matplotlib inline')


# In[2]:

#in this simple implemention of wgan-gp , using conv and deconv to build D and G
#for the sake of excerise , using mnist as the dataset of this simple demo , and i don't take the extensibility of code into
#account.


# In[3]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./datasets/MNIST_data', one_hot=False)


# In[4]:

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


# In[5]:

def deconv2d(data , out_dim , k_w=5 , k_h=5 , s_w=2 , s_h=2 , name='deconv2d' , stddev=0.2):
    with tf.variable_scope(name):
        shape = data.get_shape().as_list()
        W = tf.get_variable('filter' , [k_w , k_h , out_dim , shape[-1]] , initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias' , [out_dim] , initializer=tf.zeros_initializer())
        
        out_shape = [int(shape[0]) , int(shape[1] * s_w) , int(shape[2] * s_h) , out_dim]
        
        return tf.nn.conv2d_transpose(data , W , strides=[1,s_w,s_h,1] , output_shape=out_shape) + b


# In[6]:

def conv2d(data , out_dim , k_w=5 , k_h=5 , s_w=2 , s_h=2 , name='conved' , stddev=0.2):
    with tf.variable_scope(name):
        shape = data.get_shape()
        W = tf.get_variable('W' , [k_w,k_h,int(shape[-1]) , out_dim] , initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b' , [out_dim] , initializer=tf.zeros_initializer())
        
        return tf.nn.conv2d(data , W , strides=[1,s_w,s_h,1] , padding='SAME') + b


# In[7]:

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


# In[8]:

def leaky_relu(x , rate=0.2):
    return tf.maximum(x , x*rate)


# In[9]:

def dense(data , out_dim , name='dense' , stddev=0.02):
    with tf.variable_scope(name):
        shape = data.get_shape()
        W = tf.get_variable('W' , [shape[1] , out_dim] , initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b' , [out_dim] , initializer=tf.zeros_initializer())
        
        return tf.matmul(data , W)+b


# In[10]:

def get_noise(batch_size=50 , noise_dim=100):
    return np.random.uniform(low=-1.0 , high=1.0 , size=[batch_size , noise_dim])


# In[11]:

def generator(z , batch_size=50):
    with tf.variable_scope('generator' , reuse=tf.AUTO_REUSE , initializer=tf.truncated_normal_initializer(stddev=0.2)):
        z_ = dense(z , 4*4*256 , name='g_den1')
        data = leaky_relu(batchnorm(z_ , 'g_bn1'))
        data_ = tf.reshape(data , [batch_size,4,4,256])
        
        filter1 = tf.get_variable('filter1' , [5,5,128,256])
        h1 = tf.nn.conv2d_transpose(data_ , filter1 , output_shape=[batch_size,7,7,128] , strides=[1,2,2,1])
        h1_ = leaky_relu(batchnorm(h1 , 'g_bn2'))
        
        h2 = deconv2d(h1_ , 64 , name='g_decon1')
        h2_ = leaky_relu(batchnorm(h2 , 'g_bn3'))
        
        h3 = deconv2d(h2_ , 1 , name='g_decon2')
        h3_ = tf.reshape(h3 , [batch_size,784])
        
        return tf.nn.tanh(h3_)


# In[12]:

def discriminator(x , batch_size=50):
    with tf.variable_scope('discriminator' , reuse=tf.AUTO_REUSE , initializer=tf.truncated_normal_initializer(stddev=0.2)):
        x_ = dense(x , 32*32*1 , name='d_den1')
        data = leaky_relu(batchnorm(x_ , 'd_bn1'))
        data_ = tf.reshape(data , [batch_size,32,32,1])
        
        h1 = conv2d(data_ , 64 , name='d_conv1') #16
        h1_ = leaky_relu(batchnorm(h1 , 'd_bn2'))
        
        h2 = conv2d(h1_ , 128 , name='d_conv2') #8
        h2_ = leaky_relu(batchnorm(h2 , 'd_bn3'))
        
        h3 = conv2d(h2_ , 256 , name='d_conv3') #4
        h3_ = leaky_relu(batchnorm(h3 , 'd_bn4'))
        
        out = tf.reshape(h3_ , [batch_size,-1])
        return dense(out , 1 , name='d_den2')


# In[13]:

def gan_g_loss(fake_p):
    return tf.reduce_mean(-fake_p)


# In[14]:

def gan_d_loss(real_p , fake_p , img_real , img_fake , batch_size=50 , lam=10):
    re_loss = -real_p + fake_p
    
    alpha_dist = tf.contrib.distributions.Uniform(low=0.0 , high=1.0)
    alpha = alpha_dist.sample((batch_size,1))
    interpolated =  img_real + (1-alpha)*(img_fake-img_real)
    intet_p = discriminator(interpolated)
    grad_inter = tf.gradients(intet_p , interpolated)
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(grad_inter) , [0,1,2]))
    penalty = lam*tf.square(grad_l2-1)
    
    return tf.reduce_mean(re_loss + penalty)


# In[15]:

batch_size = 50
noise_dim = 100

x_train = tf.placeholder(shape=[batch_size,784] , dtype=tf.float32)
z_train = tf.placeholder(shape=[batch_size , noise_dim] , dtype=tf.float32)

img_real = x_train
img_fake = generator(z_train)

real_p = discriminator(img_real)
fake_p = discriminator(img_fake)

g_loss = gan_g_loss(fake_p)
d_loss = gan_d_loss(real_p , fake_p , img_real , img_fake)

d_opti = tf.train.RMSPropOptimizer(1e-4)
g_opti = tf.train.RMSPropOptimizer(1e-4)

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , 'discriminator')
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , 'generator')

d_train_step = d_opti.minimize(d_loss , var_list=d_vars)
g_train_step = g_opti.minimize(g_loss , var_list=g_vars)


# In[16]:

init = tf.global_variables_initializer()

num_iter = 1000

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(num_iter):
        tx , ty = mnist.train.next_batch(batch_size)
        tz = get_noise(batch_size , noise_dim)
        
        _ , d_cur_loss = sess.run([d_train_step , d_loss] , feed_dict={x_train:tx , z_train:tz})
        _ , g_cur_loss = sess.run([g_train_step , g_loss] , feed_dict={x_train:tx , z_train:tz})
        
        sys.stdout.write("\r%d / %d: %f, %f" % (i,1000, d_cur_loss, g_cur_loss))
        sys.stdout.flush()
    samples = sess.run(generator(tf.truncated_normal([batch_size,100])))
    show_images(samples[0:16])

