
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np


# In[2]:

def get_atom_type_num(data):
    max = 0
    length = len(data)

    for i in range(length):
        tmp = np.max(data[i])
        if tmp > max:
            max = tmp
    return max

def get_max_min_dis(data):
    max_=0.0
    min_ = 99999.0
    
    length = len(data)
    
    for i in range(length):
        tmax , tmin = np.max(data[i]) , np.min(data[i])
        if tmax > max_:
            max_ = tmax
        if tmin < min_:
            min_ = tmin
    return min_ , max_


# In[3]:

#this cell is for globe variable
dataset = np.load("QM9_nano.npz")
num_max_atom_type = get_atom_type_num(dataset['Atoms'])
num_max_atom_type += 1

num_embedding_dim = 64
num_feature_dim = 64

min_dis , max_dis = get_max_min_dis(dataset['Distance'])
min_dis = np.floor(min_dis)
max_dis = np.ceil(max_dis)

gamma_in_rbf = 3

atom_type_embedding = tf.get_variable(name='atom_type_embedding' , shape=[num_max_atom_type , num_embedding_dim])


# In[4]:

def dense(input , name , out_dim = 64 ):
    with tf.variable_scope(name_or_scope=name,initializer=tf.random_normal_initializer(stddev=0.02) , reuse=tf.AUTO_REUSE):
        shape = input.get_shape()
        W = tf.get_variable("W" , [shape[-1],out_dim])
        b = tf.get_variable("b" , [out_dim])
        
        return tf.matmul(input , W) + b


# In[5]:

def shifted_softplus(x , name):
    with tf.variable_scope(name , reuse=tf.AUTO_REUSE):
        return tf.log(tf.exp(x) + 0.5)


# In[6]:

def rbf_layer(d , min_ , max_ , name , interval=0.1):
    with tf.variable_scope(name_or_scope=name , reuse=tf.AUTO_REUSE):
        num = (max_ - min_) / interval
        shape = d.get_shape()
        #print(num)
        
        u_list = []
        for i in range(int(num)):
            u_list.append(('%.2f' % (min_ + i*interval)))
        
        distance_list = []
        for i in range(len(u_list)):
            tmp = tf.exp(-1.0*gamma_in_rbf*(d - float(u_list[i]))**2)
            distance_list.append(tmp)
        test = tf.stack(distance_list , axis=2 , name='test')
        #print(test)
        return test , int(num)


# In[7]:

def convolutional(X , matrix , name):
    with tf.variable_scope(name , reuse=tf.AUTO_REUSE):
        shape_mat0 = matrix.get_shape().as_list()[0]
        shape_mat1 = matrix.get_shape().as_list()[1]
        #shape_X = [atom_len , 64]
        
        print(shape_mat0)
        
        
        list__ = []
        for i in range(shape_mat0):
            list_ = []
            for j in range(shape_mat1):
                tmp = X[j] * matrix[i][j]
                list_.append(tmp)

            tmp = list_[0]
            for i in range(len(list_)-1):
                tmp1 = tf.add(tmp , list_[i+1])
                tmp = tmp1
            list__.append(tmp)
            
        return tf.stack(list__,0)


# In[8]:

#this function must be reviewed!
def cfconv_layer(X , D , atom_len , name):
    with tf.variable_scope(name , reuse=tf.AUTO_REUSE):
        #in cfconv_layer X is a vector representing the atom in molecule and D is a matrix representing the distance between
        #atom i and atom j
        shape_D = [atom_len , atom_len]
        shape_X = [atom_len , 64]
        distance_tensor , num = rbf_layer(D , min_dis , max_dis , name='cfconv_rbf')
        distance_tensor_ = tf.reshape(distance_tensor , [-1,num])
        
        dense1 = dense(distance_tensor_ , name='cfconv_den1' , out_dim=num_feature_dim)
        ssp1 = shifted_softplus(dense1 , name='cfconv_ssp1')
        
        dense2 = dense(ssp1 ,  name='cfconv_den2' , out_dim=num_feature_dim)
        ssp2 = shifted_softplus(dense2 , name='cfconv_ssp2')
        
        matrix = tf.reshape(ssp2 , [shape_D[0],shape_D[0],num_feature_dim])
        
        X_ = convolutional(X , matrix , 'convolutional')
        return X_ 


# In[9]:

def atom_wise(X , name , dim1=64 , dim2=64):
    with tf.variable_scope(name , initializer=tf.random_normal_initializer(stddev=0.02) , reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W" , [dim1,dim2])
        b = tf.get_variable("b" , [dim2])
        
        return tf.matmul(X,W)+b


# In[10]:

def interaction_layer(X , D  , atom_len , name):
    with tf.variable_scope(name_or_scope=name , reuse=tf.AUTO_REUSE):
        aw1 = atom_wise(X , "il_aw")
        cf1 = cfconv_layer(aw1 , D , atom_len , "il_cf1")
        aw2 = atom_wise(cf1 , "il_aw")
        ssp1 = shifted_softplus(aw2 , "il_ssp1")
        aw3 = atom_wise(ssp1 , "il_aw")
        
        return X+aw3


# In[11]:

def total_struct(Z , D , atom_len):
    D = tf.reshape(D , [atom_len , atom_len])
    X = tf.nn.embedding_lookup(atom_type_embedding , Z)
    X1 = interaction_layer(X , D , atom_len , 'ts_il1')
    X2 = interaction_layer(X1 , D , atom_len , 'ts_il2')
    X3 = interaction_layer(X2 , D , atom_len , 'ts_il3')
    
    h1 = atom_wise(X3 , 'ts_aw1' , dim1=64 , dim2=32)
    h2 = shifted_softplus(h1 , 'ts_ssp1')
    h3 = atom_wise(h2 , 'ts_aw2' , dim1=32 , dim2=1)
    
    e = tf.reduce_sum(h3)
    
    return e


# In[12]:

Z = tf.placeholder(shape=[None] , dtype=tf.int32)
R = tf.placeholder(shape=[None] , dtype=tf.float32)
U = tf.placeholder(shape=[None] , dtype=tf.float32)
atom_ = tf.placeholder(shape=None , dtype=tf.int32)

e = total_struct(Z , R , atom_)

error = (U-e)**2

optimizer = tf.train.AdamOptimizer(1e-3).minimize(error)

init = tf.global_variables_initializer()
length = len(dataset['Atoms'])

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(20):
        for i in range(length):
            z = dataset['Atoms'][i]
            r = dataset['Distance'][i]
            u = dataset['U0'][i]
            #print(data_len)
            atom_len = len(z)
            
            r = np.reshape(r , (atom_len*atom_len))
            
            error_ , _ = sess.run([error , optimizer] , feed_dict = {Z:z , R:r , U:u , atom_:atom_len})
            print(error)
    sess.run(error)


# In[ ]:




# In[ ]:



