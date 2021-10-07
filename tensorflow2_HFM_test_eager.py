#!/usr/bin/env python
# coding: utf-8

# In[1]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print(tf.__version__)


# In[2]:


tf.math.sin(tf.constant(3, dtype=tf.float64)).numpy()


# In[3]:


Pec = Rey = 100

init_model = keras.Sequential(name="2D_PST_HL")
init_model.add(keras.Input(shape=(3,), name="time_pos_input"))
for i in range(10):
    init_model.add(layers.Dense(50, activation=tf.math.sin, name=f"hidden_{i}"))

trained_concentration = layers.Dense(1, activation=tf.math.sin, name="PS_concentration")(init_model.output)

trained_velocity      = layers.Dense(3, activation=tf.math.sin, name="v_field")(init_model.output)

uninformed_model = keras.Model(
    inputs  = init_model.inputs,
    outputs = [trained_concentration, trained_velocity],
    name    = "2D_PST_NN"
)

uninformed_model.summary()


# In[5]:


class NavierStokesModel(keras.Model):
    def __init__(self, uninformed_model, Pec, Rey, loss_weights=[0.5,0.25,0.25], **kwargs):
        super(NavierStokesModel, self).__init__(**kwargs)
        
        self.ui_model = uninformed_model
        self.Pec = Pec
        self.Rey = Rey
        self.loss_weights = loss_weights
    
    def compile(self, optimizer, loss_fn, num_train_class, **kwargs):
        super(NavierStokesModel, self).compile(**kwargs)
        
        self.optimizer       = optimizer
        self.loss_fn         = loss_fn
        self.num_train_class = num_train_class
    
    def train_step(self, data):
        [x, t], y = data
        ps_x, ps_y = tf.boolean_mask(x, t == 0, axis=0), tf.boolean_mask(y, t == 0)[:,0]
        ge_x       = tf.boolean_mask(x, t == 1, axis=0)# , tf.boolean_mask(y, t == 1)
        il_x, il_y = tf.boolean_mask(x, t == 2, axis=0), tf.boolean_mask(y, t == 2)
        
        degen_grads = [0.0 for _ in self.ui_model.trainable_weights] # for graph mode
        
        ps_grads = degen_grads
        if(tf.shape(ps_x)[0] > 0):
            # print("Find gradients for data mse term")
            
            with tf.GradientTape() as tape:
                ps_c_pred, _ = self.ui_model(ps_x)
                if ps_y.shape == ps_c_pred.shape:
                    ps_loss = self.loss_fn(ps_y, ps_c_pred)
                else:
                    ps_loss = 0.0
            if ps_loss != 0.0:
                ps_grads = tape.gradient(ps_loss, self.ui_model.trainable_weights)
        
        
        ge_grads = degen_grads
        if(tf.shape(ge_x)[0] > 0):
            # print("Find gradients for residual term")
            
            t_in = ge_x[:,0:1]
            x_in = ge_x[:,1:2]
            y_in = ge_x[:,2:3]
            to_watch = [t_in, x_in, y_in]
            
            with tf.GradientTape() as tape:
                with tf.GradientTape(persistent=True) as s_d_tape:
                    s_d_tape.watch(to_watch)
                    with tf.GradientTape() as f_d_tape:
                        f_d_tape.watch(to_watch)
                        ge_c_pred, ge_uvp_pred = self.ui_model(tf.concat(to_watch, axis=1))
                        ge_cuvp_pred = tf.concat([ge_c_pred, ge_uvp_pred], axis=1)
                        
                    # cuvp_t, cuvp_x, cuvp_y = f_d_tape.gradient(ge_cuvp_pred, to_watch) # w.r.t. t, x, y
                    cuvp_t, cuvp_x, cuvp_y = [tf.reduce_sum(i, axis=[2,3]) for i in f_d_tape.jacobian(ge_cuvp_pred, to_watch)] # w.r.t. t, x, y
                    
                cuvp_xx = tf.reduce_sum(s_d_tape.jacobian(cuvp_x, x_in), axis=[2, 3])
                cuvp_yy = tf.reduce_sum(s_d_tape.jacobian(cuvp_y, x_in), axis=[2, 3])
                
                c_t, u_t, v_t      = cuvp_t[:, 0], cuvp_t[:, 1], cuvp_t[:, 2]
                c_x, u_x, v_x, p_x = cuvp_x[:, 0], cuvp_x[:, 1], cuvp_x[:, 2], cuvp_x[:, 3]
                c_y, u_y, v_y, p_y = cuvp_y[:, 0], cuvp_y[:, 1], cuvp_y[:, 2], cuvp_y[:, 3]
                
                c_xx, u_xx, v_xx   = cuvp_xx[:, 0], cuvp_xx[:, 1], cuvp_xx[:, 2]
                c_yy, u_yy, v_yy   = cuvp_yy[:, 0], cuvp_yy[:, 1], cuvp_yy[:, 2]
                
                u, v, Pec, Rey     = ge_uvp_pred[:, 0], ge_uvp_pred[:, 0], self.Pec, self.Rey
                
                e1 = c_t + (u*c_x + v*c_y) - (1.0/Pec)*(c_xx + c_yy)
                e2 = u_t + (u*u_x + v*u_y) + p_x - (1.0/Rey)*(u_xx + u_yy) 
                e3 = v_t + (u*v_x + v*v_y) + p_y - (1.0/Rey)*(v_xx + v_yy)
                e4 = u_x + v_y

                ge_loss = self.loss_fn(0, e1) + self.loss_fn(0, e2) + self.loss_fn(0, e3) + self.loss_fn(0, e4)
                # print(e1, e2, e3, e4, ge_loss)
            ge_grads = tape.gradient(ge_loss, self.ui_model.trainable_weights)
        
        
        il_grads = degen_grads
        if(tf.shape(il_x)[0] > 0):
            # print("Find gradients for inlet term")
            
            with tf.GradientTape() as tape:
                _, il_uvp_pred = self.ui_model(il_x)
                il_y_pred = il_uvp_pred[:,0:2]
                if il_y.shape == il_y_pred.shape:
                    il_loss = self.loss_fn(il_y, il_y_pred)
                else:
                    il_loss = 0.0
            if il_loss != 0.0:
                il_grads = tape.gradient(il_loss, self.ui_model.trainable_weights)
        
        
        total_grads = [
            sum([(grad if grad is not None else 0) * weight for grad, weight in zip(grad_tup, self.loss_weights)])
            for grad_tup in zip(ps_grads, ge_grads, il_grads)
        ]
        self.optimizer.apply_gradients(zip(total_grads, self.ui_model.trainable_weights))
        
    
    def call(self, x):
        return self.ui_model(x)
    
    


# In[6]:


model = NavierStokesModel(uninformed_model, Pec, Rey, name="2D_PST_NS_INN")


# In[7]:


model.compile(keras.optimizers.Adam(), keras.metrics.mean_squared_error, (50, 50, 100))# , run_eagerly=True)


# In[10]:


n = 20 # number of samples
m = 3  # number of features
x = tf.ones((n,m))
t = tf.concat([i * tf.ones((n//5,)) for i in range(5)], axis=0)
y = tf.ones((n,2))

# Eager evaluation of new model
est_c, NS_residual = model(x)
print(t)


# In[11]:


# Fitting of new model
model.fit([x, t], y)


# In[12]:


# Note of expected data format
# x is a n x 3 tensor that contains t, x, y input data
# y is a n x 2 tensor that contains output data (either passive scalar concentration (t == 0), inlet velocity (t == 2), or garbage (t == 1))
# t is a n x 1 tensor whose entries indicate what type of output the corresponding row in y is
# Phrased differently,  its entries indicate what term of the loss function the corresponding data point contributes to
# (in the case of t == 1, the input is a collocation point on which the NS residuals are evaluated)


# Remaining tasks:
# Try running with real data sample input (will require data preprocessing)
# Change data format so that the number of each type of input remains roughly constant between each mini-batch
# Run on Rockfish
# Try changing back to gradient instead of jacobian for speed reasons (will be more cluttered in terms of variables however)
# Copy code to predict lift/drag forces
# Hyperparameter Sweep


# In[13]:


# Gradient tape can be run on this to calculate gradients required for lift and drag forces
get_trained_vel_prs = keras.Model(
    inputs  = init_model.inputs,
    outputs = trained_velocity,
    name    = "2D_PST_VPL"
)

# u, v, p = (vel_prs_array := get_trained_vel_prs(x))[:,0], vel_prs_array[:,1], vel_prs_array[:,2]

# Eager evaluation of NN
vel_prs_array = get_trained_vel_prs(x)
u, v, p = vel_prs_array[:,0], vel_prs_array[:,1], vel_prs_array[:,2]
print(u)
print(v)
print(p)


# In[ ]:


# model.save(f"C:/UserTemp/models/{model.name}_model_{datetime.now().strftime('%y.%m.%d.%H.%M.%S')}")
# model.save_weights(f"C:/UserTemp/model_weights/{model.name}_model_{datetime.now().strftime('%y.%m.%d.%H.%M.%S')}")


# In[ ]:




