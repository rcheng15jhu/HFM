# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import tensorflow.random
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt
from datetime import datetime
import time

import NavierStokesModel
from NavierStokesModel import NavierStokesModel


tf.keras.backend.set_floatx('float64')

Pec = tf.constant(100, dtype='float64')
Rey = tf.constant(100, dtype='float64')

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

model = NavierStokesModel(uninformed_model, Pec, Rey, name="2D_PST_NS_INN")
model.compile(keras.optimizers.Adam(), keras.metrics.mean_squared_error, ())# , run_eagerly=True)


data = scipy.io.loadmat('Data/Cylinder2D.mat')

t_star = data['t_star']  # S x 1
x_star = data['x_star']  # N x 1
y_star = data['y_star']  # N x 1

S = t_star.shape[0]
N = x_star.shape[0]

# Rearrange Data
T_star = np.tile(t_star, (1, N)).T  # N x S
X_star = np.tile(x_star, (1, S))    # N x S
Y_star = np.tile(y_star, (1, S))    # N x S

X_1 = tf.concat([tf.reshape(i, [-1, 1]) for i in [T_star, X_star, Y_star]], axis=1)

U_star = data['U_star']  # N x S
V_star = data['V_star']  # N x S
P_star = data['P_star']  # N x S
C_star = data['C_star']  # N x S

PH  = tf.zeros((N*S,1), dtype='float64')
Y_1 = tf.concat([tf.reshape(C_star, [-1, 1]), PH], axis=1)

T_1 = PH


X_2 = X_1

Y_2 = tf.concat([PH, PH], axis=1)

T_2 = PH + 1


inlet_pos = X_1[:,1] == x_star.min()
X_3 = X_1[inlet_pos]

Y_3 = tf.concat([tf.reshape(i, [-1, 1])[inlet_pos] for i in [U_star, V_star]], axis=1)

T_3 = PH[inlet_pos] + 2


X = tf.concat([X_1, X_2, X_3], axis=0)
Y = tf.concat([Y_1, Y_2, Y_3], axis=0)
T = tf.concat([T_1, T_2, T_3], axis=0)
T = tf.reshape(T, [-1])


# some_choices = tf.concat([tf.ones((1000,)), tf.zeros((T.shape[0] - 1000,))], axis=0)
# idx_x = tf.random.shuffle(some_choices) == 1
# X = X[idx_x]
# Y = Y[idx_x]
# T = T[idx_x]


start = time.time()

epochs = 0
# train less than 3 hours, less than 500 epochs
while time.time() < start + 3600 * 3 and epochs < 500:
    model.fit([X, T], Y, epochs=5)
    epochs += 5
    model.save(f"Learned/models/{model.name}_model_{datetime.now().strftime('%y.%m.%d.%H.%M.%S')}")
    model.save_weights(f"Learned/model_weights/{model.name}_model_{datetime.now().strftime('%y.%m.%d.%H.%M.%S')}")
    #if on windows, and max_file_path_len has been reached
    # model.save(f"C:/UserTemp/models/{model.name}_model_{datetime.now().strftime('%y.%m.%d.%H.%M.%S')}")
    # model.save_weights(f"C:/UserTemp/model_weights/{model.name}_model_{datetime.now().strftime('%y.%m.%d.%H.%M.%S')}")

end = time.time()


graph_model = tf.function(model)
C_pred = 0*C_star
U_pred = 0*U_star
V_pred = 0*V_star
P_pred = 0*P_star
print(t_star.shape[0])
for snap in range(0, t_star.shape[0]):
    X_test = tf.concat([i[:, snap:snap+1] for i in [T_star, X_star, Y_star]], axis=1)

#     c_test = C_star[:, snap:snap+1]
#     u_test = U_star[:, snap:snap+1]
#     v_test = V_star[:, snap:snap+1]
#     p_test = P_star[:, snap:snap+1]

    # Prediction
    c_pred, uvp_pred = graph_model(X_test)

    C_pred[:, snap:snap+1] = c_pred
    U_pred[:, snap:snap+1] = uvp_pred[:,0:1]
    V_pred[:, snap:snap+1] = uvp_pred[:,1:2]
    P_pred[:, snap:snap+1] = uvp_pred[:,2:3]

    # # Error
    # error_c = relative_error(c_pred, c_test)
    # error_u = relative_error(u_pred, u_test)
    # error_v = relative_error(v_pred, v_test)
    # error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))
    #
    # print('Error c: %e' % error_c)
    # print('Error u: %e' % error_u)
    # print('Error v: %e' % error_v)
    # print('Error p: %e' % error_p)
    
    if snap % 25 == 0:
        print(snap)



scipy.io.savemat('Results/Cylinder2D_results_%s.mat' %(datetime.now().strftime('%Y_%m_%d')),
         {'C_pred': C_pred, 'U_pred': U_pred, 'V_pred': V_pred, 'P_pred': P_pred, 'time': np.asarray([start, end])})