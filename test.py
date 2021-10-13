# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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

model = NavierStokesModel(uninformed_model, Pec, Rey, name="2D_PST_NS_INN")
model.compile(keras.optimizers.Adam(), keras.metrics.mean_squared_error, (,))# , run_eagerly=True)

