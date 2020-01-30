
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class testModel(tf.keras.Model):
    def __init__(self):
        super(testModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(1, kernel_initializer='Ones', activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense1(inputs)

input_data = np.asarray([[10]])
module = testModel() 
module._set_inputs(input_data)
print(module(input_data))

# Export the model to a SavedModel
module.save('model', save_format='tf')