
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class testModel(tf.keras.Model):
    def __init__(self):
        super(testModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(1, kernel_initializer='Ones', activation=tf.nn.relu)
        #self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)


    def call(self, inputs):

        #ix = self.dense1(inputs)
        return self.dense1(inputs)

input_data = np.asarray([[10]])
module = testModel()
module._set_inputs(input_data)
print(module(input_data))

# Export the model to a SavedModel
module.save('model', save_format='tf')

# class MyModel(tf.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         x = Input(shape=(1,),)
#         self.d1 = layers.Dense(2, input_shape=(1,) )

#     @tf.function
#     def __call__(self, x):
#         return self.d1(x)

    # @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    # def mutate(self, new_v):
    #     self.v.assign(new_v)


# model = MyModel()

# model = tf.keras.Sequential()
# model.add(layers.Dense(2, input_shape=(1,),kernel_initializer=tf.initializers.ones ) )

# class CustomModule(tf.Module):

#   def __init__(self):
#     super(CustomModule, self).__init__()
#     self.v = tf.Variable(1.)

#   @tf.function
#   def __call__(self, x):
#     return x * self.v

#   @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
#   def mutate(self, new_v):
#     self.v.assign(new_v)

# input_data = np.asarray([[10]])
# module = MyModel()
# print(module(input_data))
# tf.saved_model.save(module, "model2")

# input_data = np.asarray([[10]])
# print(model(input_data))
# tf.saved_model.save(model,"~/project/tf/")
# print(input_data.shape)
# print(tf.__version__)