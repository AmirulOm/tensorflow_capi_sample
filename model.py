
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(InputLayer(input_shape=(30)))
model.add(Embedding(1000 + 1, 40))  # tunable output length
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(TimeDistributed(Dense(60 + 1)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.summary()
input_data = np.ones((30,1))
print(model(input_data))
model.save("lstm2")