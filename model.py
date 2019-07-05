import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.optimizers import SGD


def model_MPL():
    input_ = Input(shape = (28, 28))
    flatten_ = Flatten()(input_)
    dense_ = Dense(128, activation = 'relu')(flatten_)
    soft_max = Dense(10, activation = 'softmax')(dense_)

    return Model(inputs = input_, outputs = soft_max)

model = model_MPL()
model.summary()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

sgd = SGD(lr = 0.01, decay = 0.01)

model.compile(optimizer = sgd, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 5)

loss, score = model.evaluate(X_test, y_test)

print(loss, score)

