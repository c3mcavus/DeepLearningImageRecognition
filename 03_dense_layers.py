import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path
import tensorflow as tf

# Load data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data set to 0-to-1 range
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = x_train / 255
x_test = x_test / 255

# Convert class vectors to binary class matrices
# Our labels are single values from 0 to 9
# Instead, we want each label to be an array with on element set to 1
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Create a model and add layers
model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(32, 32, 3)))
model.add(Dense(10, activation="softmax"))

# Print a summary of the model
model.summary()