import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
digit = train_images[4]
plt.imshow(digit, cmap = plt.cm.binary)
plt.show()
model = tf.keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])