import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import layers

# Dividing a mnist data into test and training datasets. Each of datasets contains images
# and corresponding labels
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Making sure that data have been loaded correctly therefore I am checking the 5th
# element of the train_images dataset if it contains images for sure

digit = train_images[4]

# Because the image is stored in digit variable we use plt.imshow to display this encoded image as image
# cmap = plt.cm.binary this sets the colour map to binary meaning the image will be displayed
# in white and black

plt.imshow(digit, cmap = plt.cm.binary)
plt.show()

# This defines the architecture of neural network:
# -> keras.Sequential() defines that model will consist of linear stack of layers.
# The Sequential() model is used when you want to create a neural network where
# each layer has exactly one input and one output. It means the output of one layer
# is directly fed into the next layer, in a "stacked" fashion.

model = tf.keras.Sequential([
    layers.Dense(512, activation="relu"), # First hidden layer that has 512 neurons
    layers.Dense(10, activation="softmax")# Second hidden layer with 10 neurons each representing
                                                # the probability of the image being a digit from 0-9
])

# Here we compile the model with following settings:
# - optimizer = rmsprop (more info look below)
# - loss function = sparse_categorical_crossentropy (more info look below)
# - metrics = accuracy -> this tells the model to track the accuracy during the training
# The compile step prepares the model for training by specifying how to optimize it,
# how to calculate the error (loss), and which metrics to track.

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Reshaping the Images: train_images.reshape((60000, 28 * 28)). WHY?
# - Original Shape: The train_images array, when loaded from the MNIST dataset, is a 3D array with the shape (60000, 28, 28).
# The first dimension (60000) represents the number of training images. The second and third dimensions (28, 28) represent
# the height and width of each image, respectively (each image is 28x28 pixels).
# - Goal: We need to reshape this 3D array into a 2D array, since we deal with Sequential keras model which has fully connected
#  dense layers, in other words it expects the input data to be in a 2D format (a matrix where each row corresponds
# to a data point and each column corresponds to a feature).
# Reshaping Process: By calling train_images.reshape((60000, 28 * 28)), you are flattening each image (28x28 pixels)
# into a single vector of 784 elements (since 28 * 28 = 784). So the 3D array becomes a 2D array with the shape (60000, 784).
# Now, each image is represented as a 784-dimensional vector (one row per image), where each element of the vector
# corresponds to a pixel value in the original image.

train_images = train_images.reshape((60000, 28 * 28))

# We switch datatype to float32 because using float32 is a common practice in machine learning and neural networks
# because it strikes a balance between numerical precision and computational efficiency. After converting the pixel
# values to float32, the next part of the operation divides each pixel value by 255. It is because 0 represents black,
# 255 represents white, and values in between represent different shades of gray. Therefore, train_images.astype("float32") / 255
# scales the pixel values from the range [0, 255] to the range [0, 1], where 0 is white, 1 is black and in-between are shades of grey

train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# Trains the model on the training data (train_images and train_labels):
# epochs=5: The model will be trained for 5 complete passes (epochs) over the training data.
# batch_size=128: The model will process 128 samples at a time (in batches) during each training iteration.
#This is the actual training process, where the model learns from the data by adjusting its weights to minimize
# the loss function.

model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Evaluates the trained model on the test dataset (test_images and test_labels):
# test_loss: The loss value calculated on the test data, showing how well the model's predictions align with the true labels.
# test_acc: The accuracy on the test data, showing the percentage of correct predictions.

test_loss, test_acc = model.evaluate(test_images, test_labels)

# Rmsprop was developed as a stochastic technique for mini-batch learning.
# RMSprop deals with the above issue by using a moving average of squared
# gradients to normalize the gradient. This normalization balances the step size
# (momentum), decreasing the step for large gradients to avoid exploding and increasing
# the step for small gradients to avoid vanishing. Simply put, RMSprop uses an adaptive
# earning rate instead of treating the learning rate as a hyperparameter.
# This means that the learning rate changes over time.

#Cross-entropy loss is commonly used for classification problems, especially when the goal
# is to predict discrete categories (i.e., a class from a set of possible classes).
#The term "categorical" refers to the fact that we are dealing with multiple classes,
# each representing a category. For example, in a digit classification problem like MNIST
# (where the task is to predict digits 0 through 9), there are 10 possible categories (classes),
# and the model should output a probability distribution over these classes.