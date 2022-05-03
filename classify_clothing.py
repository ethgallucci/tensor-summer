# Trains a neural network to classify images of clothing
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Load the fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# Unpack the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Explore the data
print("Dataset format: ", train_images.shape)
print("Dataset labels: ", len(train_labels))
print("Test set: ", test_images.shape)
print("Test labels: ", len(test_labels))

# Preprocess the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

# Verifying that the data is in the correct format and ready to build the network
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

## Build the model

# Setup layers
model = tf.keras.Sequential(
    [
        # transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels)
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # Densely connected 128 neuron layer
        tf.keras.layers.Dense(128, activation="relu"),
        # Returns a logits array with len 10
        tf.keras.layers.Dense(10),
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)