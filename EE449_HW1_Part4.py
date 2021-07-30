import numpy as np
import tensorflow as tf
import json

from tensorflow.keras.layers import Dense as Dense
from tensorflow.keras.layers import Conv2D as Conv2D
from tensorflow.keras.layers import Input as Input
from tensorflow.keras.layers import MaxPooling2D as MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D as GlobalAvgPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from random import shuffle

# HYPERPARAMETERS
BS = 50
EPOCHS = 20
LR = 0.001

# MODEL


def create_mlp2():
    mlp_2 = tf.keras.Sequential(
        [
            Input(shape=(784,)),
            Dense(units=16, activation="relu"),
            Dense(units=64, use_bias=False, activation=None),
            Dense(units=5, activation="softmax"),
        ]
    )
    return mlp_2


# DATA
train_labels = np.load("dataset\\train_labels.npy")
test_labels = np.load("dataset\\test_labels.npy")
train_images = np.load("dataset\\train_images.npy")
test_images = np.load("dataset\\test_images.npy")


## [-1, 1] Scaling
#######################################################
test_images = (
    2 * (test_images - test_images.min()) / (test_images.max() - test_images.min())
)  # -1 to 1 scaling
########################################################
train_images = (
    2 * (train_images - train_images.min()) / (train_images.max() - train_images.min())
)  # -1 to 1 scaling
########################################################


train_image1D, validate_image1D, train_label1D, validate_label1D = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42
)

model_name = "mlp2"
model = create_mlp2()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = tf.keras.metrics.SparseCategoricalAccuracy()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
H = model.fit(
    x=train_image1D,
    y=train_label1D,
    shuffle=True,
    batch_size=BS,
    epochs=13,
    validation_data=(validate_image1D, validate_label1D),
)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
H = model.fit(
    x=train_image1D,
    y=train_label1D,
    shuffle=True,
    batch_size=BS,
    epochs=2,
    validation_data=(validate_image1D, validate_label1D),
)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
H = model.fit(
    x=train_image1D,
    y=train_label1D,
    shuffle=True,
    batch_size=BS,
    epochs=25,
    validation_data=(validate_image1D, validate_label1D),
)

test_loss, test_acc = model.evaluate(x=test_images, y=test_labels, batch_size=BS)
print(test_acc)

# In the first training experiment, accuracy stopped increasing at epoch 13.
# In the second training experiment, accuracy stopped increaing at epoch 15.
# In the third training experiment, the network was trained until epoch 30.
# The test accuracy is 0.9514


## Save dictionary
with open("PART4RESULTS/" + model_name + "_part_4_6_history.json", "w") as fp:
    json.dump(H.history, fp)

print(H.history.keys())