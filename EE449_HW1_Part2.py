import numpy as np
import tensorflow as tf
import json
import h5py

from tensorflow.keras.layers import Dense as Dense
from tensorflow.keras.layers import Conv2D as Conv2D
from tensorflow.keras.layers import Input as Input
from tensorflow.keras.layers import MaxPooling2D as MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D as GlobalAvgPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# HYPERPARAMETERS
BS = 50
EPOCHS = 15

# MODELS


def create_mlp1():
    mlp_1 = tf.keras.Sequential(
        [
            Input(shape=(784,)),
            Dense(units=64, activation="relu"),
            Dense(units=5, activation="softmax"),
        ]
    )
    return mlp_1


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


# strides = (1,1) and valid padding is already default for both pooling and conv2d.


def create_cnn3():
    cnn_3 = tf.keras.Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(filters=16, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=8, kernel_size=(7, 7), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=16, kernel_size=(5, 5), activation=None),
            MaxPooling2D(pool_size=(2, 2)),
            GlobalAvgPooling2D(data_format="channels_last"),
            Dense(units=5, activation="softmax"),
        ]
    )
    return cnn_3


def create_cnn4():
    cnn_4 = tf.keras.Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(filters=16, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=8, kernel_size=(5, 5), activation="relu"),
            Conv2D(filters=8, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=16, kernel_size=(5, 5), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            GlobalAvgPooling2D(data_format="channels_last"),
            Dense(units=5, activation="softmax"),
        ]
    )
    return cnn_4


def create_cnn5():
    cnn_5 = tf.keras.Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(filters=16, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=8, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=8, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=8, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=16, kernel_size=(3, 3), activation="relu"),
            Conv2D(filters=16, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            GlobalAvgPooling2D(data_format="channels_last"),
            Dense(units=5, activation="softmax"),
        ]
    )
    return cnn_5


## DATA
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

## CNN's are going to take input as images. So here is the reshaping:
(samples, rectified) = train_images.shape
train_images_2D = np.reshape(train_images, newshape=(samples, 28, 28, 1))
(samples, rectified) = test_images.shape
test_images_2D = np.reshape(test_images, newshape=(samples, 28, 28, 1))

train_image2D, validate_image2D, train_label2D, validate_label2D = train_test_split(
    train_images_2D, train_labels, test_size=0.1, random_state=42
)

(samples, rectified) = train_image1D.shape
ITERATION_PER_EPOCH = np.floor(samples // BS)

RESULTS_DICT = {}

for i in range(10):

    model = create_mlp2()
    model_name = "mlp2"

    ## Model parameters
    optimizer = Adam(learning_rate=0.01)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    print("MODEL COMPILED")

    ## Containers
    training_acc_curve = []
    training_loss_curve = []
    validation_acc_curve = []

    for epoch in range(int(EPOCHS)):
        for batch in range(int(ITERATION_PER_EPOCH)):

            train_data = train_image1D[
                batch * BS : (batch + 1) * BS, :
            ]  # Drop last 2 dimension for training MLPs
            train_label = train_label1D[batch * BS : (batch + 1) * BS]
            train_dict = model.train_on_batch(
                x=train_data, y=train_label, reset_metrics=False, return_dict=True
            )
            # If current iteration is multiple of 10, get training metrics
            if (epoch * BS + batch) % 10 == 0:
                train_loss, train_acc = (
                    train_dict["loss"],
                    train_dict["sparse_categorical_accuracy"],
                )
                training_acc_curve.append(train_acc)
                training_loss_curve.append(train_loss)
                val_loss, val_acc = model.evaluate(
                    x=validate_image1D, y=validate_label1D, batch_size=BS
                )
                validation_acc_curve.append(val_acc)

        # Create Permutation
        permute_idx = np.random.permutation(samples)

        ## Shuffle Training data for MLP
        train_image1D = train_image1D[permute_idx, :]

        ## Shuffle Training data for CNN
        train_image2D = train_image2D[permute_idx, :, :, :]

        ## Shuffle Labels
        train_label1D = train_label1D[permute_idx]
        train_label2D = train_label2D[permute_idx]

    ## Get the weights of the first trainable layer
    first_layer = model.trainable_weights[0].numpy().tolist()

    ## Get test accuracy
    test_loss, test_acc = model.evaluate(x=test_images, y=test_labels, batch_size=BS)

    ## Create dictionary
    result_dict = {}
    result_dict["train_acc_curve"] = training_acc_curve
    result_dict["train_loss_curve"] = training_loss_curve
    result_dict["validation_acc_curve"] = validation_acc_curve
    result_dict["first_layer"] = first_layer
    result_dict["test_acc"] = test_acc

    RESULTS_DICT["Trial" + str(i + 1)] = result_dict


## Save dictionary
with open("PART2RESULTS/" + model_name + "_results.json", "w") as fp:
    json.dump(RESULTS_DICT, fp)

## Save model
print("[INFO] Loop ended. Saving model.")
model.save(model_name + ".model")
