import numpy as np
import tensorflow as tf
from conv_util import conv_layer, fully_connected_layer, readout_layer, create_optimizer
from image_loader import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH , ALLOWED_CHARS, gen_images

def build_model():
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
    Y_ = tf.placeholder(tf.float32, [None, len(ALLOWED_CHARS)])

    # three convolutional layers with their channel counts, and a
    # fully connected layer (tha last layer has 10 softmax neurons)
    K = 40  # first convolutional layer output depth
    L = 20  # second convolutional layer output depth
    M = 10  # third convolutional layer
    N = 400  # fully connected layer

    # The model
    Y1 = conv_layer(X, K, 3, 1)
    Y2 = conv_layer(Y1, L, 4, 2)
    Y3 = conv_layer(Y2, M, 5, 2)
    Y4 = fully_connected_layer(Y3, N)
    Ylogits, Y = readout_layer(Y4, len(ALLOWED_CHARS))

    optimizer, accuracy = create_optimizer(Ylogits, Y, Y_)

    return (X, Y_, accuracy, Y, optimizer)
