import numpy as np
import tensorflow as tf
from conv_util import conv_layer, fully_connected_layer, readout_layer, create_optimizer
from image_loader import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH , ALLOWED_CHARS, gen_images

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
Y_ = tf.placeholder(tf.float32, [None, len(ALLOWED_CHARS)])

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

# The model
Y1 = conv_layer(X, K, 5, 1)
Y2 = conv_layer(Y1, L, 5, 2)
Y3 = conv_layer(Y2, M, 5, 2)
Y4 = fully_connected_layer(Y3, N)
Ylogits, Y = readout_layer(Y4, len(ALLOWED_CHARS))

train_step, accuracy = create_optimizer(Ylogits, Y, Y_)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i):
    # training on batches of 100 images with 100 labels
    # batch_X, batch_Y = mnist.train.next_batch(100)
    batch_X, batch_Y = gen_images(100)

    learning_rate = 0.001
    #Compute accuracy every once in a while
    if i % 20 == 0:
        a = sess.run(accuracy, {X: batch_X, Y_: batch_Y})
        print("Batch: " + str(i) + ": accuracy:" + str(a))

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y})

for batch in range(0, 200):
    training_step(batch)


test_X, test_Y = gen_images(2, True)

# a = sess.run(accuracy, {X: test_X, Y_: test_Y})
# print("Test accuracy:" + str(a))

predictions = sess.run(Y, {X: test_X})

print np.argmax(predictions, 1)
print np.argmax(test_Y, 1)

# print test_Y
# print "======"
# print predictions