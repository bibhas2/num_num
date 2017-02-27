import numpy as np
import tensorflow as tf
import math
from model import build_model
from image_loader import gen_images

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i):
    # training on batches of 100 images with 100 labels
    # batch_X, batch_Y = mnist.train.next_batch(100)
    batch_X, batch_Y = gen_images(50)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 400.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    #Compute accuracy every once in a while
    if i % 20 == 0:
        a = sess.run(accuracy, {X: batch_X, Y_: batch_Y})
        print("Batch: " + str(i) + " Learning rate: " + str(learning_rate) + " Accuracy:" + str(a))

    # the backpropagation training step
    sess.run(optimizer, {X: batch_X, Y_: batch_Y, learningRate: learning_rate})

X, Y_, accuracy, Y, optimizer, learningRate = build_model()
# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()

for batch in range(0, 1500):
    training_step(batch)
    #Save the weights and biases
    saver.save(sess, "./model.ckpt")

