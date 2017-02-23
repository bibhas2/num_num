import numpy as np
import tensorflow as tf
from model import build_model
from image_loader import gen_images

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
    sess.run(optimizer, {X: batch_X, Y_: batch_Y})

X, Y_, accuracy, Y, optimizer = build_model()
# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for batch in range(0, 300):
    training_step(batch)

#Save the weights and biases
saver = tf.train.Saver()
saver.save(sess, "./model.ckpt")
