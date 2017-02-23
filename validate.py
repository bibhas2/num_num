import numpy as np
import tensorflow as tf
from model import build_model
from image_loader import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH , ALLOWED_CHARS, gen_images

X, Y_, accuracy, Y, optimizer = build_model()
# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, "./model.ckpt")

test_X, test_Y = gen_images(5, True)

print "Generated test images. Getting predictions."
predictions = sess.run(Y, {X: test_X})

print "Calculating accuracy."
a = sess.run(accuracy, {X: test_X, Y_: test_Y})
print("Test accuracy:" + str(a))

# print np.argmax(predictions, 1)
# print np.argmax(test_Y, 1)
