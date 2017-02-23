import sys
import PIL as pil
import numpy as np
import tensorflow as tf
from model import build_model
from image_loader import loadImageData, ALLOWED_CHARS

if len(sys.argv) < 2:
    print "Usage: predict.py image_file"
    sys.exit(1)

imageData = loadImageData(sys.argv[1])
test_X = np.array([imageData])

X, Y_, accuracy, Y, optimizer = build_model()
# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, "./model.ckpt")

print "Loaded test image. Getting prediction."
predictions = sess.run(Y, {X: test_X})

print ALLOWED_CHARS[np.argmax(predictions, 1)[0]]
# print np.argmax(test_Y, 1)
# print predictions
