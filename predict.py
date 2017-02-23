import sys
import PIL as pil
import numpy as np
import tensorflow as tf
from model import build_model
from image_loader import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH , ALLOWED_CHARS, gen_images

if len(sys.argv) < 2:
    print "Usage: predict.py image_file"
    sys.exit(1)

image = pil.Image.open(sys.argv[1])

#Make sure the image is of right size
if image.size[0] != IMAGE_WIDTH or image.size[1] != IMAGE_HEIGHT:
    print "Image must be 28x28"
    sys.exit(1)

#Convert to greyscale and give a depth of 1
image = image.convert('L')
imageData = np.asarray(image)
imageData = np.reshape(imageData, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))

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
