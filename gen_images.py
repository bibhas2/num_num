import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv_util import conv_layer, fully_connected_layer, readout_layer, create_optimizer

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_DEPTH = 3
ALLOWED_CHARS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def gen_images(sampleCount, saveImages=False):
    fontList = [
        "Roboto-Medium.ttf", 
        "Slabo27px-Regular.ttf", 
        "Montserrat-Regular.ttf", 
        "Merriweather-Regular.ttf"]

    imageResult = np.zeros((sampleCount * len(ALLOWED_CHARS), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    classResult = np.zeros((sampleCount * len(ALLOWED_CHARS), len(ALLOWED_CHARS)))

    for charIndex, ch in enumerate(ALLOWED_CHARS):
        for i in range(0, sampleCount):
            fontSize = random.randint(19, 21)
            fontIndex = random.randint(0, len(fontList) - 1)
            font = ImageFont.truetype(fontList[fontIndex], fontSize)

            img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (0,0,0))
            draw = ImageDraw.Draw(img)

            xPos = random.randint(2, 10)
            yPos = random.randint(2, 10)
            draw.text((xPos, yPos), ch, (255, 255, 255), font=font)
            del draw
            
            if saveImages:
                img.save("image-" + str(charIndex * sampleCount + i) + ".png")

            imageResult[charIndex * sampleCount + i] = np.asarray(img) 

            thisClass = np.zeros(len(ALLOWED_CHARS))
            thisClass[charIndex] = 1.0

            classResult[charIndex * sampleCount + i] = thisClass

    return (imageResult, classResult)

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

for batch in range(0, 400):
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