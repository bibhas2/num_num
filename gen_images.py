import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import random
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
from conv_util import conv_layer, max_pool_layer

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 64
NUM_CHARS = 6

def gen_images(sampleCount):
    font = ImageFont.truetype("fake receipt.ttf", 20)

    imageResult = np.zeros((sampleCount, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    classResult = np.zeros((sampleCount, NUM_CHARS))
    stringResult = []

    for i in range(0, sampleCount):
        img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (0,0,0))
        draw = ImageDraw.Draw(img)

        amount = str(random.randrange(100000.00) / 100.0)
        filledAmount = amount.zfill(NUM_CHARS)
        draw.text((20,20), filledAmount, (255, 255, 255), font=font)

        imageResult[i] = np.asarray(img) 
        classResult[i] = np.fromstring(filledAmount, np.int8)
        #img.save(filledAmount + ".png")
        stringResult.append(filledAmount)

    return (imageResult, classResult, stringResult)

def build_model():
    network = input_data(shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    network = conv_2d(network, 48, 5, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 5, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 128, 5, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 2048, activation='relu')
    network = fully_connected(network, NUM_CHARS, activation='softmax')
    network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
    model = tflearn.DNN(network)

    return model

def build_model_tf():
    #Input data - a tensor of RGB (3 channel) images
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    #Classifications of input data
    Y_ = tf.placeholder(tf.float32, [None, 10])

    network = conv_layer(X, 48, 5, 1)
    network = max_pool_layer(network, 2, 2)
    network = conv_layer(network, 64, 5, 1)
    network = max_pool_layer(network, 2, 2)
    network = conv_layer(network, 128, 5, 1)
    network = max_pool_layer(network, 2, 2)
    print network.get_shape()

    return network
    
def train():
    trainingImages, trainingClasses, trainingStringResults = gen_images(2000)
    validateImages, validateClasses, validateStringResults = gen_images(200)
    model = build_model()

    # print imageResult.shape
    # print classResult[0]
    # print stringResult[0]

    model.fit(trainingImages, trainingClasses, n_epoch=10, validation_set=(validateImages, validateClasses), batch_size=100)

    model.save("weights.tfl")

def predict():
    validateImages, validateClasses, validateStringResults = gen_images(1)
    model = build_model()

    model.load("./weights.tfl")    

    # print imageResult.shape
    # print classResult[0]
    # print stringResult[0]

    p = model.predict(validateImages)

    print validateClasses
    print p

network = build_model_tf()

#train()
# predict()
#trainingImages, trainingClasses, trainingStringResults = gen_images(2)
#plt.imshow(trainingImages[0])
# pil_im = Image.open('824.65.png', 'r')
# plt.imshow(np.asarray(pil_im))
#plt.show()
#pil_im.show()
