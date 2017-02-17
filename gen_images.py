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
from conv_util import conv_layer, max_pool_layer, weight_variable, bias_variable

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 64
ALLOWED_CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']
INPUT_CHAR_COUNT = 6 #How many chars in a sample?

def gen_images(sampleCount):
    font = ImageFont.truetype("fake receipt.ttf", 20)

    imageResult = np.zeros((sampleCount, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    classResult = np.zeros((sampleCount, INPUT_CHAR_COUNT * len(ALLOWED_CHARS)))
    stringResult = []

    for i in range(0, sampleCount):
        img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (0,0,0))
        draw = ImageDraw.Draw(img)

        amount = str(random.randrange(100000.00) / 100.0)
        filledAmount = amount.zfill(INPUT_CHAR_COUNT)
        draw.text((20,20), filledAmount, (255, 255, 255), font=font)

        imageResult[i] = np.asarray(img) 

        thisClass = np.zeros((INPUT_CHAR_COUNT, len(ALLOWED_CHARS)))
        charList = list(filledAmount)

        for (charPosition, char) in enumerate(charList):
            charClass = ALLOWED_CHARS.index(char)
            thisClass[charPosition, charClass] = 1.0

        #Unroll the class matrix
        classResult[i] = thisClass.reshape(INPUT_CHAR_COUNT * len(ALLOWED_CHARS))
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
    network = fully_connected(network, len(ALLOWED_CHARS), activation='softmax')
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
    network = max_pool_layer(network, 2, 2) #1
    network = conv_layer(network, 64, 5, 1)
    network = max_pool_layer(network, 2, 2) #2
    network = conv_layer(network, 128, 5, 1)
    network = max_pool_layer(network, 2, 2) #3
    
    #Get the final size of the tensor after all the convolution and
    #pooling
    finalHeight = network.get_shape()[1].value
    finalWidth = network.get_shape()[2].value
    finalDepth = network.get_shape()[3].value

    #Build the fully connected layer.
    #A fully connected layer can work with depth 1 input only.
    #Unroll the output from the last pooling layer into a 2D matrix
    #The X is already transposed this way
    Xf = tf.reshape(network, shape=[-1, finalHeight * finalWidth * finalDepth])
    numNeurons = 2048
    Wf = weight_variable([finalHeight * finalWidth * finalDepth, numNeurons])
    Bf = bias_variable([numNeurons])
    network = tf.nn.relu(tf.matmul(Xf, Wf) + Bf)

    #Build the output layer. One neuron per class. 
    #For each letter position we need probability of each letter type.
    #Number of neurons = number of classes = (letter positions) * (number of possible letters)
    numNeurons = INPUT_CHAR_COUNT * len(ALLOWED_CHARS)
    #Dimension of weight matrix = number of neurons from previous layer X number of classes
    previousNumNeurons = network.get_shape()[1].value
    Wo = weight_variable([previousNumNeurons, numNeurons])
    Bo = bias_variable([numNeurons])
    network = tf.matmul(network, Wo) + Bo

    return network

def train_tf():
    trainingImages, trainingClasses, trainingStringResults = gen_images(2000)
    network = build_model_tf()

    
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

#network = build_model_tf()
#print network.get_shape()

#train()
# predict()
trainingImages, trainingClasses, trainingStringResults = gen_images(1)
print trainingClasses[0]

#plt.imshow(trainingImages[0])
# pil_im = Image.open('824.65.png', 'r')
# plt.imshow(np.asarray(pil_im))
#plt.show()
#pil_im.show()
