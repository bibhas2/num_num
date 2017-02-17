import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import random
import numpy as np
import tensorflow as tf
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

def build_model_tf():
    #Input data - a tensor of RGB (3 channel) images
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

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
    Ylogits = tf.matmul(network, Wo) + Bo
    network = tf.nn.softmax(Ylogits)

    return (network, Ylogits, X)

def train_tf():
    BATCH_SIZE = 500
    ITERATION_COUNT = 10

    (network, Ylogits, X) = build_model_tf()
    #For training classes
    Y_ = tf.placeholder(tf.float32, [None, INPUT_CHAR_COUNT * len(ALLOWED_CHARS)])
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy) * BATCH_SIZE
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(0, ITERATION_COUNT):
        trainingImages, trainingClasses, trainingStringResults = gen_images(BATCH_SIZE)
        
        sess.run(train_step, {X: trainingImages, Y_: trainingClasses})

def classToString(classVector):
    classMatrix = classVector.reshape((INPUT_CHAR_COUNT, len(ALLOWED_CHARS)))
    charIndices = np.argmax(classMatrix, 1)
    result = ""
    for i in charIndices:
        result += str(ALLOWED_CHARS[i])
    
    return result

#network = build_model_tf()
#print network.get_shape()

train_tf()
# predict()
# trainingImages, trainingClasses, trainingStringResults = gen_images(1)
# print trainingClasses[0]
# print trainingStringResults[0]
# print classToString(trainingClasses[0])

#plt.imshow(trainingImages[0])
# pil_im = Image.open('824.65.png', 'r')
# plt.imshow(np.asarray(pil_im))
#plt.show()
#pil_im.show()
