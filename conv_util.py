import tensorflow as tf

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
  return tf.Variable(tf.ones(shape) / 10.0)

def conv_layer(inputTensor, depth, filterSize, stride):
    inputDepth = inputTensor.get_shape()[3].value
    W = weight_variable([filterSize, filterSize, inputDepth, depth])
    b = bias_variable([depth])

    return tf.nn.relu(tf.nn.conv2d(inputTensor, W, strides=[1, stride, stride, 1], padding='SAME') + b)

def fully_connected_layer(inputTensor, numNeurons):
    inputHeight = inputTensor.get_shape()[1].value
    inputWidth = inputTensor.get_shape()[2].value
    inputDepth = inputTensor.get_shape()[3].value

    #Build the fully connected layer.
    #A fully connected layer can work with depth 1 input only.
    #Unroll the output from the last pooling layer into a 2D matrix
    #The X is already transposed this way
    Xf = tf.reshape(inputTensor, shape=[-1, inputHeight * inputWidth * inputDepth])
    Wf = weight_variable([inputHeight * inputWidth * inputDepth, numNeurons])
    Bf = bias_variable([numNeurons])

    return tf.nn.relu(tf.matmul(Xf, Wf) + Bf)

def readout_layer(inputTensor, numClasses):
    W = weight_variable([inputTensor.get_shape()[1].value, numClasses])
    B = bias_variable([numClasses])
    Ylogits = tf.matmul(inputTensor, W) + B
    Y = tf.nn.softmax(Ylogits)

    return (Ylogits, Y)

def create_optimizer(Ylogits, predictions, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=labels)
    cross_entropy = tf.reduce_mean(cross_entropy) * 100

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Learning rate
    lr = tf.placeholder(tf.float32)

    # training step, the learning rate is a placeholder
    graph = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    return (graph, lr, accuracy)

def max_pool_layer(inputTensor, windowSize, stride):
  return tf.nn.max_pool(inputTensor, ksize=[1, windowSize, windowSize, 1],
                        strides=[1, stride, stride, 1], padding='SAME')

