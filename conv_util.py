import tensorflow as tf

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv_layer(inputTensor, numFilters, filterSize, stride):
    inputDepth = inputTensor.get_shape()[3].value
    W = weight_variable([filterSize, filterSize, inputDepth, numFilters])
    b = bias_variable([numFilters])

    return tf.nn.relu(tf.nn.conv2d(inputTensor, W, strides=[1, stride, stride, 1], padding='SAME') + b)



def max_pool_layer(inputTensor, windowSize, stride):
  return tf.nn.max_pool(inputTensor, ksize=[1, windowSize, windowSize, 1],
                        strides=[1, stride, stride, 1], padding='SAME')

