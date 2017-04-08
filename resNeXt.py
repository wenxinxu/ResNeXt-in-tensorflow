# Coder: Wenxin Xu
# Source paper: https://arxiv.org/abs/1611.05431
# Github: https://github.com/wenxinxu/ResNeXt-in-tensorflow
# ==============================================================================
'''
This is main body of the ResNext structure
'''

import numpy as np
from hyper_parameters import *

BN_EPSILON = 0.001

def activation_summary(x):
    '''
    Add histogram and sparsity summaries of a tensor to tensorboard
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    Create a variable with tf.get_variable()
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    '''
    Generate the output layer
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride, relu=True):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :param relu: boolean. Relu after BN?
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    if relu is True:
        output = tf.nn.relu(bn_layer)
    else:
        output = bn_layer
    return output


def split(input_layer, stride):
    '''
    The split structure in Figure 3b of the paper. It takes an input tensor. Conv it by [1, 1,
    64] filter, and then conv the result by [3, 3, 64]. Return the
    final resulted tensor, which is in shape of [batch_size, input_height, input_width, 64]

    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, input_height, input_width, input_channel/64]
    '''

    input_channel = input_layer.get_shape().as_list()[-1]
    num_filter = FLAGS.block_unit_depth
    # according to Figure 7, they used 64 as # filters for all cifar10 task

    with tf.variable_scope('bneck_reduce_size'):
        conv = conv_bn_relu_layer(input_layer, filter_shape=[1, 1, input_channel, num_filter],
                                  stride=stride)
    with tf.variable_scope('bneck_conv'):
        conv = conv_bn_relu_layer(conv, filter_shape=[3, 3, num_filter, num_filter], stride=1)

    return conv


def bottleneck_b(input_layer, stride):
    '''
    The bottleneck strucutre in Figure 3b. Concatenates all the splits
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
    '''
    split_list = []
    for i in range(FLAGS.cardinality):
        with tf.variable_scope('split_%i'%i):
            splits = split(input_layer=input_layer, stride=stride)
        split_list.append(splits)

    # Concatenate splits and check the dimension
    concat_bottleneck = tf.concat(values=split_list, axis=3, name='concat')

    return concat_bottleneck


def bottleneck_c1(input_layer, stride):
    '''
    The bottleneck strucutre in Figure 3c. Grouped convolutions
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
    '''
    input_channel = input_layer.get_shape().as_list()[-1]
    bottleneck_depth = FLAGS.block_unit_depth
    with tf.variable_scope('bottleneck_c_l1'):
        l1 = conv_bn_relu_layer(input_layer=input_layer,
                                filter_shape=[1, 1, input_channel, bottleneck_depth],
                                stride=stride)
    with tf.variable_scope('group_conv'):
        filter = create_variables(name='depthwise_filter', shape=[3, 3, bottleneck_depth, FLAGS.cardinality])
        l2 = tf.nn.depthwise_conv2d(input=l1,
                                    filter=filter,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
    return l2


def bottleneck_c(input_layer, stride):
    '''
    The bottleneck strucutre in Figure 3c. Grouped convolutions
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
    :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
    '''
    input_channel = input_layer.get_shape().as_list()[-1]
    bottleneck_depth = FLAGS.block_unit_depth * FLAGS.cardinality
    with tf.variable_scope('bottleneck_c_l1'):
        l1 = conv_bn_relu_layer(input_layer=input_layer,
                                filter_shape=[1, 1, input_channel, bottleneck_depth],
                                stride=stride)
    with tf.variable_scope('group_conv'):
        filter = create_variables(name='depthwise_filter', shape=[3, 3, bottleneck_depth, FLAGS.cardinality])
        l2 = conv_bn_relu_layer(input_layer=l1,
                                filter_shape=[3, 3, bottleneck_depth, bottleneck_depth],
                                stride=1)
    return l2


def resnext_block(input_layer, output_channel):
    '''
    The block structure in Figure 3b. Takes a 4D tensor as input layer and splits, concatenates
    the tensor and restores the depth. Finally adds the identity and ReLu.
    :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
    input_channel]
    :param output_channel: int, the number of channels of the output
    :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    if FLAGS.bottleneck_implementation == 'b':
        concat_bottleneck = bottleneck_b(input_layer, stride)
    else:
        assert FLAGS.bottleneck_implementation == 'c'
        concat_bottleneck = bottleneck_c(input_layer, stride)

    bottleneck_depth = concat_bottleneck.get_shape().as_list()[-1]
    assert bottleneck_depth == FLAGS.block_unit_depth * FLAGS.cardinality

    # Restore the dimension. Without relu here
    restore = conv_bn_relu_layer(input_layer=concat_bottleneck,
                                 filter_shape=[1, 1, bottleneck_depth, output_channel],
                                 stride=1, relu=False)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    # According to section 4 of the paper, relu is played after adding the identity.
    output = tf.nn.relu(restore + padded_input)

    return output


def inference(input_tensor_batch, n, reuse):
    '''
    The main function that defines the ResNeXt. total layers = 1 + 3n + 3n + 3n +1 = 9n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_resnext_blocks. The paper used n=3, 29 layers as demo
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 64], 1)
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            conv1 = resnext_block(layers[-1], 64)
            activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = resnext_block(layers[-1], 128)
            activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = resnext_block(layers[-1], 256)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [8, 8, 256]

    with tf.variable_scope('fc', reuse=reuse):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [256]
        output = output_layer(global_pool, 10)
        layers.append(output)

    return layers[-1]


def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference(input_tensor, FLAGS.num_resnext_blocks, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

# test_graph()
