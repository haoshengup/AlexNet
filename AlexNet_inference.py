import tensorflow as tf

#conv1
CONV1_SIZE = 11
CONV1_DEEP = 96
CONV1_STRIDE = 4
CONV1_MODE = 'VALID'

POOL1_SIZE = 3
POOL1_STRIDE = 2

#conv2
CONV2_SIZE = 5
CONV2_DEEP = 256
CONV2_STRIDE = 1
CONV2_MODE = 'SAME'

POOL2_SIZE = 3
POOL2_STRIDE = 2

#conv3
CONV3_SIZE = 3
CONV3_DEEP = 384
CONV3_STRIDE = 1
CONV3_MODE = 'SAME'

#conv4
CONV4_SIZE = 3
CONV4_DEEP = 384
CONV4_STRIDE = 1
CONV4_MODE = 'SAME'

#conv5
CONV5_SIZE = 3
CONV5_DEEP = 256
CONV5_STRIDE = 1
CONV5_MODE = 'SAME'

POOL5_SIZE = 3
POOL5_STRIDE = 2

#FC6
FC6_NODE = 4096
FC6_KEEP_PROB = 0.5

#FC7
FC7_NODE = 4096
FC7_KEEP_PROB = 0.5

#FC8
FC8_NODE = 1000

def inference(input):
    with tf.variable_scope('layer1'):
        weights = tf.get_variable('weights', [CONV1_SIZE, CONV1_SIZE, 3, CONV1_DEEP],\
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.1))
        conv1 = tf.nn.conv2d(input, weights, [1, CONV1_STRIDE, CONV1_STRIDE, 1], padding=CONV1_MODE)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias))
        pool1 = tf.nn.max_pool(relu1, [1, POOL1_SIZE, POOL1_SIZE, 1], [1, POOL1_STRIDE, POOL1_STRIDE, 1], padding='VALID')

    with tf.variable_scope('layer2'):
        weights = tf.get_variable('weights', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], \
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(pool1, weights, [1, CONV2_STRIDE, CONV2_STRIDE, 1], padding=CONV2_MODE)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias))
        pool2 = tf.nn.max_pool(relu2, [1, POOL2_SIZE, POOL2_SIZE, 1], [1, POOL2_STRIDE, POOL2_STRIDE, 1], padding='VALID')

    with tf.variable_scope('layer3'):
        weights = tf.get_variable('weights', [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP], \
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [CONV3_DEEP], initializer=tf.constant_initializer(0.1))
        conv3 = tf.nn.conv2d(pool2, weights, [1, CONV3_STRIDE, CONV3_STRIDE, 1], padding=CONV3_MODE)
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, bias))

    with tf.variable_scope('layer4'):
        weights = tf.get_variable('weights', [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP], \
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [CONV4_DEEP], initializer=tf.constant_initializer(0.1))
        conv4 = tf.nn.conv2d(relu3, weights, [1, CONV4_STRIDE, CONV4_STRIDE, 1], padding=CONV4_MODE)
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, bias))

    with tf.variable_scope('layer5'):
        weights = tf.get_variable('weights', [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP], \
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [CONV5_DEEP], initializer=tf.constant_initializer(0.1))
        conv5 = tf.nn.conv2d(relu4, weights, [1, CONV5_STRIDE, CONV5_STRIDE, 1], padding=CONV5_MODE)
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, bias))
        pool5 = tf.nn.max_pool(relu5, [1, POOL5_SIZE, POOL5_SIZE, 1], [1, POOL5_STRIDE, POOL5_STRIDE, 1], padding='VALID')
        pool5_shape = pool5.get_shape().as_list()
        node = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
        pool5_reshape = tf.reshape(pool5, [pool5_shape[0], node])

    with tf.variable_scope('layer6'):
        weights = tf.get_variable('weights', [node, FC6_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [FC6_NODE], initializer=tf.constant_initializer(0.1))
        relu6 = tf.nn.relu(tf.matmul(pool5_reshape, weights) + bias)
        relu6 = tf.nn.dropout(relu6, FC6_KEEP_PROB)

    with tf.variable_scope('layer7'):
        weights = tf.get_variable('weights', [FC6_NODE, FC7_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [FC7_NODE], initializer=tf.constant_initializer(0.1))
        relu7 = tf.nn.relu(tf.matmul(relu6, weights) + bias)
        relu7 = tf.nn.dropout(relu7, FC7_KEEP_PROB)

    with tf.variable_scope('layer8'):
        weights = tf.get_variable('weights', [FC7_NODE, FC8_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [FC8_NODE], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(relu7, weights) + bias

    return logit

















