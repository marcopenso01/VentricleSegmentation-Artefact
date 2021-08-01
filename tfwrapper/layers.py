import tensorflow as tf
import numpy as np
import configuration as config

from tfwrapper import utils

from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer


def max_pool_layer2d(x, kernel_size=(2, 2), strides=(2, 2), padding="SAME"):
    '''
    2D max pooling layer with standard 2x2 pooling as default
    '''

    kernel_size_aug = [1, kernel_size[0], kernel_size[1], 1]
    strides_aug = [1, strides[0], strides[1], 1]

    op = tf.nn.max_pool2d(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op


def max_pool_layer3d(x, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="SAME"):
    '''
    3D max pooling layer with 2x2x2 pooling as default
    '''

    kernel_size_aug = [1, kernel_size[0], kernel_size[1], kernel_size[2], 1]
    strides_aug = [1, strides[0], strides[1], strides[2], 1]

    op = tf.nn.max_pool3d(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op


def crop_and_concat_layer(inputs, axis=-1):

    '''
    Layer for cropping and stacking feature maps of different size along a different axis. 
    Currently, the first feature map in the inputs list defines the output size. 
    The feature maps can have different numbers of channels. 
    :param inputs: A list of input tensors of the same dimensionality but can have different sizes
    :param axis: Axis along which to concatentate the inputs
    :return: The concatentated feature map tensor
    '''

    output_size = inputs[0].get_shape().as_list()
    concat_inputs = [inputs[0]]

    for ii in range(1,len(inputs)):

        larger_size = inputs[ii].get_shape().as_list()
        start_crop = np.subtract(larger_size, output_size) // 2

        if len(output_size) == 5:  # 3D images
            cropped_tensor = inputs[ii][:,
                             start_crop[1]:start_crop[1] + output_size[1],
                             start_crop[2]:start_crop[2] + output_size[2],
                             start_crop[3]:start_crop[3] + output_size[3],...]
        elif len(output_size) == 4:  # 2D images
            cropped_tensor = inputs[ii][:,
                             start_crop[1]:start_crop[1] + output_size[1],
                             start_crop[2]:start_crop[2] + output_size[2], ...]
        else:
            raise ValueError('Unexpected number of dimensions on tensor: %d' % len(output_size))

        concat_inputs.append(cropped_tensor)

    return tf.concat(concat_inputs, axis=axis)


def pad_to_size(bottom, output_size):

    ''' 
    A layer used to pad the tensor bottom to output_size by padding zeros around it
    TODO: implement for 3D data
    '''

    input_size = bottom.get_shape().as_list()
    size_diff = np.subtract(output_size, input_size)

    pad_size = size_diff // 2
    odd_bit = np.mod(size_diff, 2)

    if len(input_size) == 4:

        padded =  tf.pad(bottom, paddings=[[0,0],
                                        [pad_size[1], pad_size[1] + odd_bit[1]],
                                        [pad_size[2], pad_size[2] + odd_bit[2]],
                                        [0,0]])

        return padded

    elif len(input_size) == 5:
        raise NotImplementedError('This layer has not yet been extended to 3D')
    else:
        raise ValueError('Unexpected input size: %d' % input_size)


def dropout_layer(bottom, name, training, keep_prob=0.5):
    '''
    Performs dropout on the activations of an input
    '''

    keep_prob_pl = tf.cond(training,
                           lambda: tf.constant(keep_prob, dtype=bottom.dtype),
                           lambda: tf.constant(1.0, dtype=bottom.dtype))

    # The tf.nn.dropout function takes care of all the scaling
    # (https://www.tensorflow.org/get_started/mnist/pros)
    return tf.nn.dropout(bottom, keep_prob=keep_prob_pl, name=name)


def batch_normalisation_layer(bottom, name, training):
    '''
    Alternative wrapper for tensorflows own batch normalisation function. I had some issues with 3D convolutions
    when using this. The version below works with 3D convs as well.
    :param bottom: Input layer (should be before activation)
    :param name: A name for the computational graph
    :param training: A tf.bool specifying if the layer is executed at training or testing time
    :return: Batch normalised activation
    '''

    h_bn = tf.contrib.layers.batch_norm(inputs=bottom, decay=0.99, epsilon=1e-3, is_training=training,
                                        scope=name, center=True, scale=True)

    return h_bn


def Upsample(tensor, rate=2):
    return tf.image.resize_bilinear(tensor, (tf.shape(tensor)[1] * rate, tf.shape(tensor)[2] * rate))


### FEED_FORWARD LAYERS ##############################################################################

def conv2D_layer(bottom,
                 name,
                 kernel_size=(3,3),
                 num_filters=32,
                 strides=(1,1),
                 activation=tf.identity,
                 padding="SAME",
                 weight_init='he_normal',
                 add_bias=True):

    '''
    Standard 2D convolutional layer
    bottom = input data
    '''
    #number of input channels
    bottom_num_filters = bottom.get_shape().as_list()[-1]
    
    #define weights and bias structures
    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], 1]

    with tf.compat.v1.variable_scope(name):
        #initialise weights
        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)
        op = tf.nn.conv2d(bottom, filter=weights, strides=strides_augm, padding=padding)

        biases = None
        if add_bias:
            #initialise bias for the filter
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        # apply activation
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op
    
    
# dense block
def dense_block(bottom,
                name,
                training,
                kernel_size=(3, 3),
                num_filters=32,
                strides=(1, 1),
                activation=tf.nn.relu,
                padding="SAME",
                weight_init='he_normal',
                n_layers=4):
    '''
    dense block of 4 conv layers (default n_layers=4) where the input 
    consists of k feature maps. 
    Each conv layer outputs k/4 feature maps which is concatenated with 
    the input to all the next conv layers. The output of all these conv 
    layers are then concatenated to obtain k output feature maps.
    This is added with the input and sent to the next layer in the network.
    MOre details here: https://towardsdatascience.com/paper-review-densenet-densely-connected-convolutional-networks-acf9065dfefb
    '''
    #bottom_num_filters = bottom.get_shape().as_list()[-1]
    
    x = batch_normalisation_layer(bottom, name+'_bn0', training)
    
    x = activation(x)
    
    #real dense block provides a 1x1 conv layer (Bottleneck Layer).
    #here it is not implemented
    x = conv2D_layer(bottom=x,
                     name=name+'_0'),
                     kernel_size=kernel_size,
                     num_filters=num_filters/n_layers,
                     strides=strides,
                     activation=tf.identity,
                     padding=padding,
                     weight_init=weight_init,
                     add_bias=False)
        
    concat_feat = tf.concat([bottom, x], axis=-1)
    
    for i in range(1, n_layers):
        
        x = batch_normalisation_layer(concat_feat, name+'_bn'+str(i), training)
    
        x = activation(x)
    
        x = conv2D_layer(bottom=x,
                         name=name+'_'+str(i),
                         kernel_size=kernel_size,
                         num_filters=num_filters/n_layers,
                         strides=strides,
                         activation=tf.identity,
                         padding=padding,
                         weight_init=weight_init,
                         add_bias=False)
        
        concat_feat = tf.concat([concat_feat, x], axis=-1)

    return concat_feat


def conv3D_layer(bottom,
                 name,
                 kernel_size=(3,3,3),
                 num_filters=32,
                 strides=(1,1,1),
                 activation=tf.nn.relu,
                 padding="SAME",
                 weight_init='he_normal',
                 add_bias=True):

    '''
    Standard 3D convolutional layer
    '''

    bottom_num_filters = bottom.get_shape().as_list()[-1]

    weight_shape = [kernel_size[0], kernel_size[1], kernel_size[2], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], strides[2], 1]

    with tf.compat.v1.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)
        op = tf.nn.conv3d(bottom, filter=weights, strides=strides_augm, padding=padding)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def deconv2D_layer(bottom,
                   name,
                   kernel_size=(4,4),
                   num_filters=32,
                   strides=(2,2),
                   output_shape=None,
                   activation=tf.nn.relu,
                   padding="SAME",
                   weight_init='he_normal',
                   add_bias=True):

    '''
    Standard 2D transpose (also known as deconvolution) layer. Default behaviour upsamples the input by a
    factor of 2. 
    '''

    bottom_shape = bottom.get_shape().as_list()
    if output_shape is None:
        output_shape = tf.stack([bottom_shape[0], bottom_shape[1]*strides[0], bottom_shape[2]*strides[1], num_filters])

    bottom_num_filters = bottom_shape[3]

    weight_shape = [kernel_size[0], kernel_size[1], num_filters, bottom_num_filters]
    bias_shape = [num_filters]
    strides_augm = [1, strides[0], strides[1], 1]

    with tf.compat.v1.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)

        op = tf.nn.conv2d_transpose(bottom,
                                    filter=weights,
                                    output_shape=output_shape,
                                    strides=strides_augm,
                                    padding=padding)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def deconv3D_layer(bottom,
                   name,
                   kernel_size=(4,4,4),
                   num_filters=32,
                   strides=(2,2,2),
                   output_shape=None,
                   activation=tf.nn.relu,
                   padding="SAME",
                   weight_init='he_normal',
                   add_bias=True):

    '''
    Standard 2D transpose (also known as deconvolution) layer. Default behaviour upsamples the input by a
    factor of 2. 
    '''

    bottom_shape = bottom.get_shape().as_list()

    if output_shape is None:
        output_shape = tf.stack([bottom_shape[0], bottom_shape[1]*strides[0], bottom_shape[2]*strides[1], bottom_shape[3]*strides[2], num_filters])

    bottom_num_filters = bottom_shape[4]

    weight_shape = [kernel_size[0], kernel_size[1], kernel_size[2], num_filters, bottom_num_filters]

    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], strides[2], 1]

    with tf.compat.v1.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)

        op = tf.nn.conv3d_transpose(bottom,
                                    filter=weights,
                                    output_shape=output_shape,
                                    strides=strides_augm,
                                    padding=padding)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def conv2D_dilated_layer(bottom,
                         name,
                         kernel_size=(3,3),
                         num_filters=32,
                         rate=1,
                         activation=tf.nn.relu,
                         padding="SAME",
                         weight_init='he_normal',
                         add_bias=True):

    '''
    2D dilated convolution layer. This layer can be used to increase the receptive field of a network. 
    It is described in detail in this paper: Yu et al, Multi-Scale Context Aggregation by Dilated Convolutions, 
    2015 (https://arxiv.org/pdf/1511.07122.pdf) 
    '''

    bottom_num_filters = bottom.get_shape().as_list()[3]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    with tf.compat.v1.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)

        op = tf.nn.atrous_conv2d(bottom, filters=weights, rate=rate, padding=padding)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


def dense_layer(bottom,
                name,
                hidden_units=512,
                activation=tf.nn.relu,
                weight_init='he_normal',
                add_bias=True):

    '''
    Dense a.k.a. fully connected layer
    '''

    bottom_flat = utils.flatten(bottom)
    bottom_rhs_dim = utils.get_rhs_dim(bottom_flat)

    weight_shape = [bottom_rhs_dim, hidden_units]
    bias_shape = [hidden_units]

    with tf.compat.v1.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)

        op = tf.matmul(bottom_flat, weights)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op


# Squeeze and Excitation
'''
Let’s add parameters to each channel of a convolutional block so that the 
network can adaptively adjust the weighting of each feature map. 
As described in https://arxiv.org/pdf/1709.01507.pdf
'''
def se_block(tensor,
             name,
             activation=tf.nn.relu,
             weight_init='he_normal',
             ratio=16):
    
    init = tensor
    #number of input channels
    num_filters = tensor.get_shape().as_list()[-1]
    
    se = tf.reduce_mean(init,[1,2], keepdims=True)
    
    se = dense_layer(se,
                     name=name+'_0',
                     hidden_units=num_filters // ratio,
                     activation=tf.nn.relu,
                     weight_init=weight_init,
                     add_bias=False)
    
    se = dense_layer(se,
                     name=name+'_1',
                     hidden_units=num_filters,
                     activation=tf.math.sigmoid,
                     weight_init=weight_init,
                     add_bias=False)
    
    #se = tf.reshape(se, [-1,1,1,num_filters])
    
    x = tf.math.multiply(init, se)
    
    return x


# Selective Kernel 
'''
It is a process in which convolutional layers can adaptively adjust their 
receptive field (RF) sizes.
Advantage: Feature map with different receptive field (RF) in order to collect 
multi-scale spatial information.
As described in https://arxiv.org/pdf/1903.06586.pdf
'''
def selective_kernel_block(bottom, 
                           name, 
                           training,
                           activation=tf.nn.relu,
                           kernel_size=(3,3),
                           strides=(1,1),
                           activation=tf.nn.relu,
                           padding0"SAME",
                           weight_init='he_normal',
                           M=2,
                           r=16):
    '''
    M: number of path
    r: number of parameters in the fuse operator
    G: controls the cardinality of each path
    '''
    input_feature = bottom.get_shape().as_list()[-1]
    net = bottom
    d = max(int(input_feature / r), 32)
    
    x = bottom
    
    xs = []
    
    for i in range(M):
        
        net = conv2D_dilated_layer_bn(bottom=x,
                                      name=name'_dil'+str(i),
                                      training=training,
                                      kernel_size=kernel_size,
                                      num_filters=input_feature,
                                      rate=1+i,
                                      activation=activation,
                                      padding=padding,
                                      weight_init=weight_init)
        
        xs.append(net)
    
    for i in range(M):   
        if i == 0:
            U = xs[0]
        else:
            U = tf.add(U, xs[i])
    
    gap = tf.reduce_mean(U,[1,2], keepdims=True)
    
    fc = dense_layer(bottom=gap,
                     name=name+'_fc',
                     hidden_units=d
                     activation=tf.identity,
                     weight_init=weight_init,
                     add_bias=False)
    bn = batch_normalisation_layer(fc, name+'_bn', training)
    act = activation(bn)
    
    att_vec = []
    
    for i in range(M):
        
        fcs = dense_layer(bottom=act,
                          name=name+'_fc'+str(i),
                          hidden_units=input_feature
                          activation=tf.identity,
                          weight_init=weight_init,
                          add_bias=False)
        fcs = tf.expand_dims(fcs, axis=1)
        fcs = tf.expand_dims(fcs, axis=1)
        fcs_softmax = tf.nn.softmax(fcs)
        fea_v = tf.multiply(fcs_softmax, xs[i])
        
        att_vec.append(fea_v)
     
    for i in range(M):
        if i == 0:
            y = att_vec[0]
        else:
            y = tf.add(y, att_vec[i])
    
    return y                  

# Convolutional block attention module (CBAM)
'''
Contains the implementation of Convolutional Block Attention Module(CBAM) block.
As described in https://arxiv.org/abs/1807.06521.
'''
def conv_block_att_module(bottom,
                          name,
                          kernel_size=(7,7),
                          ratio=8,
                          activation=tf.nn.relu,
                          weight_init='he_normal'):
  
    attention_feature = channel_attention(bottom=bottom, 
                                          name=name,
                                          ratio=ratio,
                                          activation=activation,
                                          weight_init=weight_init)
    
    attention_feature = spatial_attention(bottom=attention_feature,
                                          name=name,
                                          kernel_size=kernel_size,
                                          activation=activation,
                                          weight_init=weight_init)
    
    return attention_feature
    
 
def channel_attention(bottom,
                      name,
                      ratio=8,
                      activation=tf.nn.relu,
                      weight_init='he_normal'):
    
    channel = bottom.get_shape().as_list()[-1]
    
    avg_pool = tf.reduce_mean(bottom,[1,2], keepdims=True)
    
    avg_pool = dense_layer(bottom=avg_pool,
                           name=name+'_mpl0',
                           hidden_units=channel//ratio
                           activation=activation,
                           weight_init=weight_init,
                           add_bias=False)
    
    avg_pool = dense_layer(bottom=avg_pool,
                           name=name+'_mpl1',
                           hidden_units=channel
                           activation=activation,
                           weight_init=weight_init,
                           add_bias=False)
    
    max_pool = tf.reduce_max(bottom, axis=[1,2], keepdims=True) 
    
    max_pool = dense_layer(bottom=max_pool,
                           name=name+'_mpl2',
                           hidden_units=channel//ratio
                           activation=activation,
                           weight_init=weight_init,
                           add_bias=False)
    
    max_pool = dense_layer(bottom=max_pool,
                           name=name+'_mpl3',
                           hidden_units=channel
                           activation=activation,
                           weight_init=weight_init,
                           add_bias=False)
    
    scale = tf.add(avg_pool, max_pool)
    scale = tf.math.sigmoid(scale)
    
    return tf.multiply(bottom, scale)


def spatial_attention(bottom, 
                      name,
                      kernel_size=(7,7),
                      activation=tf.nn.relu,
                      weight_init='he_normal'):
    
    avg_pool = tf.reduce_mean(bottom, axis=[3], keepdims=True)
    max_pool = tf.reduce_max(bottom, axis=[3], keepdims=True)
    
    concat = tf.concat([avg_pool,max_pool], 3)
    
    concat = conv2D_layer(bottom=concat,
                          name=name+'_spatial',
                          kernel_size=kernel_size,
                          num_filters=1,
                          strides=(1,1),
                          activation=tf.identity,
                          padding="SAME",
                          weight_init=weight_init,
                          add_bias=False)
    
    concat = tf.math.sigmoid(concat)
    
    return tf.multiply(bottom, concat)


#Attention gate
'''Attention, in the context of image segmentation, is a way to highlight 
only the relevant activations during training. This reduces the computational 
resources wasted on irrelevant activations, providing the network with better 
generalization power. Essentially, the network can pay “attention” to certain 
parts of the image.
As described in https://arxiv.org/pdf/1804.03999.pdf
'''
def attention(down_tensor,
              up_tensor,
              weight_init='he_normal'):
    
    channel = down_tensor.get_shape().as_list()[-1]
    
    x = conv2D_layer(bottom=down_tensor,
                     name=name+'_g',
                     kernel_size=(1,1),
                     num_filters=channel,
                     strides=(2,2),
                     activation=tf.identity,
                     padding="VALID",
                     weight_init=weight_init,
                     add_bias=False)
    
    g = conv2D_layer(bottom=up_tensor,
                     name=name+'_x',
                     kernel_size=(1,1),
                     num_filters=channel,
                     strides=(1,1),
                     activation=tf.identity,
                     padding="SAME",
                     weight_init=weight_init,
                     add_bias=False)
    
    net = tf.add(g, x)
    net = tf.nn.relu(net)
    net = conv2D_layer(bottom=net,
                       name=name+'_x',
                       kernel_size=(1,1),
                       num_filters=1,
                       strides=(1,1),
                       activation=tf.identity,
                       padding="SAME",
                       weight_init=weight_init,
                       add_bias=False)
    net = tf.math.sigmoid(net)
    net = tf.image.resize_bilinear(net, (tf.shape(net)[1] * 2, tf.shape(net)[2] * 2))
    
    return tf.multiply(net, down_tensor)
              
              

### BATCH_NORM SHORTCUTS #####################################################################################

def conv2D_layer_bn(bottom,
                    name,
                    training,
                    kernel_size=(3,3),
                    num_filters=32,
                    strides=(1,1),
                    activation=tf.nn.relu,
                    padding="SAME",
                    weight_init='he_normal'):
    '''
    Shortcut for batch normalised 2D convolutional layer
    '''

    conv = conv2D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=False)

    conv_bn = batch_normalisation_layer(conv, name+'_bn', training)

    act = activation(conv_bn)

    return act


def conv3D_layer_bn(bottom,
                    name,
                    training,
                    kernel_size=(3,3,3),
                    num_filters=32,
                    strides=(1,1,1),
                    activation=tf.nn.relu,
                    padding="SAME",
                    weight_init='he_normal'):

    '''
    Shortcut for batch normalised 3D convolutional layer
    '''

    conv = conv3D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=False)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training)

    act = activation(conv_bn)

    return act


def deconv2D_layer_bn(bottom,
                      name,
                      training,
                      kernel_size=(4,4),
                      num_filters=32,
                      strides=(2,2),
                      output_shape=None,
                      activation=tf.nn.relu,
                      padding="SAME",
                      weight_init='he_normal'):
    '''
    Shortcut for batch normalised 2D transposed convolutional layer
    '''

    deco = deconv2D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=tf.identity,
                          padding=padding,
                          weight_init=weight_init,
                          add_bias=False)

    deco_bn = batch_normalisation_layer(deco, name + '_bn', training=training)

    act = activation(deco_bn)

    return act


def deconv3D_layer_bn(bottom,
                      name,
                      training,
                      kernel_size=(4,4,4),
                      num_filters=32,
                      strides=(2,2,2),
                      output_shape=None,
                      activation=tf.nn.relu,
                      padding="SAME",
                      weight_init='he_normal'):

    '''
    Shortcut for batch normalised 3D transposed convolutional layer
    '''

    deco = deconv3D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=tf.identity,
                          padding=padding,
                          weight_init=weight_init,
                          add_bias=False)

    deco_bn = batch_normalisation_layer(deco, name + '_bn', training=training)

    act = activation(deco_bn)

    return act


def conv2D_dilated_layer_bn(bottom,
                           name,
                           training,
                           kernel_size=(3,3),
                           num_filters=32,
                           rate=1,
                           activation=tf.nn.relu,
                           padding="SAME",
                           weight_init='he_normal'):

    '''
    Shortcut for batch normalised 2D dilated convolutional layer
    '''

    conv = conv2D_dilated_layer(bottom=bottom,
                                name=name,
                                kernel_size=kernel_size,
                                num_filters=num_filters,
                                rate=rate,
                                activation=tf.identity,
                                padding=padding,
                                weight_init=weight_init,
                                add_bias=False)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training=training)

    act = activation(conv_bn)

    return act


def residual_block(bottom,
                   name,
                   training,
                   kernel_size=(3,3),
                   num_filters=32,
                   strides=[1,1],
                   activation=tf.nn.relu,
                   padding="SAME",
                   weight_init='he_normal'):
    
    '''
    As described in https://arxiv.org/pdf/1711.10684.pdf
    '''
    
    x = batch_normalisation_layer(bottom, name+'_bn1', training)
    
    x = activation(x)
    
    res1 = conv2D_layer(bottom=x,
                        name=name+'_1',
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=(strides[0],strides[0]),
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=True)
    
    res1 = batch_normalisation_layer(res1, name+'_bn2', training)
    
    res1 = activation(res1)
    
    res2 = conv2D_layer(bottom=res1,
                        name=name+'_2',
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=(strides[1],strides[1]),
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=True)
    
    shortcut = conv2D_layer(bottom=bottom,
                            name=name+'_shortcut',
                            kernel_size=(1,1),
                            num_filters=num_filters,
                            strides=(strides[0],strides[0]),
                            activation=tf.identity,
                            padding=padding,
                            weight_init=weight_init,
                            add_bias=True)
    
    shortcut = batch_normalisation_layer(shortcut, name+'_bn3', training)
    
    output = tf.add(shortcut, res2)
    
    return output


def res_block_initial(bottom,
                      name,
                      training,
                      kernel_size=(3,3),
                      num_filters=32,
                      strides=[1,1],
                      activation=tf.nn.relu,
                      padding="SAME",
                      weight_init='he_normal'):
    
    '''
    As described in https://arxiv.org/pdf/1711.10684.pdf
    '''
    
    x = conv2D_layer(bottom=bottom,
                     name=name+'_1',
                     kernel_size=kernel_size,
                     num_filters=num_filters,
                     strides=(strides[0],strides[0]),
                     activation=tf.identity,
                     padding=padding,
                     weight_init=weight_init,
                     add_bias=True)
    
    x = batch_normalisation_layer(x, name+'_bn1', training)
    
    x = activation(x)
    
    x = conv2D_layer(bottom=x,
                     name=name+'_2',
                     kernel_size=kernel_size,
                     num_filters=num_filters,
                     strides=(strides[1],strides[1]),
                     activation=tf.identity,
                     padding=padding,
                     weight_init=weight_init,
                     add_bias=True)
    
    shortcut = conv2D_layer(bottom=bottom,
                            name=name+'_shortcut',
                            kernel_size=(1,1),
                            num_filters=num_filters,
                            strides=(1,1),
                            activation=tf.identity,
                            padding=padding,
                            weight_init=weight_init,
                            add_bias=True)
    
    shortcut = batch_normalisation_layer(shortcut, name+'_bn2', training)
    
    output = tf.add(shortcut, x)
    
    return output
                      
                      
def dense_layer_bn(bottom,
                   name,
                   training,
                   hidden_units=512,
                   activation=tf.nn.relu,
                   weight_init='he_normal'):

    '''
    Shortcut for batch normalised 2D dilated convolutional layer
    '''

    linact = dense_layer(bottom=bottom,
                         name=name,
                         hidden_units=hidden_units,
                         activation=tf.identity,
                         weight_init=weight_init,
                         add_bias=False)

    batchnorm = batch_normalisation_layer(linact, name + '_bn', training=training)
    act = activation(batchnorm)

    return act

### VARIABLE INITIALISERS ####################################################################################

def get_weight_variable(shape, name=None, type='xavier_uniform', regularize=True, **kwargs):
    
    type = config.weight_init
    initialise_from_constant = False
    if type == 'xavier_uniform':
        initial = xavier_initializer(uniform=True, dtype=tf.float32)
    elif type == 'xavier_normal':
        initial = xavier_initializer(uniform=False, dtype=tf.float32)
    elif type == 'he_normal':
        initial = variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'he_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'caffe_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=1.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'simple':
        stddev = kwargs.get('stddev', 0.02)
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        initialise_from_constant = True
    elif type == 'bilinear':
        weights = _bilinear_upsample_weights(shape)
        initial = tf.constant(weights, shape=shape, dtype=tf.float32)
        initialise_from_constant = True
    else:
        raise ValueError('Unknown initialisation requested: %s' % type)

    if name is None:  # This keeps to option open to use unnamed Variables
        weight = tf.Variable(initial)
    else:
        if initialise_from_constant:
            weight = tf.compat.v1.get_variable(name, initializer=initial)
        else:
            weight = tf.compat.v1.get_variable(name, shape=shape, initializer=initial)

    if regularize:
        tf.compat.v1.add_to_collection('weight_variables', weight)

    return weight


def get_bias_variable(shape, name=None, init_value=0.0):

    initial = tf.constant(init_value, shape=shape, dtype=tf.float32)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.compat.v1.get_variable(name, initializer=initial)



def _upsample_filt(size):
    '''
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def _bilinear_upsample_weights(shape):
    '''
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    '''

    if not shape[0] == shape[1]: raise ValueError('kernel is not square')
    if not shape[2] == shape[3]: raise ValueError('input and output featuremaps must have the same size')

    kernel_size = shape[0]
    num_feature_maps = shape[2]

    weights = np.zeros(shape, dtype=np.float32)
    upsample_kernel = 
    
    (kernel_size)

    for i in range(num_feature_maps):
        weights[:, :, i, i] = upsample_kernel

    return weights

def _add_summaries(op, weights, biases):

    # Tensorboard variables
    tf.compat.v1.summary.histogram(weights.name, weights)
    if biases:
        tf.compat.v1.summary.histogram(biases.name, biases)
    tf.compat.v1.summary.histogram(op.op.name + '/activations', op)
