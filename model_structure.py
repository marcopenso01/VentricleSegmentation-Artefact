import tensorflow as tf
import configuration as config
from tfwrapper import layers
from tensorflow.contrib.layers.python.layers import initializers
slim = tf.contrib.slim
import logging


#same
def unet2D_same(images, training, nlabels):

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=64, training=training)
    logging.info('conv1_1')
    logging.info(conv1_1.shape)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training)
    logging.info('conv1_2')
    logging.info(conv1_2.shape)

    pool1 = layers.max_pool_layer2d(conv1_2)
    logging.info('pool1')
    logging.info(pool1.shape)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training)
    logging.info('conv2_1')
    logging.info(conv2_1.shape)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training)
    logging.info('conv2_2')
    logging.info(conv2_2.shape)
    
    pool2 = layers.max_pool_layer2d(conv2_2)
    logging.info('pool2')
    logging.info(pool2.shape)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training)
    logging.info('conv3_1')
    logging.info(conv3_1.shape)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training)
    logging.info('conv3_2')
    logging.info(conv3_2.shape)

    pool3 = layers.max_pool_layer2d(conv3_2)
    logging.info('pool3')
    logging.info(pool3.shape)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training)
    logging.info('conv4_1')
    logging.info(conv4_1.shape)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training)
    logging.info('conv4_2')
    logging.info(conv4_2.shape)

    pool4 = layers.max_pool_layer2d(conv4_2)
    logging.info('pool4')
    logging.info(pool4.shape)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training)
    logging.info('conv5_1')
    logging.info(conv5_1.shape)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training)
    logging.info('conv5_2')
    logging.info(conv5_2.shape)
    
    upconv4 = layers.Upsample(conv5_2)
    #upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=512, weight_init='bilinear', training=training)
    logging.info('upconv4')
    logging.info(upconv4.shape)
    concat4 = layers.crop_and_concat_layer([conv4_2, upconv4], axis=3)
    logging.info('concat4')
    logging.info(concat4.shape)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training)
    logging.info('conv6_1')
    logging.info(conv6_1.shape)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training)
    logging.info('conv6_2')
    logging.info(conv6_2.shape)
    
    upconv3 = layers.Upsample(conv6_2)
    #upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256, weight_init='bilinear', training=training)
    logging.info('upconv3')
    logging.info(upconv3.shape)
    concat3 = tf.concat([conv3_2, upconv3], axis=3, name='concat3')
    logging.info('concat3')
    logging.info(concat3.shape)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training)
    logging.info('conv7_1')
    logging.info(conv7_1.shape)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training)
    logging.info('conv7_2')
    logging.info(conv7_2.shape)

    upconv2 = layers.Upsample(conv7_2)
    #upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128, weight_init='bilinear', training=training)
    logging.info('upconv2')
    logging.info(upconv2.shape)
    concat2 = tf.concat([conv2_2, upconv2], axis=3, name='concat2')
    logging.info('concat2')
    logging.info(concat2.shape)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training)
    logging.info('conv8_1')
    logging.info(conv8_1.shape)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training)
    logging.info('conv8_2')
    logging.info(conv8_2.shape)

    upconv1 = layers.Upsample(conv8_2)
    #upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, weight_init='bilinear', training=training)
    logging.info('upconv1')
    logging.info(upconv1.shape)
    concat1 = tf.concat([conv1_2, upconv1], axis=3, name='concat1')
    logging.info('concat1')
    logging.info(concat1.shape)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training)
    logging.info('conv9_1')
    logging.info(conv9_1.shape)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training)
    logging.info('conv9_2')
    logging.info(conv9_2.shape)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)
    logging.info('pred')
    logging.info(pred.shape)
    
    return pred


#ResUnet
'''
As described in https://arxiv.org/pdf/1711.10684.pdf
'''
def ResUNet(images, training, nlabels):
    
    #encoder
    e1 = layers.res_block_initial(images, 'e1', num_filters=64, strides=[1,1], training=training)
    e2 = layers.residual_block(e1, 'e2', num_filters=128, strides=[2,1], training=training)
    e3 = layers.residual_block(e2, 'e3', num_filters=256, strides=[2,1], training=training)
    
    #bridge layer, number of filters is double that of the last encoder layer
    b0 = layers.residual_block(e3, 'b0', num_filters=512, strides=[2,1], training=training)
    
    #decoder
    up3 = layers.Upsample(b0)
    concat3 = tf.concat([up3, e3], axis=3,  name='concat3')
    d3 = layers.residual_block(concat3, 'd3', num_filters=256, strides=[1,1], training=training)
    
    up2 = layers.Upsample(d3)
    concat2 = tf.concat([up2, e2], axis=3,  name='concat2')
    d2 = layers.residual_block(concat2, 'd2', num_filters=128, strides=[1,1], training=training)
    
    up1 = layers.Upsample(d2)
    concat1 = tf.concat([up1, e1], axis=3,  name='concat1')
    d1 = layers.residual_block(concat1, 'd1', num_filters=64, strides=[1,1], training=training)
    
    pred = layers.conv2D_layer(d1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity)
    
    return pred
    

#Dense-UNet
'''
As described in https://arxiv.org/pdf/1709.07330.pdf
Downsampling: DenseNet-121
'''
def DenseUNet(images, training, nlabels):
    
    #encoder
    conv1 = layers.conv2D_layer_bn(images, 'conv1', num_filters=64, kernel_size=(7,7), strides=(2,2), training=training)
    
    pool1 = layers.max_pool_layer2d(conv1)

    dens1 = layers.dense_block(pool1, 'dens1', growth_rate=32, n_layers=6, training=training)
    trans1 = layers.transition_layer(dens1, 'trans1', num_filters=128, training=training)
    
    dens2 = layers.dense_block(trans1, 'dens2', growth_rate=32, n_layers=12, training=training)
    trans2 = layers.transition_layer(dens2, 'trans2', num_filters=256, training=training)
    
    dens3 = layers.dense_block(trans2, 'dens3', growth_rate=32, n_layers=24, training=training)
    trans3 = layers.transition_layer(dens3, 'trans3', num_filters=512, training=training)
    
    #bridge
    dens4 = layers.dense_block(trans3, 'dens4', growth_rate=32, n_layers=16, training=training)

    #decoder
    up1 = layers.Upsample(dens4)
    concat1 = tf.concat([up1, dens3], axis=3, name='concat1')
    conv2 = layers.conv2D_layer_bn(concat1, 'conv2', num_filters=640, kernel_size=(3,3), training=training)
    
    up2 = layers.Upsample(conv2)
    concat2 = tf.concat([up2, dens2], axis=3, name='concat2')
    conv3 = layers.conv2D_layer_bn(concat2, 'conv3', num_filters=256, kernel_size=(3,3), training=training)

    up3 = layers.Upsample(conv3)
    concat3 = tf.concat([up3, dens1], axis=3, name='concat3')
    conv4 = layers.conv2D_layer_bn(concat3, 'conv4', num_filters=64, kernel_size=(3,3), training=training)
    
    up4 = layers.Upsample(conv4)
    concat4 = tf.concat([up4, conv1], axis=3, name='concat4')
    conv5 = layers.conv2D_layer_bn(concat4, 'conv5', num_filters=64, kernel_size=(3,3), training=training)
    
    up5 = layers.Upsample(conv5)
    conv6 = layers.conv2D_layer_bn(up5, 'conv6', num_filters=48, kernel_size=(3,3), training=training)
    
    pred = layers.conv2D_layer_bn(conv6, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)

    return pred
    

#Proposed network
def net1(images, training, nlabels):

    #encoder
    e1 = layers.selective_kernel_block(images, 'e1', num_filters=32, training=training)
    conv1 = layer.conv2D_layer_bn(e1, 'conv1', num_filters=32, training=training)
    conc1 = tf.concat([e1, conv1], axis=3, name='conc1')
    
    p1 = layers.max_pool_layer2d(conc1)
    
    e2 = layers.selective_kernel_block(p1, 'e2', num_filters=64, training=training)
    conv2 = layer.conv2D_layer_bn(e2, 'conv2', num_filters=64, training=training)
    conc2 = tf.concat([e2, conv2], axis=3, name='conc2')
    
    p2 = layers.max_pool_layer2d(conc2)
    
    e3 = layers.selective_kernel_block(p2, 'e3', num_filters=128, training=training)
    conv3 = layer.conv2D_layer_bn(e3, 'conv3', num_filters=128, training=training)
    conc3 = tf.concat([e3, conv3], axis=3, name='conc3')
    
    p3 = layers.max_pool_layer2d(conc3)
    
    e4 = layers.selective_kernel_block(p3, 'e4', num_filters=256, training=training)
    conv4 = layer.conv2D_layer_bn(e4, 'conv4', num_filters=256, training=training)
    conc4 = tf.concat([e4, conv4], axis=3, name='conc4')
    
    p4 = layers.max_pool_layer2d(conc4)
    
    #bridge
    conv5 = layer.conv2D_layer_bn(p4, 'conv5', num_filters=256, training=training)
    cbam = layer.conv_block_att_module(conv5, 'cbam')
    
    #decoder    
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training)
    logging.info('conv1_2')
    logging.info(conv1_2.shape)

    pool1 = layers.max_pool_layer2d(conv1_2)
    logging.info('pool1')
    logging.info(pool1.shape)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training)
    logging.info('conv2_1')
    logging.info(conv2_1.shape)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training)
    logging.info('conv2_2')
    logging.info(conv2_2.shape)
    
    pool2 = layers.max_pool_layer2d(conv2_2)
    logging.info('pool2')
    logging.info(pool2.shape)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training)
    logging.info('conv3_1')
    logging.info(conv3_1.shape)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training)
    logging.info('conv3_2')
    logging.info(conv3_2.shape)

    pool3 = layers.max_pool_layer2d(conv3_2)
    logging.info('pool3')
    logging.info(pool3.shape)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training)
    logging.info('conv4_1')
    logging.info(conv4_1.shape)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training)
    logging.info('conv4_2')
    logging.info(conv4_2.shape)

    pool4 = layers.max_pool_layer2d(conv4_2)
    logging.info('pool4')
    logging.info(pool4.shape)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training)
    logging.info('conv5_1')
    logging.info(conv5_1.shape)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training)
    logging.info('conv5_2')
    logging.info(conv5_2.shape)
    
    upconv4 = layers.Upsample(conv5_2)
    #upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=512, weight_init='bilinear', training=training)
    logging.info('upconv4')
    logging.info(upconv4.shape)
    concat4 = layers.crop_and_concat_layer([conv4_2, upconv4], axis=3)
    logging.info('concat4')
    logging.info(concat4.shape)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training)
    logging.info('conv6_1')
    logging.info(conv6_1.shape)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training)
    logging.info('conv6_2')
    logging.info(conv6_2.shape)
    
    upconv3 = layers.Upsample(conv6_2)
    #upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256, weight_init='bilinear', training=training)
    logging.info('upconv3')
    logging.info(upconv3.shape)
    concat3 = tf.concat([conv3_2, upconv3], axis=3, name='concat3')
    logging.info('concat3')
    logging.info(concat3.shape)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training)
    logging.info('conv7_1')
    logging.info(conv7_1.shape)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training)
    logging.info('conv7_2')
    logging.info(conv7_2.shape)

    upconv2 = layers.Upsample(conv7_2)
    #upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128, weight_init='bilinear', training=training)
    logging.info('upconv2')
    logging.info(upconv2.shape)
    concat2 = tf.concat([conv2_2, upconv2], axis=3, name='concat2')
    logging.info('concat2')
    logging.info(concat2.shape)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training)
    logging.info('conv8_1')
    logging.info(conv8_1.shape)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training)
    logging.info('conv8_2')
    logging.info(conv8_2.shape)

    upconv1 = layers.Upsample(conv8_2)
    #upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, weight_init='bilinear', training=training)
    logging.info('upconv1')
    logging.info(upconv1.shape)
    concat1 = tf.concat([conv1_2, upconv1], axis=3, name='concat1')
    logging.info('concat1')
    logging.info(concat1.shape)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training)
    logging.info('conv9_1')
    logging.info(conv9_1.shape)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training)
    logging.info('conv9_2')
    logging.info(conv9_2.shape)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)
    logging.info('pred')
    logging.info(pred.shape)
    
    return pred
