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

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=512, weight_init='bilinear', training=training)
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

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256, weight_init='bilinear', training=training)
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

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128, weight_init='bilinear', training=training)
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

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, weight_init='bilinear', training=training)
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


def ResUNet(images, training, nlabels):
    
    #encoder
    e1 = layers.res_block_initial(images, 'e1', num_filters=64, strides=[1,1], training=training)
    e2 = layers.residual_block(e1, 'e2', num_filters=128, strides=[2,1], training=training)
    e3 = layers.residual_block(e2, 'e3', num_filters=256, strides=[2,1], training=training)
    
    #bridge layer, number of filters is double that of the last encoder layer
    b0 = layers.residual_block(e3, 'b0', num_filters=512, strides=[2,1], training=training)
    
    #decoder
    up3 = layers.deconv2D_layer_bn(b0, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256, weight_init='bilinear', training=training)
    concat3 = tf.concat([up3, e3], axis=3,  name='concat3')
    d3 = layers.residual_block(concat3, 'd3', num_filters=256, strides=[1,1], training=training)
    
    up2 = layers.deconv2D_layer_bn(d3, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128, weight_init='bilinear', training=training)
    concat2 = tf.concat([up2, e2], axis=3,  name='concat2')
    d2 = layers.residual_block(concat2, 'd2', num_filters=128, strides=[1,1], training=training)
    
    up1 = layers.deconv2D_layer_bn(d2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, weight_init='bilinear', training=training)
    concat1 = tf.concat([up1, e1], axis=3,  name='concat1')
    d1 = layers.residual_block(concat1, 'd1', num_filters=64, strides=[1,1], training=training)
    
    pred = layers.conv2D_layer(d1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)
    
    return pred
    
    
def DenseUNet(images, training, nlabels):
    
    #encoder
    conv1 = layers.conv2D_layer_bn(images, 'conv1', num_filters=64, training=training)
    dens1 = layers.dense_block(conv1, 'dens1', num_filters=64, n_layers=4, training=training)
    pool1 = layers.max_pool_layer2d(dens1)
    
    conv2 = layers.conv2D_layer_bn(pool1, 'conv2', num_filters=128, training=training)
    dens2 = layers.dense_block(conv2, 'dens2', num_filters=128, n_layers=4, training=training)
    pool2 = layers.max_pool_layer2d(dens2)
    
    conv3 = layers.conv2D_layer_bn(pool2, 'conv3', num_filters=256, training=training)
    dens3 = layers.dense_block(conv3, 'dens3', num_filters=256, n_layers=4, training=training)
    pool3 = layers.max_pool_layer2d(dens3)
    
    conv4 = layers.conv2D_layer_bn(pool3, 'conv4', num_filters=512, training=training)
    dens4 = layers.dense_block(conv4, 'dens4', num_filters=512, n_layers=4, training=training)
    pool4 = layers.max_pool_layer2d(dens4)
    
    #bridge
    conv5 = layers.conv2D_layer_bn(pool4, 'conv5', num_filters=1024, training=training)
    dens5 = layers.dense_block(conv5, 'dens5', num_filters=1024, n_layers=4, training=training)
    
    #decoder
    upconv4 = layers.deconv2D_layer_bn(dens5, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=512, weight_init='bilinear', training=training)
    concat4 = tf.concat([upconv4, dens4], axis=3, name='concat4')
    
    conv6 = layers.conv2D_layer_bn(concat4, 'conv6', num_filters=512, training=training)
    dens6 = layers.dense_block(conv6, 'dens6', num_filters=512, n_layers=4, training=training)
    
    upconv3 = layers.deconv2D_layer_bn(dens6, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256, weight_init='bilinear', training=training)
    concat3 = tf.concat([upconv3, dens3], axis=3, name='concat3')
    
    conv7 = layers.conv2D_layer_bn(concat3, 'conv7', num_filters=256, training=training)
    dens7 = layers.dense_block(conv7, 'dens7', num_filters=256, n_layers=4, training=training)
    
    upconv2 = layers.deconv2D_layer_bn(dens7, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128, weight_init='bilinear', training=training)
    concat2 = tf.concat([upconv2, dens2], axis=3, name='concat2')
    
    conv8 = layers.conv2D_layer_bn(concat2, 'conv8', num_filters=128, training=training)
    dens8 = layers.dense_block(conv8, 'dens8', num_filters=128, n_layers=4, training=training)
    
    upconv1 = layers.deconv2D_layer_bn(dens8, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, weight_init='bilinear', training=training)
    concat1 = tf.concat([upconv1, dens1], axis=3, name='concat1')
    
    conv9 = layers.conv2D_layer_bn(concat1, 'conv9', num_filters=64, training=training)
    dens9 = layers.dense_block(conv9, 'dens9', num_filters=64, n_layers=4, training=training)
    
    pred = layers.conv2D_layer_bn(dens9, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)

    return pred
    
