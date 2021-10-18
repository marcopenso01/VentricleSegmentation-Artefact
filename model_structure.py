import tensorflow as tf

from tfwrapper import layers

import logging


#same
def unet2D_same(images, training, nlabels):

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=56, training=training)
    logging.info('conv1_1')
    logging.info(conv1_1.shape)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=56, training=training)
    logging.info('conv1_2')
    logging.info(conv1_2.shape)

    pool1 = layers.max_pool_layer2d(conv1_2)
    logging.info('pool1')
    logging.info(pool1.shape)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=112, training=training)
    logging.info('conv2_1')
    logging.info(conv2_1.shape)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=112, training=training)
    logging.info('conv2_2')
    logging.info(conv2_2.shape)
    
    pool2 = layers.max_pool_layer2d(conv2_2)
    logging.info('pool2')
    logging.info(pool2.shape)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=224, training=training)
    logging.info('conv3_1')
    logging.info(conv3_1.shape)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=224, training=training)
    logging.info('conv3_2')
    logging.info(conv3_2.shape)

    pool3 = layers.max_pool_layer2d(conv3_2)
    logging.info('pool3')
    logging.info(pool3.shape)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=448, training=training)
    logging.info('conv4_1')
    logging.info(conv4_1.shape)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=448, training=training)
    logging.info('conv4_2')
    logging.info(conv4_2.shape)

    pool4 = layers.max_pool_layer2d(conv4_2)
    logging.info('pool4')
    logging.info(pool4.shape)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=896, training=training)
    logging.info('conv5_1')
    logging.info(conv5_1.shape)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=896, training=training)
    logging.info('conv5_2')
    logging.info(conv5_2.shape)
    
    upconv4 = layers.upsample(conv5_2)
    #upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=512, weight_init='bilinear', training=training)
    logging.info('upconv4')
    logging.info(upconv4.shape)
    concat4 = layers.crop_and_concat_layer([conv4_2, upconv4], axis=3)
    logging.info('concat4')
    logging.info(concat4.shape)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=448, training=training)
    logging.info('conv6_1')
    logging.info(conv6_1.shape)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=448, training=training)
    logging.info('conv6_2')
    logging.info(conv6_2.shape)
    
    upconv3 = layers.upsample(conv6_2)
    #upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256, weight_init='bilinear', training=training)
    logging.info('upconv3')
    logging.info(upconv3.shape)
    concat3 = tf.concat([conv3_2, upconv3], axis=3, name='concat3')
    logging.info('concat3')
    logging.info(concat3.shape)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=224, training=training)
    logging.info('conv7_1')
    logging.info(conv7_1.shape)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=224, training=training)
    logging.info('conv7_2')
    logging.info(conv7_2.shape)

    upconv2 = layers.upsample(conv7_2)
    #upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128, weight_init='bilinear', training=training)
    logging.info('upconv2')
    logging.info(upconv2.shape)
    concat2 = tf.concat([conv2_2, upconv2], axis=3, name='concat2')
    logging.info('concat2')
    logging.info(concat2.shape)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=112, training=training)
    logging.info('conv8_1')
    logging.info(conv8_1.shape)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=112, training=training)
    logging.info('conv8_2')
    logging.info(conv8_2.shape)

    upconv1 = layers.upsample(conv8_2)
    #upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, weight_init='bilinear', training=training)
    logging.info('upconv1')
    logging.info(upconv1.shape)
    concat1 = tf.concat([conv1_2, upconv1], axis=3, name='concat1')
    logging.info('concat1')
    logging.info(concat1.shape)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=56, training=training)
    logging.info('conv9_1')
    logging.info(conv9_1.shape)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=56, training=training)
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
    up3 = layers.upsample(b0)
    concat3 = tf.concat([up3, e3], axis=3,  name='concat3')
    d3 = layers.residual_block(concat3, 'd3', num_filters=256, strides=[1,1], training=training)
    
    up2 = layers.upsample(d3)
    concat2 = tf.concat([up2, e2], axis=3,  name='concat2')
    d2 = layers.residual_block(concat2, 'd2', num_filters=128, strides=[1,1], training=training)
    
    up1 = layers.upsample(d2)
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
    nb_filter = 96

    #encoder
    conv1 = layers.conv2D_layer_bn(images, 'conv1', num_filters=nb_filter, kernel_size=(7,7), strides=(2,2), training=training)
    pool1 = layers.max_pool_layer2d(conv1)

    dens1, nb_filter = layers.dense_block(pool1, 'dens1', init_filt=192, growth_rate=48, n_layers=6,
                                          nb_filter=nb_filter, keep_prob=0.2, training=training)
    trans1 = layers.transition_layer(dens1, 'trans1', num_filters=nb_filter, training=training)
    
    dens2, nb_filter = layers.dense_block(trans1, 'dens2', init_filt=192, growth_rate=48, n_layers=12,
                                          nb_filter=nb_filter, keep_prob=0.2, training=training)
    trans2 = layers.transition_layer(dens2, 'trans2', num_filters=nb_filter, training=training)
    
    dens3, nb_filter = layers.dense_block(trans2, 'dens3', init_filt=192, growth_rate=48, n_layers=36,
                                          nb_filter=nb_filter, keep_prob=0.2, training=training)
    trans3 = layers.transition_layer(dens3, 'trans3', num_filters=nb_filter, training=training)
    
    #bridge
    dens4, nb_filter = layers.dense_block(trans3, 'dens4', init_filt=192, growth_rate=48, n_layers=24,
                                          nb_filter=nb_filter, keep_prob=0.2, training=training)

    #decoder
    up1 = layers.upsample(dens4)
    line0 = layers.conv2D_layer(dens3, name='line0', kernel_size=(1,1), num_filters=2208)
    concat1 = tf.concat([up1, line0], axis=3, name='concat1')
    conv2 = layers.conv2D_layer_bn(concat1, 'conv2', num_filters=768, kernel_size=(3,3), training=training)
    
    up2 = layers.upsample(conv2)
    concat2 = tf.concat([up2, dens2], axis=3, name='concat2')
    conv3 = layers.conv2D_layer_bn(concat2, 'conv3', num_filters=384, kernel_size=(3,3), training=training)

    up3 = layers.upsample(conv3)
    concat3 = tf.concat([up3, dens1], axis=3, name='concat3')
    conv4 = layers.conv2D_layer_bn(concat3, 'conv4', num_filters=96, kernel_size=(3,3), training=training)
    
    up4 = layers.upsample(conv4)
    concat4 = tf.concat([up4, conv1], axis=3, name='concat4')
    conv5 = layers.conv2D_layer_bn(concat4, 'conv5', num_filters=96, kernel_size=(3,3), training=training)
    
    up5 = layers.upsample(conv5)
    conv6 = layers.conv2D_layer_bn_drop(up5, 'conv6', num_filters=64, kernel_size=(3,3), keep_prob=0.3, training=training)
    
    pred = layers.conv2D_layer_bn(conv6, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)

    print('input', images.shape)
    print('conv1', conv1.shape)
    print('pool1', pool1.shape)
    print('dens1', dens1.shape)
    print('trans1', trans1.shape)
    print('dens2', dens2.shape)
    print('trans2', trans2.shape)
    print('dens3', dens3.shape)
    print('trans3', trans3.shape)
    print('dens4', dens4.shape)
    print('up1', up1.shape)
    print('line0', line0.shape)
    print('concat1', concat1.shape)
    print('conv2', conv2.shape)
    print('up2', up2.shape)
    print('concat2', concat2.shape)
    print('conv3', conv3.shape)
    print('up3', up3.shape)
    print('concat3', concat3.shape)
    print('conv4', conv4.shape)
    print('up4', up4.shape)
    print('concat4', concat4.shape)
    print('conv5', conv5.shape)
    print('up5', up5.shape)
    print('conv6', conv6.shape)
    print('pred', pred.shape)
    return pred
    



#Proposed network
def mod1unet2D(images, training, nlabels):
    n_filt = 56     #48    56    64
    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=n_filt, training=training)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=n_filt, training=training)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=n_filt*2, training=training)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=n_filt*2, training=training)

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=n_filt*4, training=training)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=n_filt*4, training=training)

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=n_filt*8, training=training)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=n_filt*8, training=training)

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=n_filt*16, training=training)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=n_filt*16, training=training)

    deconv4 = layers.deconv2D_layer_bn(conv5_2, 'deconv4', num_filters=4, training=training)
    concat4 = tf.concat([conv4_2, deconv4], axis=-1, name='concat4')

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=n_filt*8, training=training)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=n_filt*8, training=training)

    deconv3 = layers.deconv2D_layer_bn(conv6_2, 'deconv3', num_filters=4, training=training)
    concat3 = tf.concat([conv3_2, deconv3], axis=-1, name='concat3')

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=n_filt*4, training=training)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=n_filt*4, training=training)

    deconv2 = layers.deconv2D_layer_bn(conv7_2, 'deconv2', num_filters=4, training=training)
    concat2 = tf.concat([conv2_2, deconv2], axis=-1, name='concat2')

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=n_filt*2, training=training)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=n_filt*2, training=training)

    deconv1 = layers.deconv2D_layer_bn(conv8_2, 'deconv1', num_filters=4, training=training)
    concat1 = tf.concat([conv1_2, deconv1], axis=-1, name='concat1')

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=n_filt, training=training)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=n_filt, training=training)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1, 1), activation=tf.identity,
                                  training=training)

    print('conv1_1', conv1_1.shape)
    print('conv1_2', conv1_2.shape)
    print('pool1', pool1.shape)
    print('conv2_1', conv2_1.shape)
    print('conv2_2', conv2_2.shape)
    print('pool2', pool2.shape)
    print('conv3_1', conv3_1.shape)
    print('conv3_2', conv3_2.shape)
    print('pool3', pool3.shape)
    print('conv4_1', conv4_1.shape)
    print('conv4_2', conv4_2.shape)
    print('pool4', pool4.shape)
    print('conv5_1', conv5_1.shape)
    print('conv5_2', conv5_2.shape)
    print('deconv4', deconv4.shape)
    print('concat4', concat4.shape)
    print('conv6_1', conv6_1.shape)
    print('conv6_2', conv6_2.shape)
    print('deconv3', deconv3.shape)
    print('concat3', concat3.shape)
    print('conv7_1', conv7_1.shape)
    print('conv7_2', conv7_2.shape)
    print('deconv2', deconv2.shape)
    print('concat2', concat2.shape)
    print('conv8_1', conv8_1.shape)
    print('conv8_2', conv8_2.shape)
    print('deconv1', deconv1.shape)
    print('concat1', concat1.shape)
    print('conv9_1', conv9_1.shape)
    print('conv9_2', conv9_2.shape)
    print('pred', pred.shape)

    return pred


# Proposed network
def mod2unet2D(images, training, nlabels):
    n_filt = 56
    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=n_filt, training=training)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=n_filt, training=training)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=n_filt*2, training=training)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=n_filt*2, training=training)

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=n_filt*4, training=training)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=n_filt*4, training=training)

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=n_filt*8, training=training)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=n_filt*8, training=training)

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=n_filt*16, training=training)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=n_filt*16, training=training)

    deconv4 = layers.deconv2D_layer_bn(conv5_2, 'deconv4', num_filters=4, training=training)
    concat4 = tf.concat([conv4_2, deconv4], axis=-1, name='concat4')

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=n_filt*8, training=training)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=n_filt*8, training=training)

    deconv3 = layers.deconv2D_layer_bn(conv6_2, 'deconv3', num_filters=4, training=training)
    concat3 = tf.concat([conv3_2, deconv3], axis=-1, name='concat3')

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=n_filt*4, training=training)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=n_filt*4, training=training)

    deconv2 = layers.deconv2D_layer_bn(conv7_2, 'deconv2', num_filters=4, training=training)
    concat2 = tf.concat([conv2_2, deconv2], axis=-1, name='concat2')

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=n_filt*2, training=training)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=n_filt*2, training=training)

    deep1 = layers.deconv2D_layer_bn(conv7_2, 'deep1', num_filters=n_filt*2, kernel_size=(1,1), training=training)
    add1 = tf.add(deep1, conv8_2)

    deconv1 = layers.deconv2D_layer_bn(conv8_2, 'deconv1', num_filters=4, training=training)
    concat1 = tf.concat([conv1_2, deconv1], axis=-1, name='concat1')

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=n_filt, training=training)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=n_filt, training=training)

    deep2 = layers.deconv2D_layer_bn(add1, 'deep2', num_filters=n_filt, kernel_size=(1,1), training=training)
    add2 = tf.add(deep2, conv9_2)

    pred = layers.conv2D_layer_bn(add2, 'pred', num_filters=nlabels, kernel_size=(1, 1), activation=tf.identity,
                                  training=training)

    print('conv1_1', conv1_1.shape)
    print('conv1_2', conv1_2.shape)
    print('pool1', pool1.shape)
    print('conv2_1', conv2_1.shape)
    print('conv2_2', conv2_2.shape)
    print('pool2', pool2.shape)
    print('conv3_1', conv3_1.shape)
    print('conv3_2', conv3_2.shape)
    print('pool3', pool3.shape)
    print('conv4_1', conv4_1.shape)
    print('conv4_2', conv4_2.shape)
    print('pool4', pool4.shape)
    print('conv5_1', conv5_1.shape)
    print('conv5_2', conv5_2.shape)
    print('deconv4', deconv4.shape)
    print('concat4', concat4.shape)
    print('conv6_1', conv6_1.shape)
    print('conv6_2', conv6_2.shape)
    print('deconv3', deconv3.shape)
    print('concat3', concat3.shape)
    print('conv7_1', conv7_1.shape)
    print('conv7_2', conv7_2.shape)
    print('deconv2', deconv2.shape)
    print('concat2', concat2.shape)
    print('conv8_1', conv8_1.shape)
    print('conv8_2', conv8_2.shape)
    print('deep1', deep1.shape)
    print('add1', add1.shape)
    print('deconv1', deconv1.shape)
    print('concat1', concat1.shape)
    print('conv9_1', conv9_1.shape)
    print('conv9_2', conv9_2.shape)
    print('deep2', deep2.shape)
    print('add2', add2.shape)
    print('pred', pred.shape)

    return pred


# Proposed network
def mod3unet2D(images, training, nlabels):
    n_filt=56
    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=n_filt, training=training)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=n_filt, training=training)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=n_filt*2, training=training)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=n_filt*2, training=training)

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=n_filt*4, training=training)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=n_filt*4, training=training)

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=n_filt*8, training=training)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=n_filt*8, training=training)

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=n_filt*16, training=training)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=n_filt*16, training=training)

    deconv4 = layers.deconv2D_layer_bn(conv5_2, 'deconv4', num_filters=4, training=training)

    cbam4 = layers.conv_block_att_module(conv4_2, 'cbam4', kernel_size=(7, 7), ratio=16)
    concat4 = tf.concat([cbam4, deconv4], axis=-1, name='concat4')

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=n_filt*8, training=training)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=n_filt*8, training=training)

    deconv3 = layers.deconv2D_layer_bn(conv6_2, 'deconv3', num_filters=4, training=training)

    cbam3 = layers.conv_block_att_module(conv3_2, 'cbam3', kernel_size=(7, 7), ratio=16)
    concat3 = tf.concat([cbam3, deconv3], axis=-1, name='concat3')

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=n_filt*4, training=training)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=n_filt*4, training=training)

    deconv2 = layers.deconv2D_layer_bn(conv7_2, 'deconv2', num_filters=4, training=training)

    cbam2 = layers.conv_block_att_module(conv2_2, 'cbam2', kernel_size=(7, 7), ratio=16)
    concat2 = tf.concat([cbam2, deconv2], axis=-1, name='concat2')

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=n_filt*2, training=training)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=n_filt*2, training=training)

    deep1 = layers.deconv2D_layer_bn(conv7_2, 'deep1', num_filters=n_filt*2, kernel_size=(1,1), training=training)
    add1 = tf.add(deep1, conv8_2)

    deconv1 = layers.deconv2D_layer_bn(conv8_2, 'deconv1', num_filters=4, training=training)
    cbam1 = layers.conv_block_att_module(conv1_2, 'cbam1', kernel_size=(7, 7), ratio=16)
    concat1 = tf.concat([cbam1, deconv1], axis=-1, name='concat1')

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=n_filt, training=training)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=n_filt, training=training)

    deep2 = layers.deconv2D_layer_bn(add1, 'deep2', num_filters=n_filt, kernel_size=(1,1), training=training)
    add2 = tf.add(deep2, conv9_2)

    pred = layers.conv2D_layer_bn(add2, 'pred', num_filters=nlabels, kernel_size=(1, 1), activation=tf.identity,
                                  training=training)

    print('conv1_1', conv1_1.shape)
    print('conv1_2', conv1_2.shape)
    print('pool1', pool1.shape)
    print('conv2_1', conv2_1.shape)
    print('conv2_2', conv2_2.shape)
    print('pool2', pool2.shape)
    print('conv3_1', conv3_1.shape)
    print('conv3_2', conv3_2.shape)
    print('pool3', pool3.shape)
    print('conv4_1', conv4_1.shape)
    print('conv4_2', conv4_2.shape)
    print('pool4', pool4.shape)
    print('conv5_1', conv5_1.shape)
    print('conv5_2', conv5_2.shape)
    print('deconv4', deconv4.shape)
    print('cbam4', cbam4.shape)
    print('concat4', concat4.shape)
    print('conv6_1', conv6_1.shape)
    print('conv6_2', conv6_2.shape)
    print('deconv3', deconv3.shape)
    print('cbam3', cbam3.shape)
    print('concat3', concat3.shape)
    print('conv7_1', conv7_1.shape)
    print('conv7_2', conv7_2.shape)
    print('deconv2', deconv2.shape)
    print('cbam2', cbam2.shape)
    print('concat2', concat2.shape)
    print('conv8_1', conv8_1.shape)
    print('conv8_2', conv8_2.shape)
    print('deep1', deep1.shape)
    print('add1', add1.shape)
    print('deconv1', deconv1.shape)
    print('cbam1', cbam1.shape)
    print('concat1', concat1.shape)
    print('conv9_1', conv9_1.shape)
    print('conv9_2', conv9_2.shape)
    print('deep2', deep2.shape)
    print('add2', add2.shape)
    print('pred', pred.shape)

    return pred


# Proposed network
def mod4unet2D(images, training, nlabels):
    n_filt=56
    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=n_filt, training=training)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=n_filt, training=training)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=n_filt*2, training=training)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=n_filt*2, training=training)

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=n_filt*4, training=training)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=n_filt*4, training=training)

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=n_filt*8, training=training)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=n_filt*8, training=training)

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=n_filt*16, training=training)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=n_filt*16, training=training)

    deconv4 = layers.deconv2D_layer_bn(conv5_2, 'deconv4', num_filters=4, training=training)

    att4 = layers.attention(conv4_2, 'att4', conv5_2)
    concat4 = tf.concat([att4, deconv4], axis=-1, name='concat4')

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=n_filt*8, training=training)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=n_filt*8, training=training)

    deconv3 = layers.deconv2D_layer_bn(conv6_2, 'deconv3', num_filters=4, training=training)

    att3 = layers.attention(conv3_2, 'att3', conv6_2)
    concat3 = tf.concat([att3, deconv3], axis=-1, name='concat3')

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=n_filt*4, training=training)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=n_filt*4, training=training)

    deconv2 = layers.deconv2D_layer_bn(conv7_2, 'deconv2', num_filters=4, training=training)

    att2 = layers.attention(conv2_2, 'att2', conv7_2)
    concat2 = tf.concat([att2, deconv2], axis=-1, name='concat2')

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=n_filt*2, training=training)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=n_filt*2, training=training)

    deep1 = layers.deconv2D_layer_bn(conv7_2, 'deep1', num_filters=n_filt*2, kernel_size=(1,1), training=training)
    add1 = tf.add(deep1, conv8_2)

    deconv1 = layers.deconv2D_layer_bn(conv8_2, 'deconv1', num_filters=4, training=training)

    att1 = layers.attention(conv1_2, 'att1', conv8_2)
    concat1 = tf.concat([att1, deconv1], axis=-1, name='concat1')

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=n_filt, training=training)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=n_filt, training=training)

    deep2 = layers.deconv2D_layer_bn(add1, 'deep2', num_filters=n_filt, kernel_size=(1,1), training=training)
    add2 = tf.add(deep2, conv9_2)

    pred = layers.conv2D_layer_bn(add2, 'pred', num_filters=nlabels, kernel_size=(1, 1), activation=tf.identity,
                                  training=training)

    print('conv1_1', conv1_1.shape)
    print('conv1_2', conv1_2.shape)
    print('pool1', pool1.shape)
    print('conv2_1', conv2_1.shape)
    print('conv2_2', conv2_2.shape)
    print('pool2', pool2.shape)
    print('conv3_1', conv3_1.shape)
    print('conv3_2', conv3_2.shape)
    print('pool3', pool3.shape)
    print('conv4_1', conv4_1.shape)
    print('conv4_2', conv4_2.shape)
    print('pool4', pool4.shape)
    print('conv5_1', conv5_1.shape)
    print('conv5_2', conv5_2.shape)
    print('deconv4', deconv4.shape)
    print('att4', att4.shape)
    print('concat4', concat4.shape)
    print('conv6_1', conv6_1.shape)
    print('conv6_2', conv6_2.shape)
    print('deconv3', deconv3.shape)
    print('att3', att3.shape)
    print('concat3', concat3.shape)
    print('conv7_1', conv7_1.shape)
    print('conv7_2', conv7_2.shape)
    print('deconv2', deconv2.shape)
    print('att2', att2.shape)
    print('concat2', concat2.shape)
    print('conv8_1', conv8_1.shape)
    print('conv8_2', conv8_2.shape)
    print('deep1', deep1.shape)
    print('add1', add1.shape)
    print('deconv1', deconv1.shape)
    print('att1', att1.shape)
    print('concat1', concat1.shape)
    print('conv9_1', conv9_1.shape)
    print('conv9_2', conv9_2.shape)
    print('deep2', deep2.shape)
    print('add2', add2.shape)
    print('pred', pred.shape)

    return pred
