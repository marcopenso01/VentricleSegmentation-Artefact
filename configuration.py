import model_structure
import tensorflow as tf
import os
import socket
import logging

experiment_name = 'test1'

# Model settings Unet2D
weight_init = 'he_normal'    # xavier_uniform/ xavier_normal/ he_normal /he_uniform /caffe_uniform/ simple/ bilinear
#model_handle = model_structure.unet2D_same_mod
#model_handle = model_structure.Dunet2D_same_mod


# Data settings
data_mode = '2D' 
image_size = (192, 192)   #(212,212)
target_resolution = (1, 1)
pixel_size = (1,1) 
nlabels = 4
split_test_train = True   #divide patients in train and test. If true define split
split = 5                 #  2: 50% train and 50% validation,    5: 80% training, 20% validation
train_on_all_data = False 
gt_exists = True    #True if it exists the ground_trth images, otherwise False.
z_dim = 8

# Training settings
batch_size = 4      #4 
learning_rate = 0.001   #unet: 0.01    enet: 0.0005
optimizer_handle = tf.compat.v1.train.AdamOptimizer     #(beta1 = 0.9, beta2 = 0.999, epsilon=1e-08)
schedule_lr = False    #decrease 10 times the LR when loss gradient lower than threshold
weight_decay = 0  # enet:2e-4    #unet: 0.00000
momentum = None
# loss can be 'weighted_crossentropy'/'crossentropy'/'dice'/'dice_onlyfg'/
# 'crossentropy_and_dice (alfa,beta)'/'tversky'/'focal_tversky'
loss_type = 'crossentropy_and_dice'
alfa = 1     #1      
beta = 0.4   #1      
augment_batch = True

# Augmentation settings
do_rotation_range = True   #random rotation in range "rg" (min,max)
rg = (-20,20)     
gamma = True
prob = 1                    #Probability [0.0/1.0] (0 no augmentation, 1 always)

# Paths settings (need to mount MyDrive before)
data_root = 'F:\ARTEFACTS'      
test_data_root = 'F:\ARTEFACTS'
project_root = 'F:\ARTEFACTS'                       
log_root = os.path.join(project_root, 'artefact_logdir')
weights_root = os.path.join(log_root, experiment_name)

# Pre-process settings
standardize = False
normalize = True

# Rarely changed settings
max_epochs = 1000

train_eval_frequency = 200
val_eval_frequency = 150 
