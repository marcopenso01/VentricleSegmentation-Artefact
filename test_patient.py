import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# for GPU process:
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# for CPU process:
# os.environ["CUDA_VISIBLE_DEVICES"] =

import logging
import time

import h5py
import numpy as np
import tensorflow as tf

import configuration as config
import image_utils
import metrics
import model as model
import utils
from packaging import version
from tensorflow.python.client import device_lib

logging.basicConfig(
    level=logging.INFO  # allow DEBUG level messages to pass through the logger
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# print(device_lib.list_local_devices())   #
# nvidia-smi
# print(K.tensorflow_backend._get_available_gpus())

assert 'GPU' in str(device_lib.list_local_devices())

print('is_gpu_available: %s' % tf.test.is_gpu_available())  # True/False
# Or only check for gpu's with cuda support
print('gpu with cuda support: %s' % tf.test.is_gpu_available(cuda_only=True))
# tf.config.list_physical_devices('GPU') #The above function is deprecated in tensorflow > 2.1

log_dir = os.path.join(config.log_root, config.experiment_name)

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "this notebook requires Tensorflow 2.0 or above"


def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


def score_data(input_path, output_folder, model_path, config, do_postprocessing=False, dice=True):
    nx, ny = config.image_size[:2]
    batch_size = 1
    num_channels = config.nlabels
    gt_exists = config.gt_exists

    image_tensor_shape = [batch_size] + list(config.image_size) + [1]
    images_pl = tf.compat.v1.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    mask_pl, softmax_pl = model.predict(images_pl, config)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        if dice:
            nn = 'model_best_dice.ckpt'
            data_file_name = 'pred_on_dice.hdf5'
        else:
            nn = 'model_best_loss.ckpt'
            data_file_name = 'pred_on_loss.hdf5'

        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, nn)
        saver.restore(sess, checkpoint_path)
        init_iteration = int(checkpoint_path.split('/')[-1].split('-')[-1])
        total_time = 0
        total_volumes = 0

        if not os.path.exists(output_folder):
            makefolder(output_folder)
        data_file_path = os.path.join(output_folder, data_file_name)
        out_file = h5py.File(data_file_path, "w")

        RAW = []
        PRED = []
        PAZ = []
        PHS = []
        MASK = []
        CIR_MASK = []

        for paz in os.listdir(input_path):

            start_time = time.time()
            logging.info('------- Reading %s' % paz)
            data = h5py.File(os.path.join(input_path, paz, 'pre_proc', 'artefacts.hdf5'), 'r')

            n_file = len(data['img_raw'][()])

            for ii in range(n_file):

                img = data['img_raw'][ii].copy()
                RAW.append(img)
                PAZ.append(paz)
                PHS.append(data['phase'][ii])
                if gt_exists:
                    MASK.append(data['mask'][ii])
                    CIR_MASK.append(data['mask_cir'][ii])

                if config.standardize:
                    img = image_utils.standardize_image(img)
                if config.normalize:
                    img = image_utils.normalize_image(img)

                # GET PREDICTION
                feed_dict = {
                    images_pl: img,
                }

                mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict=feed_dict)
                prediction_cropped = np.squeeze(logits_out[0, ...])

                prediction = np.uint8(np.argmax(prediction_cropped, axis=-1))

                if do_postprocessing:
                    prediction = image_utils.keep_largest_connected_components(prediction)

                PRED.append(prediction)

            data.close()
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            total_volumes += 1
            logging.info('Evaluation of volume took %f secs.' % elapsed_time)

        n_file = len(PRED)
        dt = h5py.special_dtype(vlen=str)
        out_file.create_dataset('img_raw', [n_file] + [nx, ny], dtype=np.float32)
        out_file.create_dataset('pred', [n_file] + [nx, ny], dtype=np.uint8)
        out_file.create_dataset('paz', (n_file,), dtype=dt)
        out_file.create_dataset('phase', (n_file,), dtype=dt)
        if gt_exists:
            out_file.create_dataset('mask', [n_file] + [nx, ny], dtype=np.uint8)
            out_file.create_dataset('mask_cir', [n_file] + [nx, ny], dtype=np.uint8)

        for i in range(n_file):
            out_file['img_raw'][i, ...] = RAW[i]
            out_file['pred'][i, ...] = PRED[i]
            out_file['paz'][i, ...] = PAZ[i]
            out_file['phase'][i, ...] = PHS[i]
            if gt_exists:
                out_file['mask'][i, ...] = MASK[i]
                out_file['mask_cir'][i, ...] = CIR_MASK[i]

        # free memory
        RAW = []
        PRED = []
        PAZ = []
        PHS = []
        MASK = []
        CIR_MASK = []

        out_file.close()

        logging.info('Average time per volume: %f' % (total_time / total_volumes))


if __name__ == '__main__':
    log_root = config.log_root
    model_path = os.path.join(log_root, config.experiment_name)
    logging.info(model_path)

    logging.warning('EVALUATING ON TEST SET')
    input_path = config.test_data_root
    output_path = os.path.join(model_path, 'predictions')

    score_data(input_path,
               output_path,
               model_path,
               config=config,
               do_postprocessing=True,
               dice=True)

    if config.gt_exists:
        metrics.main(output_path)
