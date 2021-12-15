import os
import numpy as np
import h5py

'file to match'
input_path = r'F:/ARTEFACTS/test'
out_path = r'F:/ARTEFACTS'

dt = h5py.special_dtype(vlen=str)

data_file_path = os.path.join(out_path, 'test.hdf5')
hdf5_file = h5py.File(data_file_path, "w")
c=1
for fold in os.listdir(input_path):
    print(fold)
    path = os.path.join(input_path, fold)
    data = h5py.File(os.path.join(path, 'pre_proc', 'artefacts.hdf5'), 'r')
    d1 = data['img_raw'][()]
    d2 = data['mask'][()]
    d3 = data['paz'][()]
    d4 = data['phase'][()]
    d5 = data['num_img'][()]
    d6 = data['img_seg'][()]
    d7 = data['img_cir'][()]
    d8 = data['mask_cir'][()]
    
    print("img_raw:", d1.shape, d1.dtype)
    print("mask:", d2.shape, d2.dtype)
    if c==1:
        img_raw = d1
        mask = d2
        paz = d3
        phase = d4
        num_img = d5
        img_seg = d6
        img_cir = d7
        mask_cir = d8
        c += 1
    else:
        img_raw = np.concatenate((img_raw, d1), axis=0)
        mask = np.concatenate((mask, d2), axis=0)
        paz = np.concatenate((paz, d3), axis=0)
        phase = np.concatenate((phase, d4), axis=0)
        num_img = np.concatenate((num_img, d5), axis=0)
        img_seg = np.concatenate((img_seg, d6), axis=0)
        img_cir = np.concatenate((img_cir, d7), axis=0)
        mask_cir = np.concatenate((mask_cir, d8), axis=0)
    print("img_raw after conc:", img_raw.shape)
    print("mask after conc:", mask.shape)
    data.close()

hdf5_file.create_dataset('img_raw', img_raw.shape, img_raw.dtype)
hdf5_file.create_dataset('mask', mask.shape, mask.dtype)
hdf5_file.create_dataset('paz', paz.shape, dtype=dt)
hdf5_file.create_dataset('phase', phase.shape, dtype=dt)
hdf5_file.create_dataset('num_img', num_img.shape, dtype=dt)
hdf5_file.create_dataset('img_seg', img_seg.shape, img_seg.dtype)
hdf5_file.create_dataset('img_cir', img_cir.shape, img_cir.dtype)
hdf5_file.create_dataset('mask_cir', mask_cir.shape, mask_cir.dtype)


hdf5_file['img_raw'][()] = img_raw
hdf5_file['mask'][()] = mask
hdf5_file['paz'][()] = paz
hdf5_file['phase'][()] = phase
hdf5_file['num_img'][()] = num_img
hdf5_file['img_seg'][()] = img_seg
hdf5_file['img_cir'][()] = img_cir
hdf5_file['mask_cir'][()] = mask_cir

hdf5_file.close()
