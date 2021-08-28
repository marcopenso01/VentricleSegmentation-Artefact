"""
Created on Fri Aug 27 13:44:33 2021

@author: Marco Penso
"""

def imfill(img):
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)

def crop_or_pad_slice_to_size(slice, nx, ny):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
    
    for i in range(len(stack)):
        
        img = stack[i]
            
        x, y = img.shape
        
        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2
    
        if x > nx and y > ny:
            slice_cropped = img[x_s:x_s + nx, y_s:y_s + ny]
        else:
            slice_cropped = np.zeros((nx, ny))
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]
        if len(stack)>1:
            RGB.append(slice_cropped)
    
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped


def generator_mask(img, green_pixels):
    
    flagENDO = False
    flagRV = False
    
    mask_epi = imfill(green_pixels)
        
    red_pixels = cv2.inRange(img, (110, 0, 0), (255, 100, 100))
    
    if len(np.argwhere(red_pixels)) > 5:
        
        flagENDO = True
        
        mask_endo = imfill(red_pixels)  #mask LV

        mask_myo = mask_epi - mask_endo  #mask Myo
        
        mask_endo[mask_endo>0]=3
        
        mask_myo[mask_myo>0]=2

    yellow_pixels = cv2.inRange(img, (110, 110, 0), (255, 255, 100))
    
    if len(np.argwhere(yellow_pixels)) > 5:
        
        flagRV = True

        temp = imfill(green_pixels + yellow_pixels)

        mask_RV = temp - mask_epi
        
        mask_RV[mask_RV>0]=1
    
    mask_epi[mask_epi>0]=2
                
    #binary mask 0-1
    if flagENDO:
        final_mask = mask_endo + mask_myo
    else:
        final_mask = mask_epi
    if flagRV:
        final_mask += mask_RV
    
    return final_mask
    

def prepare_data(input_folder, output_file, nx, ny):
    
    hdf5_file = h5py.File(output_file, "w")
    
    # 1: 'RV', 2: 'Myo', 3: 'LV'
addrs = []
MASK = []
IMG_SEG = []  #img in uint8 con segmentazione
IMG_RAW = []  #img in float senza segmentazione
IMG_CIR = []  #img in uint8 con segmentazione
MASK_CIR = []

path_seg = os.path.join(input_folder, 'SEG')
path_seg = os.path.join(path_seg, os.listdir(path_seg)[0])

path_raw = os.path.join(input_folder, 'RAW')
path_raw = os.path.join(path_raw, os.listdir(path_raw)[0])

path_cir = os.path.join(input_folder, 'CIRCLE')
path_cir = os.path.join(path_cir, os.listdir(path_cir)[0])

for i in range(len(os.listdir(path_seg))):
    dcmPath = os.path.join(path_seg, os.listdir(path_seg)[i])
    data_row_img = pydicom.dcmread(dcmPath)
    img = data_row_img.pixel_array
    img = crop_or_pad_slice_to_size(img, nx, ny)
    
    green_pixels = cv2.inRange(img, (0, 110, 0), (100, 255, 100))
    
    if len(np.argwhere(green_pixels)) > 5:
        
        final_mask = generator_mask(img, green_pixels)
        
        if final_mask.max() > 3:
            print('ERROR: max value of the mask %d is %d' % (i, final_mask.max()))
        MASK.append(final_mask)
        addrs.append(dcmPath)
        IMG_SEG.append(img)
        
        # save data raw
        dcmPath = os.path.join(path_raw, os.listdir(path_raw)[i])
        data_row_img = pydicom.dcmread(dcmPath)
        img = data_row_img.pixel_array
        img = crop_or_pad_slice_to_size(img, nx, ny)
        IMG_RAW.append(img)
        
        # circle
        dcmPath = os.path.join(path_cir, os.listdir(path_cir)[i])
        data_row_img = pydicom.dcmread(dcmPath)
        img = data_row_img.pixel_array
        img = crop_or_pad_slice_to_size(img, nx, ny)
        IMG_CIR.append(img)
        
        green_pixels = cv2.inRange(img, (0, 110, 0), (100, 255, 100))
        yellow_pixels = cv2.inRange(img, (110, 110, 0), (255, 255, 100))
    
        if len(np.argwhere(green_pixels)) > 5:
            cir_mask = generator_mask(img, green_pixels)
            MASK_CIR.append(cir_mask)
        elif len(np.argwhere(yellow_pixels)) > 5:
            cir_mask = imfill(yellow_pixels)
            cir_mask[cir_mask>0]=1
            MASK_CIR.append(cir_mask)
        else:
            MASK_CIR.append(np.zeros((nx,ny), dtype=np.uint8))
                
    
    dt = h5py.special_dtype(vlen=str)
    hdf5_file.create_dataset('paz', (len(addrs),), dtype=dt)
    hdf5_file.create_dataset('num_img', (len(addrs),), dtype=dt)
    hdf5_file.create_dataset('mask', [len(addrs)] + [nx, ny], dtype=np.uint8)
    hdf5_file.create_dataset('img_seg', [len(addrs)] + [nx, ny], dtype=np.uint8)
    hdf5_file.create_dataset('img_raw', [len(addrs)] + [nx, ny], dtype=np.float32)
    hdf5_file.create_dataset('img_cir', [len(addrs)] + [nx, ny], dtype=np.uint8)
    hdf5_file.create_dataset('mask_cir', [len(addrs)] + [nx, ny], dtype=np.uint8)

    
    for i in range(len(addrs)):
         hdf5_file['paz'][i, ...] = addrs[i].split("\\")[3]
         hdf5_file['num_img'][i, ...] = addrs[i].split("\\")[-1]
         hdf5_file['mask'][i, ...] = MASK[i]
         hdf5_file['img_seg'][i, ...] = IMG_SEG[i]
         hdf5_file['img_raw'][i, ...] = IMG_RAW[i]
         hdf5_file['img_cir'][i, ...] = IMG_CIR[i]
         hdf5_file['mask_cir'][i, ...] = MASK_CIR[i]
    
    # After loop:
    hdf5_file.close()
    


def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                nx,
                                ny):
        
    '''
    This function is used to load and if necessary preprocesses the dataset
    
    :param input_folder: Folder where the raw data is located 
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :parm nx, ny: crop size
    
    :return: Returns an h5py.File handle to the dataset
    '''  
    data_file_name = 'artefacts.hdf5'

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path):

        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path, nx, ny)

    else:

        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':

    # Paths settings
    input_folder = r'F:\ARTEFACTS\ARTEFATTI\paz1'
    preprocessing_folder = os.path.join(input_folder, 'pre_proc')
    nx = 250
    ny = 250
    d=load_and_maybe_process_data(input_folder, preprocessing_folder, nx, ny)
