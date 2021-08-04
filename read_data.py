def imfill(img):
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)

def prepare_data(input_folder, output_file):
    
    hdf5_file = h5py.File(output_file, "w")
    
    flagENDO = False
    flagRV = False
    
    # 1: 'RV', 2: 'Myo', 3: 'LV'
    addrs = []
    MASK = []
    IMG_SEG = []  #img in uint8 con segmentazione
    IMG_RAW = []  #img in float senza segmentazione
    
    path_seg = os.path.join(input_folder, 'SEG')
    path_seg = os.path.join(path_seg, os.listdir(path_seg)[0])
    
    path_raw = os.path.join(input_folder, 'RAW')
    path_raw = os.path.join(path_raw, os.listdir(path_raw)[0])
    
    for i in range(len(os.listdir(path_seg))):
        dcmPath = os.path.join(path_seg, os.listdir(path_seg)[i])
        data_row_img = pydicom.dcmread(dcmPath)
        img = data_row_img.pixel_array
        
        green_pixels = cv2.inRange(img, (0, 225, 0), (0, 255, 0))
        
        if len(np.argwhere(green_pixels)) > 5:
            
            mask_epi = imfill(green_pixels)
        
            red_pixels = cv2.inRange(img, (250, 0, 0), (255, 0, 0))
            
            if len(np.argwhere(red_pixels)) > 5:
                
                flagENDO = True
                
                mask_endo = imfill(red_pixels)  #mask LV
        
                mask_myo = mask_epi - mask_endo  #mask Myo
                
                mask_endo[mask_endo>0]=3
                
                mask_myo[mask_myo>0]=2
        
            yellow_pixels = cv2.inRange(img, (225, 225, 0), (255, 255, 0))
            
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
            
            MASK.append(final_mask)
            addrs.append(dcmPath)
            IMG_SEG.append(img)
            
            dcmPath = os.path.join(path_raw, os.listdir(path_raw)[i])
            data_row_img = pydicom.dcmread(dcmPath)
            img = data_row_img.pixel_array
            IMG_RAW.append(img)
            
        flagENDO = False
        flagRV = False
    
    dt = h5py.special_dtype(vlen=str)
    hdf5_file.create_dataset('addrs', (len(addrs),), dtype=dt)
    hdf5_file.create_dataset('mask', [len(addrs)] + [512, 512], dtype=np.uint8)
    hdf5_file.create_dataset('img_seg', [len(addrs)] + [512, 512], dtype=np.uint8)
    hdf5_file.create_dataset('img_raw', [len(addrs)] + [512, 512], dtype=np.float32)
    
    for i in range(len(addrs)):
         hdf5_file['addrs'][i, ...] = addrs[i]
         hdf5_file['mask'][i, ...] = MASK[i]
         hdf5_file['img_seg'][i, ...] = IMG_SEG[i]
         hdf5_file['img_raw'][i, ...] = IMG_RAW[i]
    
    # After loop:
    hdf5_file.close()
    


def load_and_maybe_process_data(input_folder,
                                preprocessing_folder):
        
    '''
    This function is used to load and if necessary preprocesses the dataset
    
    :param input_folder: Folder where the raw data is located 
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    
    :return: Returns an h5py.File handle to the dataset
    '''  
    data_file_name = 'artefacts.hdf5'

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path):

        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path)

    else:

        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':

    # Paths settings
    input_folder = r'F:/PERFETTI LUISA'
    preprocessing_folder = os.path.join(input_folder, 'pre_proc')
    d=load_and_maybe_process_data(input_folder, preprocessing_folder)
