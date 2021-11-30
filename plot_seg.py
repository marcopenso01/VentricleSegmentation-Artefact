from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2

data = h5py.File('pred_on_dice.hdf5', "r")

# firt method
for i in range(len(data['img_raw'])):
    print(data['paz'][i], data['phase'][i])
    img_raw = cv2.normalize(src=data['img_raw'][i], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    mask = data['mask'][i].astype(np.uint8)
    pred = data['pred'][i].astype(np.uint8)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_axis_off()
    ax1.imshow(color.label2rgb(mask,img_raw,colors=[(255,0,0),(0,255,0),(255,255,0)],alpha=0.0008, bg_label=0, bg_color=None))
    ax2 = fig.add_subplot(122)
    ax2.set_axis_off()
    ax2.imshow(color.label2rgb(pred,img_raw,colors=[(255,0,0),(0,255,0),(255,255,0)],alpha=0.0008, bg_label=0, bg_color=None))
    ax1.title.set_text('groud truth')
    ax2.title.set_text('pred')
    plt.show()


# second method
img = data['img_raw'][0]
img_o = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
img_o = cv2.cvtColor(img_o, cv2.COLOR_GRAY2RGB)
color = [(255,0,0),(0,255,0),(255,255,0)]
for struc in [1,2,3]:
    mask = data['mask'][0].copy()
    if struc == 2:
      mask[mask==1]=0
    else:
      mask[mask!=struc]=0
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image=img_o, contours=contours, contourIdx=-1, color=color[struc-1], thickness=1, lineType=cv2.LINE_AA)
plt.imshow(img_o)
