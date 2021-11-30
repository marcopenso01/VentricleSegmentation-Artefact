from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2

# firt method
data = h5py.File('pred_on_dice.hdf5', "r")
img = data['img_raw'][0]
img_o = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
io.imshow(color.label2rgb(data['mask'][0],img_o,colors=[(255,0,0),(0,255,0),(255,255,0)],alpha=0.0008, bg_label=0, bg_color=None))
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
