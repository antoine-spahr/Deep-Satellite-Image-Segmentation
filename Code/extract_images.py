import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os
import skimage

from utils import load_image, contrast_stretch

# %% Get to the external drive and check for its presence
path_to_drive = '/Volumes/SupMem/DSTL_data/'
if not os.path.isdir(path_to_drive): print("Cannot find the external drive!")

# load the img_id
wkt_df = pd.read_csv(path_to_drive+'train_wkt_v4.csv')

# get all the image id
img_id_list = list(wkt_df.ImageId.unique())

""" BANDS
|-- A
    |--- 0 : SWIR-1 1195 - 1225 n
    |--- 1 : SWIR-2 1550 - 1590 n
    |--- 2 : SWIR-3 1640 - 1680 n
    |--- 3 : SWIR-4 1710 - 1750 n
    |--- 4 : SWIR-5 2145 - 2185 nm
    |--- 5 : SWIR-6 2185 - 2225 nm
    |--- 6 : SWIR-7 2235 - 2285 nm
    |--- 7 : SWIR-8 2295 - 2365 nm
|-- M
    |--- 0 : Coastal Blue 400 - 450 nm
    |--- 1 : Blue 450 - 510 nm
    |--- 2 : Green 510 - 580 nm
    |--- 3 : Yellow 585 - 625 nm
    |--- 4 : Red 630 - 690 nm
    |--- 5 : Red-edges 705 - 745 nm
    |--- 6 : NIR-1 770 - 895 nm
    |--- 7 : NIR-2 860 - 1040 nm
|-- P
    |--- 0 : 450 - 800 nm
"""

# %%
img_id = '6120_2_2'

a, m, p = load_image(path_to_drive+'sixteen_band/', img_id)
m = contrast_stretch(m)
a.shape
m.shape
p.shape

#%%
fig, ax = plt.subplots(1,1,figsize=(9,9))
ax.imshow(np.moveaxis(m[[4,2,1],:,:], 0, 2))
plt.show()

#%%

def load_image(filepath, img_id):
    """

    ------------
    INPUT
        |---- filepath
        |---- img_id
    OUTPUT
        |---- polygon_list
    """
    img_M = skimage.img_as_float(skimage.io.imread(filepath+img_id+'_M.tif', plugin="tifffile"))
    img_A = skimage.img_as_float(skimage.io.imread(filepath+img_id+'_A.tif', plugin="tifffile"))
    img_P = skimage.img_as_float(skimage.io.imread(filepath+img_id+'_P.tif', plugin="tifffile"))

    return img_A, img_M, img_P

def contrast_stretch(img, percentile=(0.5,99.5), out_range=(0,1)):
    """

    ------------
    INPUT
        |---- filepath
        |---- img_id
    OUTPUT
        |---- polygon_list
    """
    n_band = img.shape[0]
    q = [tuple(np.percentile(img[i,:,:], [0,99.5])) for i in range(n_band)]
    img = np.stack([skimage.exposure.rescale_intensity(img[i,:,:], in_range=q[i], out_range=out_range) for i in range(n_band)], axis=0)
    return img
