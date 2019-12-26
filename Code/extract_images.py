import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os
import skimage

import warnings
warnings.filterwarnings("ignore", message="Possible precision loss when converting from float64 to uint16")

from utils import load_image, contrast_stretch, NDVI, EVI, pansharpen

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

# %% Get to the external drive and check for its presence
path_to_drive = '/Volumes/SupMem/DSTL_data/'
if not os.path.isdir(path_to_drive): print("Cannot find the external drive!")

# load the img_id
wkt_df = pd.read_csv(path_to_drive+'train_wkt_v4.csv')

# get all the image id
img_id_list = list(wkt_df.ImageId.unique())

#%%
print(f'>>>> Convert Polygon to mask \n'+'-'*80)
for i, img_id in enumerate(img_id_list):
    print(f'\t|---- {i+1:02} : Processing image {img_id}')
    _, m, p = load_image(path_to_drive+'sixteen_band/', img_id)
    img_fused = pansharpen(m, p, order=3, W=1.5, stretch_perc=(1,99))
    ndvi = NDVI(img_fused[:,:,4], img_fused[:,:,6])
    ndwi = NDVI(img_fused[:,:,6], img_fused[:,:,2])
    evi = EVI(img_fused[:,:,4], img_fused[:,:,6], img_fused[:,:,1])
    img = np.concatenate([img_fused, np.expand_dims(ndvi,2), np.expand_dims(ndwi,2), np.expand_dims(evi,2)], axis=2)
    img = np.moveaxis(img, 2, 0)
    skimage.external.tifffile.imsave(path_to_drive+'processed_img/'+img_id+'.tiff', skimage.img_as_uint(img))

# %% -----------------------------------------------------------------------------------
# processing example
img_id = '6120_2_2'
_, m, p = load_image(path_to_drive+'sixteen_band/', img_id)
m_tmp = contrast_stretch(m)
p_tmp = np.squeeze(contrast_stretch(np.expand_dims(p,2)))
m_up = skimage.transform.resize(m_tmp, p_tmp.shape, order=3)
img_fused = pansharpen(m, p, order=3, W=1.5, stretch_perc=(1,99))

#%%
fig, axs = plt.subplots(1,3,figsize=(12,6))
axs[0].imshow(m_tmp[300:400,300:400,[6,4,2]])
axs[0].set_title('Multispectral Image', fontsize=12)
axs[1].imshow(m_up[1200:1600,1200:1600,[6,4,2]])
axs[1].set_title('Bicubic upsampling', fontsize=12)
axs[2].imshow(img_fused[1200:1600,1200:1600,[6,4,2]])
axs[2].set_title('Image fusion', fontsize=12)
for ax in axs: ax.set_axis_off()
fig.tight_layout()
fig.savefig('../Figures/pansharpening.png', dpi=200)
plt.show()

#%% Index example
img_id = '6100_2_2'
_, m, p = load_image(path_to_drive+'sixteen_band/', img_id)
img_fused = pansharpen(m, p, order=3, W=1.5, stretch_perc=(1,99))

#%%
ndvi = NDVI(img_fused[:,:,4], img_fused[:,:,6])
ndwi = NDVI(img_fused[:,:,6], img_fused[:,:,2])
evi = EVI(img_fused[:,:,4], img_fused[:,:,6], img_fused[:,:,1])
fig, axs = plt.subplots(1,4,figsize=(16,6))
axs[0].set_title('NDVI', fontsize=12)
axs[0].imshow(ndvi[:,:], cmap='PiYG', vmin=-1, vmax=1)
axs[1].set_title('NDWI', fontsize=12)
axs[1].imshow(ndwi[:,:], cmap='coolwarm_r', vmin=-1, vmax=1)
axs[2].set_title('EVI', fontsize=12)
axs[2].imshow(evi[:,:], cmap='PiYG', vmin=-1, vmax=1)
axs[3].set_title('False Color Composition', fontsize=12)
axs[3].imshow(img_fused[:,:,[4,2,1]])
for ax in axs: ax.set_axis_off()
fig.tight_layout()
fig.savefig('../Figures/indices.png', dpi=200)
plt.show()
