import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os
import shapely
import descartes
import skimage

from utils import get_polygon_dict, get_polygons_masks, plot_polygons, plot_masks

Class_dict = {'building':[1], 'misc':[2], 'road':[3], 'track':[4], \
              'tree':[5], 'crop':[6], 'water':[7,8], 'vehicle':[9,10]}

order_dict = {'building':1, 'misc':2, 'road':3, 'track':4, \
              'tree':5, 'crop':6, 'water':7, 'vehicle':8}

color_dict = {'building':'0.7', 'misc':'0.4', 'road':'#b35806', 'track':'#dfc27d', \
              'tree':'#1b7837', 'crop':'#a6dba0', 'water':'#74add1', 'vehicle':'#f46d43'}

zorder_dict = {'crop':1, 'water':2, 'road':3, 'track':4,\
               'building':5, 'misc':6, 'vehicle':7, 'tree':8}

# %% Get to the external drive and check for its presence
path_to_drive = '/Volumes/SupMem/DSTL_data/'
if not os.path.isdir(path_to_drive): print("Cannot find the external drive!")

# load the grid for conversion of polygon into image space
grid = pd.read_csv(path_to_drive+'grid_sizes.csv').rename(columns={'Unnamed: 0':'ImageId'})

# load the polygons
wkt_df = pd.read_csv(path_to_drive+'train_wkt_v4.csv')

# get all the image id
img_id_list = list(wkt_df.ImageId.unique())

# %%
masks = {}
print(f'>>>> Convert Polygon to mask \n'+'-'*80)
for i, img_id in enumerate(img_id_list):
    print(f'\t|---- {i+1:02} : Getting segmentation of image {img_id}')
    img_size = skimage.io.imread(path_to_drive+'sixteen_band/'+img_id+'_P.tif', plugin="tifffile").shape
    pdict = get_polygon_dict(img_id, Class_dict, img_size, wkt_df, grid)
    masks[img_id] = get_polygons_masks(pdict, order_dict, img_size, filename=path_to_drive+'masks/'+img_id+'_mask.tif')

# %% plot masks
fig, axs = plt.subplots(5,5,figsize=(20,20))
fig.patch.set_alpha(0)
for (id, mask), ax in zip(masks.items(), axs.reshape(-1)):
    plot_masks(ax, mask, order_dict, color_dict, zorder_dict, legend=False)
    ax.set_title(id, fontsize=14)
    ax.tick_params(axis='both', which='both',bottom=False, top=False,\
                   labelbottom=False, right=False, left=False, labelleft=False)

handles = [matplotlib.patches.Patch(facecolor=pcol) for pcol in color_dict.values()]
labels = list(color_dict.keys())
lgd = fig.legend(handles, labels, ncol=8, loc='lower center', fontsize=14, \
           bbox_to_anchor=(0.5, -0.03), bbox_transform=fig.transFigure)
fig.tight_layout()
fig.savefig('../Figures/Segmentations_labels.png', dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
