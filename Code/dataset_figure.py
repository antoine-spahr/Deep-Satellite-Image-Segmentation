import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib

from learning_classes import dataset

path_to_drive = '/Volumes/SupMem/'
path_to_data = path_to_drive + 'DSTL_data/'
path_to_mask = path_to_drive + 'DSTL_data/masks/'
path_to_img = path_to_drive + 'DSTL_data/processed_img/'

# index of class in masks
class_position = {'building':0, 'misc':1, 'road':2, 'track':3, \
                  'tree':4, 'crop':5, 'water':6, 'vehicle':7}

# %% load sample_df
df = pd.read_csv(path_to_data+'train_samples.csv', index_col=0, converters={'classes' : literal_eval})

# %% plot
N_img = 6
fig, axs = plt.subplots(8,N_img,figsize=(N_img*3,8*3+2), gridspec_kw={'wspace':0.05, 'hspace':0.2})
fig.patch.set_alpha(0)

for i, (class_name, class_pos) in enumerate(class_position.items()):
    # creat dataset
    df_tmp = df[pd.DataFrame(df.classes.tolist()).isin([class_name]).any(1)]
    data_set = dataset(df_tmp, path_to_img, path_to_mask, class_position[class_name], augment=True, crop_size=(144,144))
    # draw group rectangles
    pos = axs[i,0].get_position()
    fig.patches.extend([plt.Rectangle((pos.x0-0.06,pos.y0-0.05*pos.height), pos.width*1.125*N_img , pos.height*1.1,
                                       facecolor='whitesmoke', ec='black', alpha=1, zorder=-1,
                                       transform=fig.transFigure, figure=fig)])
    fig.text(pos.x0-0.03, pos.y0+0.5*pos.height, class_name.title(), rotation=90, rotation_mode='anchor', \
             fontweight='bold', fontsize=14, ha='center', va='center')
    # plot image + mask
    for j in range(N_img):
        img, mask = data_set.__getitem__(np.random.randint(0, df_tmp.shape[0]))
        axs[i,j].imshow(np.moveaxis(np.array(img[[4,2,1], :, :]), 0, 2))
        m = np.ma.masked_where(mask == 0, mask)
        axs[i,j].imshow(m, cmap = matplotlib.colors.ListedColormap(['white', 'red']), vmin=0, vmax=1, alpha=0.3)
        axs[i,j].set_axis_off()
#fig.tight_layout()
fig.savefig('../Figures/sample_ex.png', dpi=150, bbox_inches='tight')
plt.show()
