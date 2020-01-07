import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.cuda as cuda

from sklearn.metrics import jaccard_score, accuracy_score, f1_score

import numpy as np
import pandas as pd
from ast import literal_eval

import os
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib
import glob
from prettytable import PrettyTable

from learning_classes import dataset, U_net
from utils import print_param_summary, append_scores

#%%-------------------------------------------------------------------------------------------------
# General declaration

path_to_drive = '/Volumes/SupMem/'
path_to_data = path_to_drive + 'DSTL_data/'
path_to_mask = path_to_drive + 'DSTL_data/masks/'
path_to_img = path_to_drive + 'DSTL_data/processed_img/'
path_to_output = '../Outputs/'
if not os.path.isdir(path_to_drive): print("Cannot find the external drive!")

# index of class in masks
class_position = {'building':0, 'misc':1, 'road':2, 'track':3, \
                  'tree':4, 'crop':5, 'water':6, 'vehicle':7}

#%%-------------------------------------------------------------------------------------------------
# Get the Dataset and Dataloader

# recover the class to train from passed argument or define it
class_type = 'water'
# if sys.argv[1]:
#     if sys.argv[1] in [i for i in range(8)]:
#         class_type = sys.argv[1]
#     else:
#         raise ValueError(f'Wrong Input class type. Should be one of {list(class_position.keys())}')

# dataset parameters
augment_data = True
crop_size = (144,144)
train_frac = 0.85

# the full data
df = pd.read_csv(path_to_data+'train_samples.csv', index_col=0, converters={'classes' : literal_eval})
df = df[pd.DataFrame(df.classes.tolist()).isin([class_type]).any(1)]
# the size for the split
train_size = int(df.shape[0]*train_frac)
val_size = df.shape[0]-train_size

# the train and validation datasets
full_dataset = dataset(df, path_to_img, path_to_mask, class_position[class_type], augment=augment_data, crop_size=crop_size)
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# parameters for the dataloader , adapt batch size to ensure at least 4 batches
train_dataloader_params = {'batch_size': min(64, int(train_set.__len__()/4)), 'shuffle': True, 'num_workers': 6}
val_dataloader_params = {'batch_size': min(64, int(val_set.__len__()/4)), 'shuffle': True, 'num_workers': 6}

# The data loader
train_dataloader = torch.utils.data.DataLoader(train_set, **train_dataloader_params)
val_dataloader = torch.utils.data.DataLoader(val_set, **val_dataloader_params)

#%%-------------------------------------------------------------------------------------------------
# Training Settings
# get GPU if available
if cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# initialize output dict
log_train = {}

# learning parameters
n_epoch = 200
nb_epochs_finished = 0
lr = 0.0001

# the loss
criterion = nn.CrossEntropyLoss()

# initialize de model
model = U_net(in_channel=11)
model = model.to(device)

# the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# model name
model_name = 'Unet_' + class_type
# save params to log
log_train['params'] = {'device':device, 'train_dataloader_params':train_dataloader_params, \
                       'val_dataloader_params':val_dataloader_params, \
                       'n_epoch':n_epoch, 'lr':lr, 'loss_function':criterion, \
                       'optimizer':optimizer, 'class':class_type, 'data_augmentation':augment_data, \
                       'train size':train_size, 'validation size':val_size, \
                       'N parameters Unet':sum(p.numel() for p in model.parameters())}
print_param_summary(**log_train['params'])

#%%-------------------------------------------------------------------------------------------------
# Training procedure
print('_'*80+'\nTraining\n'+'_'*80)
# load model from checkpoint if any
checkpoint_name = path_to_output+model_name+'_checkpoint.pth'
try:
    checkpoint = torch.load(checkpoint_name)
    nb_epochs_finished = checkpoint['nb_epochs_finished']
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    log_train = checkpoint['log_train']
    print(f'\n>>> Checkpoint loaded with {nb_epochs_finished} epochs finished.\n')
except FileNotFoundError:
    print('\n>>> Starting from scratch.\n')
except:
    print('Error when loading the checkpoint.')
    exit(1)

t = PrettyTable()
t.field_names = ['Epoch', 'Sum loss     ', 'Validation Jaccard   ', 'Validation F1     ']
print(t)

for epoch in range(nb_epochs_finished, n_epoch):
    sum_loss = 0.0
    jaccard_train = []
    jaccard_val = []
    f1_train = []
    f1_val = []

    for train_data, train_label in train_dataloader:
        # Transfer to GPU if available
        train_data, train_label = train_data.to(device), train_label.to(device).long()
        # forward pass
        output = model(train_data)
        # compute the loss
        loss = criterion(output, train_label) # Output B x C x H x W ; Target B x H x W
        sum_loss += loss.item()
        #reset gardient
        optimizer.zero_grad()
        #backward pass
        loss.backward()
        # Gradient step
        optimizer.step()
        # compute train scores
        with torch.set_grad_enabled(False):
            jaccard_train.append(jaccard_score(train_label.flatten(), output.argmax(dim=1).flatten()))
            f1_train.append(f1_score(train_label.flatten(), output.argmax(dim=1).flatten()))

    # compute validation scores
    with torch.set_grad_enabled(False):
        for val_data, val_label in val_dataloader:
            val_data, val_label = val_data.to(device), val_label.to(device).long()
            val_pred = model(val_data).argmax(dim=1)
            jaccard_val.append(jaccard_score(val_label.flatten(), val_pred.flatten()))
            f1_val.append(f1_score(val_label.flatten(), val_pred.flatten()))

    # append values to log
    append_scores(log_train, epoch=epoch+1, loss=sum_loss, \
                             jaccard_train=jaccard_train, jaccard_val=jaccard_val, \
                             f1_train=f1_train, f1_val=f1_val)
    # print epoch summary
    t.add_row([f'{epoch+1:03d}', f'{sum_loss:.2f}', f'{log_train["jaccard_val"]["mean"][-1]:.2%}', f'{log_train["f1_val"]["mean"][-1]:.2%}'])
    print('\n'.join(t.get_string().splitlines()[-2:]))

    # save the current model state as checkpoint
    checkpoint = {'nb_epochs_finished': epoch+1, \
                  'model_state': model.state_dict(), \
                  'optimizer_state': optimizer.state_dict(), \
                  'log_train':log_train}
    torch.save(checkpoint, checkpoint_name)

# Save the log of training
with open(path_to_output+model_name+'_log_train.pickle', 'wb') as handle:
    pickle.dump(log_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('\n>>> LOG saved on disk at'+path_to_output+model_name+'_log_train.pickle')

# Save trained model state dict
torch.save(model.state_dict(), path_to_output+model_name+'_trained.pt')
print('\n>>> Trained model saved on disk at'+path_to_output+model_name+'_trained.pt')

#%%-------------------------------------------------------------------------------------------------
####################################################################################################
####################################################################################################
####################################################################################################
# PlayGround

# %% load sample_df
df = pd.read_csv(path_to_data+'train_samples.csv', index_col=0, converters={'classes' : literal_eval})
# %% get only selected class
this_class = 'water'
df_c = df[pd.DataFrame(df.classes.tolist()).isin([this_class]).any(1)]
train_set = dataset(df_c, path_to_img, path_to_mask, class_position[this_class], augment=True, crop_size=(144,144))

fig, axs = plt.subplots(4,5,figsize=(20,15))
for i, ax in enumerate(axs.reshape(-1)):
    img, mask = train_set.__getitem__(np.random.randint(0, df_c.shape[0]))
    ax.imshow(np.moveaxis(np.array(img[[4,2,1], :, :]), 0, 2))
    m = np.ma.masked_where(mask == 0, mask)
    ax.imshow(m, cmap = matplotlib.colors.ListedColormap(['Orangered', 'white']), vmin=0, vmax=1, alpha=0.25)
    ax.set_axis_off()
fig.tight_layout()
plt.show()
mask
