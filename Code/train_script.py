import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.cuda as cuda

from sklearn.metrics import jaccard_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from ast import literal_eval

import os
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib

from learning_classes import dataset, U_net, U_net2
from utils import print_param_summary, append_scores, load_sample_df

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
class_type = 'building'
# if sys.argv[1]:
#     if sys.argv[1] in list(class_position.keys()):
#         class_type = sys.argv[1]
#     else:
#         raise ValueError(f'Wrong Input class type. Should be one of {list(class_position.keys())}')

# dataset parameters
augment_data = True
crop_size = (144,144)
train_frac = 0.85
non_class_fraction = 0.15

# the full data
df = load_sample_df(path_to_data+'train_samples.csv', class_type=class_type, others_frac=non_class_fraction, seed=1)
# train validation split
train_df, val_df = train_test_split(df, train_size=train_frac, random_state=1)

# the train and validation datasets
train_set = dataset(train_df, path_to_img, path_to_mask, class_position[class_type], augment=augment_data, crop_size=crop_size)
val_set = dataset(val_df, path_to_img, path_to_mask, class_position[class_type], augment=augment_data, crop_size=crop_size)

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
model = U_net2(in_channel=11)
model = model.to(device)

# the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# model name
model_name = 'Unet_' + class_type
# save params to log
log_train['params'] = {'device':device, 'train_dataloader_params':train_dataloader_params, \
                       'val_dataloader_params':val_dataloader_params, \
                       'n_epoch':n_epoch, 'lr':lr, 'loss_function':str(criterion), \
                       'optimizer':str(optimizer), 'class':class_type, 'data_augmentation':augment_data, \
                       'train size':train_df.shape[0], 'validation size':val_df.shape[0], \
                       'non_class_fraction':non_class_fraction, \
                       'N parameters Unet':sum(p.numel() for p in model.parameters())}
# save split indices
log_train['split_indices'] = {'train':train_set.sample_df.index.tolist(), 'val':val_set.sample_df.index.tolist()}
# print parameter summary
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

# initial print
pads = [8,14,25,25]
print('-'*(sum(pads)+1))
print('| Epoch'.ljust(pads[0]) + '| Sum loss'.ljust(pads[1]) + '| Validation Jaccard'.ljust(pads[2]) + '| Validation F1'.ljust(pads[3])+'|')
print('-'*(sum(pads)+1))

# train
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
            jaccard_train.append(jaccard_score(train_label.flatten().cpu(), output.argmax(dim=1).flatten().cpu()))
            f1_train.append(f1_score(train_label.flatten().cpu(), output.argmax(dim=1).flatten().cpu()))

    # compute validation scores
    with torch.set_grad_enabled(False):
        for val_data, val_label in val_dataloader:
            val_data, val_label = val_data.to(device), val_label.to(device).long()
            val_pred = model(val_data).argmax(dim=1)
            jaccard_val.append(jaccard_score(val_label.flatten().cpu(), val_pred.flatten().cpu()))
            f1_val.append(f1_score(val_label.flatten().cpu(), val_pred.flatten().cpu()))

    # append values to log
    append_scores(log_train, epoch=epoch+1, loss=sum_loss, \
                             jaccard_train=jaccard_train, jaccard_val=jaccard_val, \
                             f1_train=f1_train, f1_val=f1_val)
    # print epoch summary
    print(f'| {epoch+1:03d}'.ljust(pads[0]) + \
          f'| {sum_loss:.2f}'.ljust(pads[1]) + \
          f'| {log_train["jaccard_val"]["mean"][-1]:.2%}'.ljust(pads[2]) + \
          f'| {log_train["f1_val"]["mean"][-1]:.2%}'.ljust(pads[3])+'|')
    print('-'*(sum(pads)+1))

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
