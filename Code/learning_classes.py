import numpy as np
import skimage
import skimage.transform
import rasterio

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data

from utils import load_image_part

class dataset(data.Dataset):
    """
    Define a pytorch dataset for the DSTL data.
    """
    def __init__(self, sample_df, image_path, target_path, class_type, augment=True, crop_size=(144,144)):
        """
        Initialize a dataset for the DSTL data from the crop informations.
        ------------
        INPUT
            |---- sample_df (pandas.DataFrame) Dataframe with the samples informations
            |                each row is a sample with image_id, crop corrdinates
            |                and crops dimension.
            |---- image_path (str) the path to the folder of images
            |---- mask_path (str) the path to the folder of plot_masks
            |---- class_type (int) the channel dimension to use (aka the class
            |                 coordinate in the mask)
            |---- augment (bool) whether to perform data augmentation
            |---- crop_size (tuple) the random crop size to perform
        Output
            |---- NONE
        """
        self.sample_df = sample_df
        self.image_path = image_path
        self.target_path = target_path
        self.class_type = class_type
        self.augment = augment
        self.crop_size = crop_size

    def transform(self, image, mask):
        """
        Transform the passed image and mask and perform a data augmentation if
        self.augment is True (random crop + random horizontal and vertical flip
        + random 90° rotations)
        ------------
        INPUT
            |---- image (3D numpy.array) the image to transform
            |---- mask (2D numpy.array) the corresponding mask to transform
        Output
            |---- image (3D torch.Tensor) the transformed image as B x H x W
            |---- mask (2D torch.Tensor) the transformed associated mask for the selected class
        """
        # Random crop
        if self.augment:
            r, c = self.get_crop_param(image.shape[1:3], self.crop_size)
        else:
            r, c = int((image.shape[1]-self.crop_size[0])/2), int((image.shape[2]-self.crop_size[1])/2)
        image = image[:,r:r+self.crop_size[0],c:c+self.crop_size[1]]
        mask = mask[r:r+self.crop_size[0],c:c+self.crop_size[1]]

        if self.augment:
            # Random Vertical flip
            if np.random.random() > 0.5:
                image = image[:, ::-1, :]
                mask = mask[::-1, :]
            # Random Horizontal flip
            if np.random.random() > 0.5:
                image = image[:, :, ::-1]
                mask = mask[:, ::-1]
            # Random Rotate
            angle = 90*np.random.randint(0,4) # number between 0 and 3
            image = np.moveaxis(skimage.transform.rotate(np.moveaxis(image, 0, 2), angle, preserve_range=True), 2, 0)
            mask = skimage.transform.rotate(mask, angle, preserve_range=True)

        # Transform to Tensor
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask

    def __len__(self):
        """
        Return length of the dataset (the number of sample)
        ------------
        INPUT
            |---- NONE
        Output
            |---- (int) the number of samples
        """
        return self.sample_df.shape[0]

    def __getitem__(self, index):
        """
        Return one sample of the dataset associated with the index (row of the dataframe)
        ------------
        INPUT
            |---- index (int) the index of the sample to load
        Output
            |---- image (3D torch.Tensor) the image as B x H x W
            |---- mask (2D torch.Tensor) the associated mask for the selected class
        """
        sample = self.sample_df.iloc[index, :]
        xy = (sample.row, sample.col)
        hw = (sample.h, sample.w)
        id = sample.img_id
        image = load_image_part(xy, hw, self.image_path+id+'.tif')
        mask = load_image_part(xy, hw, self.target_path+id+'_mask.tif', as_float=False)[self.class_type, :, :]
        return self.transform(image, mask)

    def get_crop_param(self, img_dim, crop_size):
        """
        Return random crop parameters given the input size and the crop size
        ------------
        INPUT
            |---- img_dim (tuple) the input image dimension as (row, col)
            |---- crop_size (tuple) the crop dimension as (row, col)
        Output
            |---- r_crop (int) the row coordinate of the crop
            |---- c_crop (int) the column coordinate of the crop
        """
        r_crop = np.random.randint(0,img_dim[0]-crop_size[0])
        c_crop = np.random.randint(0,img_dim[1]-crop_size[1])
        return r_crop, c_crop

class BinaryDiceLoss(nn.Module):
    """
    Define a Binary DiceLoss.
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        """
        Constructor of the BinaryDiceLoss.
        ------------
        INPUT
            |---- smooth (int) smooth number for the diceloss
            |---- p (int) the power to use in the denominator
            |---- reduction (str) how the loss should be reduced (should be one
            |                     of : 'mean', 'sum', or 'none')
        Output
            |---- None
        """
        nn.Module.__init__(self)
        self.smooth = smooth
        self.reduction = reduction
        self.p = p

    def forward(self, input, target):
        """
        Constructor of the BinaryDiceLoss.
        ------------
        INPUT
            |---- input (torch.FloatTensor) the binary input with dimension B x 2 x H x W.
            |                          The positive class is defined by the as a
            |                          one on the tensor.
            |---- target (torch.FloatTensor) the binary target with dimension B x H x W.
            |                          The positive class is defined by the as a
            |                          one on the tensor.
        Output
            |---- loss (torch.FloatTensor) the Dice loss with dimension depending
            |                              on the reduction chosen.
        """
        # check input
        assert input.shape[0] == target.shape[0], 'Input and Target must have the same batch size.'
        assert input.dim() == 4, f'Input dimension {input.shape} does not match. Should be 4D : Batch x 2 x Height x Width'
        assert target.dim() == 3, f'Target dimension {input.shape} does not match. Should be 3D : Batch x Height x Width'
        # convert input and target to float
        input, target = input.float(), target.float()
        # softmax on input and keep only the class of 1
        input = F.softmax(input, dim=1)[:,1,:,:]
        # linearize input and target as vector
        input = input.contiguous().view(input.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        # compute numerator and denominator
        numerator = 2*(input*target).sum(dim=1) + self.smooth
        denominator = torch.sum(input.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        # compute the dice loss
        loss = 1 - numerator / denominator
        # return according to the reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception(f'Unexpected reduction {self.reduction}')

class ResBlock(nn.Module):
    """
    Define a Residual block for the U-net. (2 3x3 convolution + BatchNorm layer
    with SELU as activation function).
    """
    def __init__(self, in_channel, out_channel):
      """
      Constructor of the resblock.
      ------------
      INPUT
          |---- in_channel (int) number of input channel
          |---- out_channel (int) number of output channel
      Output
          |---- None
      """
      nn.Module.__init__(self)
      self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
      self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
      self.BN = nn.BatchNorm2d(out_channel)

    def forward(self, x):
      """
      Forward method of the block.
      ------------
      INPUT
          |---- x (torch.Tensor) the input of dimension (batch, in_channel, img_H, img_W)
      Output
          |---- x (torch.Tensor) the output of dimension (batch, out_channel, img_H, img_W)
      """
      x = F.selu(self.BN(self.conv1(x)))
      x = F.selu(self.BN(self.conv2(x)))
      return x

class U_net(nn.Module):
  """
  Definition of the U-net model. Convolution with 5 ResBlock and MaxPool layers
  Followd by a deconvolution with 5 ResBlock and ConvTranspose layer.
  """
  def __init__(self, in_channel):
    """
    Constructor of the Unet.
    ------------
    INPUT
        |---- in_channel (int) number of input channel of the model
    Output
        |---- None
    """
    nn.Module.__init__(self)
    # Down blocks
    self.RBD1 = ResBlock(in_channel,32)
    self.RBD2 = ResBlock(32,64)
    self.RBD3 = ResBlock(64,128)
    self.RBD4 = ResBlock(128,256)
    self.RBD5 = ResBlock(256,512)

    # Up blocks
    self.convT1 = nn.ConvTranspose2d(512, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
    self.RBU1 = ResBlock(128+256,256)
    self.convT2 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, stride=2, output_padding=1)
    self.RBU2 = ResBlock(256+128,128)
    self.convT3 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
    self.RBU3 = ResBlock(128+64,64)
    self.convT4 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)
    self.RBU4 = ResBlock(64+32,32)
    self.convFinal = nn.Conv2d(32,2, kernel_size=3, padding=1)

  def forward(self, x):
    """
    Forward method of the Unet.
    ------------
    INPUT
        |---- x (torch.Tensor) the input of dimension (batch, in_channel, img_H, img_W)
    Output
        |---- x (torch.Tensor) the output of dimension (batch, 2, img_H, img_W)
    """
    # dimension Batch x Channel x Width x Height
    # down
    r1 = self.RBD1.forward(x)
    x = F.max_pool2d(r1, kernel_size=2, stride=2)
    r2 = self.RBD2.forward(x)
    x = F.max_pool2d(r2, kernel_size=2, stride=2)
    r3 = self.RBD3.forward(x)
    x = F.max_pool2d(r3, kernel_size=2, stride=2)
    r4 = self.RBD4.forward(x)
    x = F.max_pool2d(r4, kernel_size=2, stride=2)
    x = self.RBD5.forward(x)
    # up
    #x = F.interpolate(x, scale_factor=2, mode='bilinear')
    x = F.selu(self.convT1(x))
    x = self.RBU1.forward(torch.cat((r4, x), dim=1))
    #x = F.interpolate(x, scale_factor=2, mode='bilinear')
    x = F.selu(self.convT2(x))
    x = self.RBU2.forward(torch.cat((r3, x), dim=1))
    #x = F.interpolate(x, scale_factor=2, mode='bilinear')
    x = F.selu(self.convT3(x))
    x = self.RBU3.forward(torch.cat((r2, x), dim=1))
    #x = F.interpolate(x, scale_factor=2, mode='bilinear')
    x = F.selu(self.convT4(x))
    x = self.RBU4.forward(torch.cat((r1, x), dim=1))
    x = F.selu(self.convFinal(x))
    return x


# Shallower
class U_net2(nn.Module):
  """  """
  def __init__(self, in_channel):
    """  """
    nn.Module.__init__(self)
    # Down blocks
    self.RBD1 = ResBlock(in_channel,32)
    self.RBD2 = ResBlock(32,64)
    self.RBD3 = ResBlock(64,128)
    self.RBD4 = ResBlock(128,256)

    # Up blocks
    self.convT2 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
    self.RBU2 = ResBlock(128+128,128)
    self.convT3 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
    self.RBU3 = ResBlock(128+64,64)
    self.convT4 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)
    self.RBU4 = ResBlock(64+32,32)
    self.convFinal = nn.Conv2d(32,2, kernel_size=3, padding=1)

  def forward(self, x):
    """  """
    # dimension Batch x Channel x Width x Height
    # down
    r1 = self.RBD1.forward(x)
    x = F.max_pool2d(r1, kernel_size=2, stride=2)
    r2 = self.RBD2.forward(x)
    x = F.max_pool2d(r2, kernel_size=2, stride=2)
    r3 = self.RBD3.forward(x)
    x = F.max_pool2d(r3, kernel_size=2, stride=2)
    x = self.RBD4.forward(x)
    # up
    #x = F.interpolate(x, scale_factor=2, mode='bilinear')
    x = F.selu(self.convT2(x))
    x = self.RBU2.forward(torch.cat((r3, x), dim=1))
    #x = F.interpolate(x, scale_factor=2, mode='bilinear')
    x = F.selu(self.convT3(x))
    x = self.RBU3.forward(torch.cat((r2, x), dim=1))
    #x = F.interpolate(x, scale_factor=2, mode='bilinear')
    x = F.selu(self.convT4(x))
    x = self.RBU4.forward(torch.cat((r1, x), dim=1))
    x = F.selu(self.convFinal(x))
    return x

# Shallower
class U_net3(nn.Module):
  """  """
  def __init__(self, in_channel):
    """  """
    nn.Module.__init__(self)
    # Down blocks
    self.RBD1 = ResBlock(in_channel,32)
    self.RBD2 = ResBlock(32,64)

    # Up blocks
    self.convT4 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)
    self.RBU4 = ResBlock(64+32,32)
    self.convFinal = nn.Conv2d(32,2, kernel_size=3, padding=1)

  def forward(self, x):
    """  """
    # dimension Batch x Channel x Width x Height
    # down
    r1 = self.RBD1.forward(x)
    x = F.max_pool2d(r1, kernel_size=2, stride=2)
    x = self.RBD2.forward(x)
    # up
    x = F.selu(self.convT4(x))
    x = self.RBU4.forward(torch.cat((r1, x), dim=1))
    x = F.selu(self.convFinal(x))
    return x
