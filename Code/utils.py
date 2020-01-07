import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os
import shapely
import shapely.wkt
import descartes
import skimage
import rasterio
from prettytable import PrettyTable
import torch

import warnings
warnings.filterwarnings("ignore", message="Dataset has no geotransform set. The identity matrix may be returned")

# ------------------------ Extract Polygons ------------------------------------

def get_polygon_list(img_id, class_id, wkt_df):
    """
    Load the polygons from the wkt dataframe for the specifed img and class.
    ------------
    INPUT
        |---- img_id (str) the image id
        |---- class_id (int) the class identifier
        |---- wkt_df (pandas.DataFrame) dataframe containing the wkt
    OUTPUT
        |---- polygon_list (shapely.MultiPolygon) the list of polygon
    """
    all_polygon = wkt_df[wkt_df.ImageId == img_id]
    polygon = all_polygon[all_polygon.ClassType == class_id].MultipolygonWKT
    polygon_list = shapely.wkt.loads(polygon.values[0])
    return polygon_list

def get_scale_factor(img_id, grid_df, img_size):
    """
    Load the polygons from the wkt dataframe for the specifed img and class.
    ------------
    INPUT
        |---- img_id (str) the image id
        |---- grid_df (pandas.DataFrame) dataframe containing the scalling info
        |---- img_size (tuple) dimension of the image
    OUTPUT
        |---- scale (tuple) the scalling factor for the given image
    """
    img_h, img_w = img_size
    xmax = grid_df.loc[grid_df['ImageId']==img_id, 'Xmax'].values[0]
    ymin = grid_df.loc[grid_df['ImageId']==img_id, 'Ymin'].values[0]
    scale = ((img_w-2)/xmax, (img_h-2)/ymin)
    return scale

def scale_polygon_list(polygon_list, scale):
    """
    Scale the polygon list.
    ------------
    INPUT
        |---- polygon_list (shapely.MultiPolygon) the list of polygon
        |---- scale (tuple) the scalling factor for the given image
    OUTPUT
        |---- polygon_list_scaled () scaled polygon_list
    """
    polygon_list_scaled = shapely.affinity.scale(polygon_list, \
                                                 xfact=scale[0], \
                                                 yfact=scale[1], \
                                                 origin = [0., 0., 0.])
    return polygon_list_scaled


# Plot polygon
def plot_polygons(ax, polygon_dict, color_dict, legend=True):
    """
    Plot the polygon on a matplotlib Axes.
    ------------
    INPUT
        |---- ax (matplotlib.Axes) the axes on which to plot
        |---- polygon_dict (dict) dictionnary of shapely.MultiPolygon for each classe by name
        |---- color_dict (dict) the color associated with each classes
        |---- legend (bool) whether to add a legend to the plot
    OUTPUT
        |---- mask (2D numpy.array) the mask
    """
    layer_order = {1:'crop', 2:'water', 3:'road', 4:'track',\
                   5:'building', 6:'misc', 7:'vehicle', 8:'tree'}
    for i, _ in enumerate(range(len(layer_order)), start=1):
        ax.add_patch(descartes.PolygonPatch(polygon_dict[layer_order[i]], \
                                            color=color_dict[layer_order[i]], \
                                            linewidth=0, label=layer_order[i]))
    if legend : ax.legend()

def plot_masks(ax, masks, order_dict, color_dict):
    """
    Plot the masks on a matplotlib Axes.
    ------------
    INPUT
        |---- ax (matplotlib.Axes) the axes on which to plot
        |---- masks (3D numpy.array) the masks to plot (C x H x W)
        |---- order_dict (dict) dictionnary of class order in the resulting mask
        |---- color_dict (dict) the color associated with each classes
    OUTPUT
        |---- mask (2D numpy.array) the mask
    """
    layer_order = {1:'crop', 2:'water', 3:'road', 4:'track',\
                   5:'building', 6:'misc', 7:'vehicle', 8:'tree'}

    for i, _ in enumerate(range(len(layer_order)), start=1):
        class_name = layer_order[i]
        pos = order_dict[class_name]
        m = np.ma.masked_where(masks[:,:,pos-1] == 0, masks[:,:,pos-1])
        ax.imshow(m, cmap = matplotlib.colors.ListedColormap(['white', color_dict[class_name]]), \
                  vmin=0, vmax=1)

def compute_polygon_mask(polygon_list, img_size):
    """
    Convert the shapely.MultiPolygon into a numpy mask.
    ------------
    INPUT
        |---- polygon_list (shapely.MultiPolygon) the list of polygon
        |---- img_size (tuple) dimension of the image
    OUTPUT
        |---- mask (2D numpy.array) the mask
    """
    # fill mask image
    mask = np.zeros(img_size, np.uint8)

    # add polygon to mask
    for poly in polygon_list:
        rc = np.array(list(poly.exterior.coords))
        rr, cc = skimage.draw.polygon(rc[:,1], rc[:,0])
        mask[rr, cc] = 1
    # remove holes
    for poly in polygon_list:
        for poly_int in poly.interiors:
            rc = np.array(list(poly_int.coords))
            rr, cc = skimage.draw.polygon(rc[:,1], rc[:,0])
            mask[rr, cc] = 0

    return mask

def get_polygons_masks(polygon_dict, order_dict, img_size, filename=None):
    """
    Convert the shapely polygons_dict into one numpy mask.
    ------------
    INPUT
        |---- polygon_dict (dict) dictionnary of shapely.MultiPolygon for each classe by name
        |---- order_dict (dict) dictionnary of class order in the resulting mask
        |---- img_size (tuple) dimension of the image
        |---- filename (str) path to save the mask (save only if given)
    OUTPUT
        |---- all_mask (3D numpy.array) the mask in dimension (C x H x W)
    """
    all_mask = np.zeros((img_size[0], img_size[1], len(order_dict)), np.uint8)
    for class_name, poly_list in polygon_dict.items():
        all_mask[:,:,order_dict[class_name]-1] = compute_polygon_mask(poly_list, img_size)

    if filename is not None:
        #skimage.external.tifffile.imsave(filename, np.moveaxis(all_mask, 2, 0))
        save_geotiff(filename, np.moveaxis(all_mask, 2, 0), dtype='uint8')

    return all_mask

def save_geotiff(filename, img, dtype='uint16'):
    """
    Save the image in GeoTiff.
    ------------
    INPUT
        |---- filename (str) the filename to save the image
        |---- img (3D numpy array) the image to save as B x H x W
    OUTPUT
        |---- None
    """
    with rasterio.open(filename, \
                        mode='w', \
                        driver='GTiff', \
                        width=img.shape[2], \
                        height=img.shape[1], \
                        count=img.shape[0], \
                        dtype=dtype) as dst:
        dst.write(img)

def get_polygon_dict(img_id, class_dict, img_size, wkt_df, grid_df):
    """
    Get the polygon list for the specified image classes from the raw information.
    ------------
    INPUT
        |---- img_id (str) the image id
        |---- class_dict (dictionnary) dictionnary specifying group of classes and name
        |---- img_size (tuple) dimension of the image
        |---- wkt_df (pandas.DataFrame) dataframe containing the wkt
        |---- grid_df (pandas.DataFrame) dataframe containing the scalling info
    OUTPUT
        |---- polygon_dict (dict) dictionnary of shapely.MultiPolygon for each classe by name
    """
    polygon_dict = {}
    for class_name, class_val in class_dict.items():
        # get first polygon list
        poly_list = get_polygon_list(img_id, class_val[0], wkt_df)
        scale = get_scale_factor(img_id, grid_df, img_size)
        poly_list = scale_polygon_list(poly_list, scale)
        # get polygon_list for next class
        if len(class_val) > 1:
            for next_class in class_val[1:]:
                next_poly_list = get_polygon_list(img_id, next_class, wkt_df)
                scale = get_scale_factor(img_id, grid_df, img_size)
                next_poly_list = scale_polygon_list(next_poly_list, scale)
                poly_list = poly_list.union(next_poly_list)

        polygon_dict[class_name] = poly_list
    return polygon_dict

# ------------------------ Preprocess Images -----------------------------------

def load_image(filepath, img_id):
    """
    Load the image associated with the id provided.
    ------------
    INPUT
        |---- filepath (str) the path to the sixteen bands images
        |---- img_id (str) the id of the image to load
    OUTPUT
        |---- img_A (3D numpy array) the SWIR bands
        |---- img_M (3D numpy array) the Multispectral bands
        |---- img_P (2D numpy array) the Panchromatic band
    """
    img_M = skimage.img_as_float(skimage.io.imread(filepath+img_id+'_M.tif', plugin="tifffile"))
    img_A = skimage.img_as_float(skimage.io.imread(filepath+img_id+'_A.tif', plugin="tifffile"))
    img_P = skimage.img_as_float(skimage.io.imread(filepath+img_id+'_P.tif', plugin="tifffile"))

    return np.moveaxis(img_A, 0, 2), np.moveaxis(img_M, 0, 2), img_P

def contrast_stretch(img, percentile=(0.5,99.5), out_range=(0,1)):
    """
    Stretch the image histogram for each channel independantly. The image histogram
    is streched such that the lower and upper percentile are saturated.
    ------------
    INPUT
        |---- img (3D numpy array) the image to stretch (H x W x B)
        |---- percentile (tuple) the two percentile value to saturate
        |---- out_range (tuple) the output range value
    OUTPUT
        |---- img_adj (3D numpy array) the streched image (H x W x B)
    """
    n_band = img.shape[2]
    q = [tuple(np.percentile(img[:,:,i], [0,99.5])) for i in range(n_band)]
    img_adj = np.stack([skimage.exposure.rescale_intensity(img[:,:,i], in_range=q[i], out_range=out_range) for i in range(n_band)], axis=2)
    return img_adj

def pansharpen(img_MS, img_Pan, order=2, W=1.5, stretch_perc=(0.5,99.5)):
    """
    Perform an image fusion of the multispectral image using the panchromatic one.
    The image is fisrt upsampled and interpolated. Then the panchromatic image is
    summed. And the image histogram is stretched.
    ------------
    INPUT
        |---- img_MS (3D numpy.array) the multispectral data as H x W x B
        |---- img_Pan (2D numpy.array) the Panchromatic data
        |---- order (int) the interpolation method to use (according to the scikit image method resize)
        |---- W (float) the weight of the summed panchromatic
    OUTPUT
        |---- img_fused (3D numpy.array) the pansharpened multispectral data as H x W x B
    """
    m_up = skimage.transform.resize(img_MS, img_Pan.shape, order=order)
    img_fused = np.multiply(m_up, W*np.expand_dims(img_Pan, axis=2))
    img_fused = contrast_stretch(img_fused, stretch_perc)
    return img_fused

def NDVI(R, NIR):
    """
    Compute the NDVI from the red and near infrared bands. Note that this
    function can be used to compute the NDWI by calling it with NDVI(NIR, G).
    ------------
    INPUT
        |---- R (2D numpy.array) the red band
        |---- NIR (2D numpy.array) the near infrared band
    OUTPUT
        |---- NDVI (2D numpy.array) the NDVI
    """
    return (NIR - R)/(NIR + R + 1e-9)

def EVI(R, NIR, B):
    """
    Compute the Enhenced Vegetation index from the red, near infrared bands and
    blue band.
    ------------
    INPUT
        |---- R (2D numpy.array) the red band
        |---- NIR (2D numpy.array) the near infrared band
        |---- B (2D numpy.array) the blue band
    OUTPUT
        |---- evi (2D numpy.array) the EVI
    """
    L, C1, C2 = 1.0, 6.0, 7.5
    evi = (NIR - R) / (NIR + C1 * R - C2 * B + L)
    evi = evi.clip(max=np.percentile(evi, 99), min=np.percentile(evi, 1))
    evi = evi.clip(max=1, min=-1) # clip if too big
    return evi

# ------------------------ Extract images --------------------------------------

def get_crops_grid(img_h, img_w, crop_size, overlap=None):
    """
    Get a list of crop coordinates for the image in a grid fashion starting in
    the upper left corner. There might be some un-considered pixels on the
    right and bottom.
    ------------
    INPUT
        |---- img_h (int) the image height
        |---- img_w (int) the image width
        |---- crop_size (tuple) the height and width of the crop
        |---- overlap (tuple) the total amount overlapped between crop in the grid
    OUTPUT
        |---- crops (list of tuple) list of crop upper left crops coordinates
    """
    # compute the overlap if No given
    if overlap is None:
        nx = np.ceil(img_h / crop_size[0])
        ny = np.ceil(img_w / crop_size[1])
        excess_y = ny*crop_size[0] - img_h
        excess_x = nx*crop_size[1] - img_w
        overlap = (np.ceil(excess_y / (ny-1)), np.ceil(excess_x / (nx-1)))

    crops = [] # (row, col)
    for i in np.arange(0,img_h+crop_size[0],crop_size[0]-overlap[0])[:]:
        for j in np.arange(0,img_w+crop_size[1],crop_size[1]-overlap[1])[:]:
            if i+crop_size[0] <= img_h and j+crop_size[1] <= img_w:
                crops.append((i, j))
    return crops

def load_image_part(xy, hw, filename, as_float=True):
    """
    load an image subpart specified by the crop coordinates and dimensions.
    ------------
    INPUT
        |---- xy (tuple) crop coordinsates  as (row, col)
        |---- hw (tuple) crop dimensions as (h, w)
        |---- filename (str) the filename
        |---- as_float (bool) whether to convert the image in float
    OUTPUT
        |---- img (3D numpy array) the image subpart
    """
    with rasterio.open(filename, mode='r') as src:
        img = src.read(window=rasterio.windows.Window(xy[1], xy[0], hw[1], hw[0]))
    if as_float: img = skimage.img_as_float(img)
    return img

def get_represented_classes(filename_mask, order_dict, crop_coord, crop_size):
    """
    find which classes are represented on the crop specified.
    ------------
    INPUT
        |---- filename (str) the filename
        |---- order_dict (dict) a dictionnary specifying the class name associated
        |                       with which dimension of the masks {dim+1:'name'}
        |---- crop_coord (tuple) crop coordinsates  as (row, col)
        |---- crop_size (tuple) crop dimensions as (h, w)
    OUTPUT
        |---- classes (list) list of present class name
    """
    mask = load_image_part(crop_coord, crop_size, filename_mask, as_float=False)
    classes = [order_dict[cl+1] for cl in np.unique(mask.nonzero()[0])]
    return classes

def get_samples(img_id_list, img_path, mask_path, crop_size, overlap, order_dict, cl_offset, cl_size, as_fraction=False, verbose=False):
    """
    Produce a pandas datafarme containing all the crop informations.
    ------------
    INPUT
        |---- img_id_list (list) the list of ids of image to processed
        |---- img_path (str) the folder path to the images
        |---- mask_path (str) the folder path to the masks
        |---- crop_size (tuple) crop dimensions as (h, w)
        |---- overlap (tuple) the total amount overlapped between crop in the grid
        |---- order_dict (dict) a dictionnary specifying the class name associated
        |                       with which dimension of the masks {dim+1:'name'}
        |---- cl_offset (tuple) the offset from crop to check class presence (row, col)
        |---- cl_size (tuple) the size of the patch where class presence is checked (h, w)
        |---- as_fraction (bool) whether to specify crop_size as fraction. if
        |                        True, the crop_size value represent fraction and
        |                        should be between 0 and 1
        |---- verbose (bool) whether to display processing
    OUTPUT
        |---- sample_df (pandas dataframe) informations for all samples
    """
    # DataFrame (img_id, x, y, h, w classes)
    if verbose :
        print(f'>>>> Extract samples from images \n'+'-'*80)
        summary = {'building':0, 'misc':0, 'road':0, 'track':0, \
                   'tree':0, 'crop':0, 'water':0, 'vehicle':0}
    # storing variables
    ids, row, col, H, W, cl_list = [], [], [], [], [], []
    for i, id in enumerate(img_id_list):
        if verbose:
            print(f'\t|---- {i+1:02n} : cropping image {id}')
            summary2 = {'building':0, 'misc':0, 'road':0, 'track':0, \
                        'tree':0, 'crop':0, 'water':0, 'vehicle':0}
        # get height width
        with rasterio.open(img_path+id+'.tif', mode='r') as src:
            img_h, img_w = src.height, src.width
        # define crop size from fraction if requested
        if as_fraction:
            crop_size = np.floor(img_h*crop_size[0]), np.floor(img_w*crop_size[1])
        # get the grid crops
        crops = get_crops_grid(img_h, img_w, crop_size, overlap)
        # fill lists
        for crop in crops:
            ids.append(id)
            row.append(crop[0])
            col.append(crop[1])
            H.append(crop_size[0])
            W.append(crop_size[1])
            classes = get_represented_classes(mask_path+id+'_mask.tif', \
                                                   order_dict, \
                                                   (crop[0]+cl_offset[0], crop[1]+cl_offset[1]), \
                                                   cl_size)
            # count classes
            if verbose:
                for cl in classes:
                    summary[cl] += 1
                    summary2[cl] += 1
            cl_list.append(classes)
        #display count
        if verbose:
            for cl, count in summary2.items():
                print(f'\t\t|---- {cl} : {count}')
    # display total count
    if verbose:
        print('-'*80+'\n>>>> Total \n')
        for cl, count in summary.items():
            print(f'\t|---- {cl} : {count}')
    # build dataframe
    sample_df = pd.DataFrame({'img_id':ids, \
                              'row':row, 'col':col, \
                              'h':H, 'w':W, \
                              'classes':cl_list})
    return sample_df

# ------------------------ Training functions  ---------------------------------

def print_param_summary(**params):
    """
    Print the dictionnary passed as a table.
    ------------
    INPUT
        |---- params (keyword arguments) value to display in PrettyTable
    OUTPUT
        |---- None
    """
    print(f'\n>>> Training parameters summary')
    tparam = PrettyTable(['Parameter','Value'])
    tparam.hrules = 1
    tparam.align = 'l'
    for key, value in params.items():
        tparam.add_row([key, value])
    print(tparam)

def stat_from_list(list):
    """
    Compute the mean and standard deviation of the list.
    ------------
    INPUT
        |---- list (list) list of value
    OUTPUT
        |---- mean (float) the mean of the list values
        |---- std (float) the standard deviation of the list of values
    """
    list = torch.Tensor(list)
    return list.mean().item(), list.std().item()

def append_scores(dest_dict, **keys):
    """
    Add the kwargs to the passed dictionnary. Each entry (key) is then a list
    of value. The value passe is append to such list. If a key is not present
    in the disctionnary, it is added. If a value is a list, the mean and std
    are append to the dictionnary.
    ------------
    INPUT
        |---- dest_dict (dictionnary) where the values are append (modified inplace)
        |---- keys (keyword arguments) value to append
    OUTPUT
        |---- None
    """
    for name, val in keys.items():
        if type(val) is list:
            m, s = stat_from_list(val)
            if name in dest_dict.keys():
                dest_dict[name]['mean'].append(m)
                dest_dict[name]['std'].append(s)
            else:
                dest_dict[name] = {}
                dest_dict[name]['mean'] = [m]
                dest_dict[name]['std'] = [s]
        else:
            if name in dest_dict.keys():
                dest_dict[name].append(val)
            else:
                dest_dict[name] = [val]
