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

import warnings
warnings.filterwarnings("ignore", message="Dataset has no geotransform set. The identity matrix may be returned")

# ------------------------ Extract Polygons ------------------------------------

def get_polygon_list(img_id, class_id, wkt_df):
    """

    ------------
    INPUT
        |---- img_id
        |---- class_id
        |---- wkt_df
    OUTPUT
        |---- polygon_list
    """
    all_polygon = wkt_df[wkt_df.ImageId == img_id]
    polygon = all_polygon[all_polygon.ClassType == class_id].MultipolygonWKT
    polygon_list = shapely.wkt.loads(polygon.values[0])
    return polygon_list

def get_scale_factor(img_id, grid_df, img_size):
    """

    """
    img_h, img_w = img_size
    xmax = grid_df.loc[grid_df['ImageId']==img_id, 'Xmax'].values[0]
    ymin = grid_df.loc[grid_df['ImageId']==img_id, 'Ymin'].values[0]
    scale = ((img_w-2)/xmax, (img_h-2)/ymin)
    return scale

def scale_polygon_list(polygon_list, scale):
    """

    """
    polygon_list_scaled = shapely.affinity.scale(polygon_list, \
                                                 xfact=scale[0], \
                                                 yfact=scale[1], \
                                                 origin = [0., 0., 0.])
    return polygon_list_scaled


# Plot polygon
def plot_polygons(ax, polygon_dict, color_dict, legend=True):
    """

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

    """
    layer_order = {1:'crop', 2:'water', 3:'road', 4:'track',\
                   5:'building', 6:'misc', 7:'vehicle', 8:'tree'}

    for i, _ in enumerate(range(len(layer_order)), start=1):
        class_name = layer_order[i]
        pos = order_dict[class_name]
        m = np.ma.masked_where(masks[:,:,pos-1] == 0, masks[:,:,pos-1])
        ax.imshow(m, cmap = matplotlib.colors.ListedColormap(['white', color_dict[class_name]]), \
                  vmin=0, vmax=1)


# Generate Polygon mask
def compute_polygon_mask(polygon_list, img_size):
    """

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
    """ 8xHxW

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
