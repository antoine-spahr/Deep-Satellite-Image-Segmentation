import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os
import shapely
import shapely.wkt
import descartes
import skimage

# ------------------------ Extract Polygons ------------------------------------

# Load polygon_list
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

# Scale polygons
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
        skimage.external.tifffile.imsave(filename, np.moveaxis(all_mask, 2, 0))

    return all_mask


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

# upsample image

# Image fusion

# indexes (NDVI, SAVI, ...)

# ------------------------ Extract images --------------------------------------
