import sys

import nibabel as nib
import numpy as np
import scipy.ndimage.morphology as morph
import skimage.segmentation as sg
import maxflow
from remove_backgr_mouse import grow_middle
import math

def histogram_remove(img):
    i2 = np.zeros(img.shape)



def combine_image(images):
    res = None
    for i in images:
        if res is None:
            res = i.astype(int)
        else:
            res = res + i.astype(int)
    res = res / len(images)
    return res


def distance_metrics(mask):
    res_dist = np.zeros(mask.shape)
    # (l,w,h) = header.get_zooms()


    tmp_0 = mask.copy()
    tmp_1 = mask.copy()
    cnt = 1
    while True:
        tmp_1 = morph.binary_erosion(tmp_0)
        tmp_0 = np.logical_xor(tmp_1,tmp_0)
        if not tmp_0.max():
            break
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[2]):
                    if tmp_0[i,j,k]:
                        res_dist[i,j,k] = cnt
        cnt = cnt + 1
        tmp_0 = tmp_1.copy()
    return res_dist



def gc_method(img,
              narrow_mask,
              init_mask,
              k_v):

    # ===============PREPROC_ get
    # get overlap window to not process the all dataset (maybe)
    print(1)
    x0, x1, y0, y1, z0, z1 = (np.argmax(init_mask, 0)).min(), (np.argmax(init_mask, 0)).max(), \
                             (np.argmax(init_mask, 1)).min(), (np.argmax(init_mask, 1)).max(), \
                             (np.argmax(init_mask, 2)).min(), (np.argmax(init_mask, 2)).max(),
    l,w,h = x1-x0,y1-y0,z1-z0
    #GET DISTANCE METRICS

    dist_marker = distance_metrics(narrow_mask)

    # Calculate FOREGROUND AND BACKGROUND SEED

    Fw, Bw = np.zeros(shape=img.shape), np.zeros(shape=img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if not narrow_mask[i, j, k]:
                    Bw[i,j,k] = 1
                if init_mask[i, j, k]:
                    Fw[i,j,k] = 1
                pass

    # calculate mean in narrow mask (in paper it's WM)
    WM_mean = (np.ma.masked_array(img,mask=~narrow_mask)).mean()
    # Calculate outside mean value (paper threshold)
    thr = (np.ma.masked_array(img, mask=init_mask)).mean()
    k_val = k_v/(WM_mean - thr)
    # --------------------- Calculate weights ----------------------
    # 3x3 neibroughoud

    #


    weights = np.zeros((l,w,h,
                        3,#we assing weight to point from left to right from
                        # bottom to top  (x,y,z,(x+1), ) weight between point and point
                        ))


    for i in range(l):
        for j in range(w):
            for k in range(h):
                    #             x + 1
                # euclid
                weights[i,j,k,0] = (max([dist_marker[i+x0,j+y0,k+z0],dist_marker[i+x0 + 1,j+y0,k+z0]]))**2
                if (weights[i,j,k,0] != 1) | (narrow_mask[i+x0,j+y0,k+z0]==False):# maybe need to add smth else

                    # set weight for 0weight
                    if weights[i,j,k,0] == 0:

                        weights[i,j,k,0] = 0.5
                    t_val = min([img[i+x0,j+y0,k+z0],img[i+x0 + 1,j+y0,k+z0]])
                    weights[i,j,k,0] =weights[i,j,k,0] *abs(math.exp(k_val*(t_val - thr) ) - 1)
    #             y + 1
                weights[i, j, k,1] = (max([dist_marker[i+x0,j+y0,k+z0], dist_marker[i+x0,j+y0 + 1,k+z0]])) ** 2
                if (weights[i, j, k, 1] != 1) | (narrow_mask[i + x0, j + y0, k + z0] == False):  # maybe need to add smth else

                    # set weight for 0weight
                    if weights[i, j, k, 1] == 0:
                        weights[i, j, k, 1] = 0.5
                    t_val = min([img[i + x0, j + y0, k + z0], img[i + x0 + 1, j + y0, k + z0]])
                    weights[i, j, k, 1] = weights[i, j, k, 1] * abs(math.exp(k_val * (t_val - thr)) - 1)
    #             z + 1
                weights[i, j, k,2] = (max([dist_marker[i+x0,j+y0,k+z0], dist_marker[i+x0,j+y0,k+z0+1]])) ** 2
                if (weights[i,j,k,2] != 1) | (narrow_mask[i+x0,j+y0,k+z0]==False):# maybe need to add smth else

                    # set weight for 0weight
                    if weights[i,j,k,2] == 0:

                        weights[i,j,k,2] = 0.5
                    t_val = min([img[i+x0,j+y0,k+z0],img[i+x0 + 1,j+y0,k+z0]])
                    weights[i,j,k,2] =weights[i,j,k,0] *abs(math.exp(k_val*(t_val - thr) ) - 1)

    # BUILD EDGES TO PIPELINE
    graph = maxflow.Graph[float](img.shape[1]**3, img.shape[1]**3)


    length,width,height = img.shape[0],img.shape[1],img.shape[2]
    grid = graph.add_grid_nodes(img.shape)
    graph.add_grid_tedges()
    nodes = graph.add_nodes(length*width*height)




    # add grid
    graph.add_grid_tedges(grid,img, img.max() - img)
    res_MF = graph.maxflow()
    I_o = graph.get_grid_segments(grid)



    pass

if __name__ == "__main__":
    im_file = nib.load(sys.argv[1])

    # im_seg = nib.load(sys.argv[2])
    res_path = sys.argv[2]
    img = im_file.get_fdata()

    init_res = grow_middle(img)

    init_mask = init_res[0] #0
    init_fore = init_res[1] #0.8
    gc_method(img,init_fore,init_mask, 2.3)



    # imares = sg.morphological_chan_vese(img, iterations=2, init_level_set=imares > 0)
    # imares = morph.binary_fill_holes(imares)
    # imares = morph.binary_erosion(imares)
    # imares = morph.binary_dilation(imares)
    # imares = imares.astype(np.int8)
    #
    #
    # nif = nib.Nifti1Image(imares, im_file.affine)
    # nib.save(nif, res_path + '/output.nii.gz')

    print(1)