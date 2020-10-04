import sys

import nibabel as nib
import numpy as np
import scipy.ndimage.morphology as morph
import skimage.segmentation as sg
import maxflow
from remove_backgr_mouse import grow_middle


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


def distance_metrics(img,mask):
    res_dist = np.zeros(img.shape)

    pass


def gc_method(img, in_work_area,init_mask,k):

    # ===============PREPROC_ get
    mask_bord = init_mask.copy()
    mask_bord = morph.binary_dilation(mask_bord,1)
    mask_bord = (((not init_mask) | (not mask_bord)) &(init_mask | mask_bord))

    graph = maxflow.Graph[float](img.shape[1]^3, img.shape[1]^3)

    length,width,height = img.shape[0],img.shape[1],img.shape[2]
    grid = graph.add_grid_nodes(img.shape)
    nodes = graph.add_nodes(length*width*height)



    F,B = np.zeros(shape=img.shape), np.zeros(shape=img.shape)
    # add grid
    graph.add_grid_tedges(grid,img, img.max() - img)
    res_MF = graph.maxflow()
    I_o = graph.get_grid_segments(grid)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if not in_work_area[i, j, k]:
                    B[i,j,k] = 1
                if init_mask[i, j, k]:
                    F[i,j,k] = 1
                pass


    pass

if __name__ == "__main__":
    im_file = nib.load(sys.argv[1])

    # im_seg = nib.load(sys.argv[2])
    res_path = sys.argv[2]
    img = im_file.get_fdata()

    init_res = grow_middle(img)

    init_mask = init_res[0]
    init_fore = init_res[1]
    gc_method(img,init_fore,init_mask,2.3)



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