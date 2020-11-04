import sys

import nibabel as nib
import numpy as np
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morph
import skimage.segmentation as sg

import segment_mouse


def grow_middle(img,start_end, thresh, num_div):
    # center of mass
    center = measurements.center_of_mass(img)
    center = [round(x) for x in center]
    mask = np.zeros(img.shape)
    xm = center[0]
    dx = round(img.shape[0] / 20)
    if dx < 1:
        dx = 2
    ym = center[1]
    dy = round(img.shape[1] / 20)
    if dy < 1:
        dy = 2
    zm = center[2]
    dz1 = round(img.shape[2] / 20)
    if dz1 < 1:
        dz1 = 2
    mask[:, :, :] = 0

    mask[xm - dx:xm + dx, ym - dy:ym + dy, zm - dz1:zm + dz1] = 1
    # -----------------------------------------------------------------
    mask = sg.morphological_chan_vese(img, iterations=25, init_level_set=mask > 0, lambda2=2)

    #images = segment_mouse.cv_model(img, init_mask=mask, num_divisions=num_div, num_iterations=100, start_end=start_end)


    mask1= sg.morphological_chan_vese(img, iterations=100, init_level_set=mask > 0, lambda1=0.5,lambda2=4)
    mask2 = sg.morphological_chan_vese(img, iterations=100, init_level_set=mask > 0, lambda1=4,lambda2=0.5)

    return [mask1, mask2, center]

if __name__ == "__main__":
    im_file = nib.load(sys.argv[1])

    res_path = sys.argv[2]
    have_t = False
    thresh = 0
    out_file = sys.argv[3]
    ou_mask = sys.argv[4]
    try:
        if sys.argv[5] == "none":
            have_t = True
        else:
            thresh = float(sys.argv[5])
    except:
        pass
    # Measure center of mass of image and create a small mask for which is CV initialisation
    img = im_file.get_fdata()
    cent = measurements.center_of_mass(img)
    cent = [round(x) for x in cent]
    imares = np.zeros(img.shape)
    xm = cent[0]
    dx = round(img.shape[0] / 20)
    if dx < 1:
        dx = 2
    ym = cent[1]
    dy = round(img.shape[1] / 20)
    if dy < 1:
        dy = 2
    zm = cent[2]
    dz = round(img.shape[2] / 20)
    if dz < 1:
        dz = 2
    imares[:, :, :] = 0

    imares[xm - dx:xm + dx, ym - dy:ym + dy, zm - dz:zm + dz] = 1
    # -----------------------------------------------------------------
    imares = sg.morphological_chan_vese(img, iterations=25, init_level_set=imares > 0, lambda2=2)
    if ou_mask != "none":
        images = segment_mouse.cv_model(img, init_mask=imares, num_divisions=3, num_iterations=100, start_end=[0.5, 4])
        imares = segment_mouse.combine_image(images)

    nif = nib.Nifti1Image(imares, im_file.affine)
    imares = imares > 0
    imares2 = np.zeros(img.shape)
    if ~ have_t:
        for i in range(imares.shape[0]):
            for j in range(imares.shape[1]):
                for k in range(imares.shape[2]):
                    if (imares[i, j, k] > 0) | (img[i, j, k] > thresh):
                        imares2[i, j, k] = img[i, j, k]
                    else:
                        imares2[i, j, k] = 0
    else:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                        imares2[i, j, k] = img[i, j, k]

    imares = imares > 0
    imares = morph.binary_dilation(imares, iterations=5)


    nif = nib.Nifti1Image(imares.astype(np.int8), im_file.affine)
#save bfc_mask

    if ou_mask != "none":
        nib.save(nif, res_path + '/' + ou_mask)

    nif = nib.Nifti1Image(imares2, im_file.affine)
    #save file

    nib.save(nif, res_path + '/' + out_file)


