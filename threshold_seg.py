import numpy
import skimage.segmentation.morphsnakes as sg
import scipy.ndimage.morphology as morph
import skimage.measure as measure
import nibabel as nib
import sys
import numpy as np
import scipy.ndimage.measurements as measurements
import Mouse_C

def cv_model(image,init_mask, num_divisions,num_iterations,start_end):

    dt = (start_end[1] - start_end[0])/num_divisions
    res = []
    for i in range(num_divisions):
        mares = Mouse_C.morph_cv(image, iterations=num_iterations, init_level_set=init_mask > 0,lambda1=start_end[0] + i*dt,lambda2=start_end[1]- i*dt)
        res.append(mares)

    return res

def combine_image(images):
    res = None
    for i in images:
        if res is None:
            res = i.astype(int)
        else:
            res = res + i.astype(int)
    res = res / len(images)
    return res
if __name__ == "__main__":
    sub_dir = sys.argv[1]

    treshold = float(sys.argv[2])




    nif = nib.load(sub_dir + '/combined.nii.gz')
    fd = nif.get_fdata()
    aff = nif.affine
    imares = fd > treshold



    imares = morph.binary_fill_holes(imares)
    imares = morph.binary_erosion(imares)
    imares = morph.binary_dilation(imares)
    imares = imares.astype(np.int8)


    nif = nib.Nifti1Image(imares,aff)
    nib.save(nif, sub_dir + '/output.nii.gz')

