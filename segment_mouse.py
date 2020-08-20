import skimage.segmentation as sg
import scipy.ndimage.morphology as morph
import skimage.measure as measure
import nibabel as nib
import sys
import numpy as np
import scipy.ndimage.measurements as measurements


def cv_model(image,init_mask, num_divisions,num_iterations,start_end):

    dt = (start_end[1] - start_end[0])/num_divisions
    res = []
    for i in range(num_divisions):
        mares = sg.morphological_chan_vese(image, iterations=num_iterations, init_level_set=init_mask > 0,lambda1=start_end[0] + i*dt,lambda2=start_end[1]- i*dt)
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
    im_file = nib.load(sys.argv[1])

    im_seg = nib.load(sys.argv[2])
    res_path = sys.argv[3]
    img = im_file.get_fdata()


    cent = measurements.center_of_mass(img)
    cent = [round(x) for x in cent]
    moms = measure.moments(img,3)
    imares = np.zeros(img.shape)
    xm = cent[0]
    dx = round(img.shape[0]/20)
    if dx < 1:
        dx = 2
    ym = cent[1]
    dy = round(img.shape[1]/20)
    if dy < 1:
        dy = 2
    zm = cent[2]
    dz = round(img.shape[2]/20)
    if dz < 1:
        dz = 2

    imares[:, :, :] = 0

    # imares[xm - 20:xm + 20, ym - 20:ym + 20, zm - 2:zm + 2] = 1
    imares[xm - dx:xm + dx, ym - dy:ym + dy, zm - dz:zm + dz] = 1
    imares = sg.morphological_chan_vese(img, iterations=25, init_level_set=imares > 0, lambda2=2)
    images = cv_model(img,init_mask=imares,num_divisions=10,num_iterations=100,start_end=[0.5,4])

    imares = combine_image(images)
    nif = nib.Nifti1Image(imares, im_file.affine)
    nib.save(nif, res_path + '/combined.nii.gz')
    imares = imares > 0.9



    imares = sg.morphological_chan_vese(img, iterations=2, init_level_set=imares > 0)
    imares = morph.binary_fill_holes(imares)
    imares = morph.binary_erosion(imares)
    imares = morph.binary_dilation(imares)
    imares = imares.astype(np.int8)


    nif = nib.Nifti1Image(imares, im_file.affine)
    nib.save(nif, res_path + '/output.nii.gz')

    print(1)