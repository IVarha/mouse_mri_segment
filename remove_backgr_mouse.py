import skimage.segmentation as sg
import skimage.morphology as morph
import skimage.measure as measure
import nibabel as nib
import sys
import numpy as np
import scipy.ndimage.measurements as measurements
import segment_mouse


if __name__ == "__main__":
    im_file = nib.load(sys.argv[1])

    res_path = sys.argv[2]
    thresh = float(sys.argv[3])
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
    # imares = sg.morphological_chan_vese(img,iterations=40,init_level_set=pre_mask>0,lambda1=2.2)
    #
    # marker = np.where(imares == 1)
    # xm = (max(marker[0]) - min(marker[0])) // 2 + min(marker[0])
    # ym = (max(marker[1]) - min(marker[1])) // 2 + min(marker[1])
    # zm = (max(marker[2]) - min(marker[2])) // 2 + min(marker[2])
    imares[:, :, :] = 0
    # imares[xm - 20:xm + 20, ym - 20:ym + 20, zm - 2:zm + 2] = 1
    imares[xm - dx:xm + dx, ym - dy:ym + dy, zm - dz:zm + dz] = 1
    imares = sg.morphological_chan_vese(img, iterations=25, init_level_set=imares > 0, lambda2=2)
    images = segment_mouse.cv_model(img,init_mask=imares,num_divisions=10,num_iterations=100,start_end=[0.5,4])

    imares = segment_mouse.combine_image(images)
    nif = nib.Nifti1Image(imares, im_file.affine)
    imares = imares > 0
    imares2 = np.zeros(img.shape)
    for i in range(imares.shape[0]):
        for j in range(imares.shape[1]):
            for k in range(imares.shape[2]):
                if (imares[i,j,k] > 0) | (img[i,j,k] > thresh):
                    imares2[i,j,k] = img[i,j,k]
                else:
                    imares2[i, j, k] = 0
    # imares = sg.morphological_chan_vese(img, iterations=30, init_level_set=imares > 0, lambda1=2)
    # imares = sg.morphological_chan_vese(img, iterations=15, init_level_set=imares > 0, lambda1=2.5)
    # imares = sg.morphological_chan_vese(img, iterations=2, init_level_set=imares > 0)
    # imares = sg.morphological_chan_vese(img, iterations=30, init_level_set=imares > 0, lambda1=4)
    # imares = sg.morphological_chan_vese(img, iterations=2, init_level_set=imares > 0, lambda1=2)
    imares = imares>0
    nif = nib.Nifti1Image(imares.astype(np.int8), im_file.affine)
    nib.save(nif, res_path + '/mri_bfc_mask_test.nii.gz')

    # imm = list(np.zeros(imares.shape))
    # for i in range(len(imares[0,0,:])):
    #     im2 = morph.area_opening(np.asarray(imares[:,:,i]))
    #     for j in range(len(im2)):
    #         for k in range(len(im2[0])):
    #             imm[j][k][i] = im2[j,k]
    # imares = np.asarray(imm)
    # imares = morph.area_closing(imares)

    # imares = sg.watershed(img,markers=imares)
    # gimage = sg.inverse_gaussian_gradient(img)
    # gim = nib.Nifti1Image(gimage, im_seg.affine)
    # geo_seg =sg.morphological_geodesic_active_contour(gimage,230,init_level_set=pre_mask>0,balloon=1)

    # nib.save(gim, 'grads.nii.gz')

    # nib.save(nif, 'output.nii.gz')
    # nif = nib.Nifti1Image(geo_seg, im_seg.affine)
    nif = nib.Nifti1Image(imares2, im_file.affine)
    nib.save(nif, res_path + '/mri2.nii.gz')

    print(1)