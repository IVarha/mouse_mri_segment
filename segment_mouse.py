import skimage.segmentation as sg
import skimage.morphology as morph
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
    pre_mask = im_seg.get_fdata()
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
    images = cv_model(img,init_mask=imares,num_divisions=10,num_iterations=100,start_end=[0.5,4])

    imares = combine_image(images)
    nif = nib.Nifti1Image(imares, im_file.affine)
    nib.save(nif, res_path + '/combined.nii.gz')
    imares = imares > 0.9

    # imares = sg.morphological_chan_vese(img, iterations=30, init_level_set=imares > 0, lambda1=2)
    # imares = sg.morphological_chan_vese(img, iterations=15, init_level_set=imares > 0, lambda1=2.5)
    # imares = sg.morphological_chan_vese(img, iterations=2, init_level_set=imares > 0)
    # imares = sg.morphological_chan_vese(img, iterations=30, init_level_set=imares > 0, lambda1=4)
    # imares = sg.morphological_chan_vese(img, iterations=2, init_level_set=imares > 0, lambda1=2)



    # imm = list(np.zeros(imares.shape))
    # for i in range(len(imares[0,0,:])):
    #     im2 = morph.area_opening(np.asarray(imares[:,:,i]))
    #     for j in range(len(im2)):
    #         for k in range(len(im2[0])):
    #             imm[j][k][i] = im2[j,k]
    # imares = np.asarray(imm)
    # imares = morph.area_closing(imares)

    imares = sg.morphological_chan_vese(img, iterations=2, init_level_set=imares > 0)
    imares = imares.astype(np.int8)
    # imares = sg.watershed(img,markers=imares)
    # gimage = sg.inverse_gaussian_gradient(img)
    # gim = nib.Nifti1Image(gimage, im_seg.affine)
    # geo_seg =sg.morphological_geodesic_active_contour(gimage,230,init_level_set=pre_mask>0,balloon=1)

    # nib.save(gim, 'grads.nii.gz')

    # nib.save(nif, 'output.nii.gz')
    # nif = nib.Nifti1Image(geo_seg, im_seg.affine)
    nif = nib.Nifti1Image(imares, im_file.affine)
    nib.save(nif, res_path + '/output.nii.gz')

    print(1)