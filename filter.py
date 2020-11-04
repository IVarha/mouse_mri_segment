import math
import sys

import nibabel as nib
import numpy as np
import scipy.ndimage.morphology as morph
import scipy.ndimage.filters as filt

if __name__ == "__main__":
    im_file = nib.load(sys.argv[1])

    # im_seg = nib.load(sys.argv[2])
    res_path = sys.argv[2]
    out = sys.argv[3]

    siz = int(sys.argv[4])
    img = im_file.get_fdata()

    imares = filt.median_filter(img, size=siz)
    nif = nib.Nifti1Image(imares.astype(np.float), im_file.affine)
    nib.save(nif, res_path + '/' + out)
