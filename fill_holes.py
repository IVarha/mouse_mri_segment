import sys

import nibabel as nib
import numpy as np
import scipy.ndimage.morphology as morph
from gc_model import post_processing
if __name__ == "__main__":
    work_path = sys.argv[1]
    im_file = nib.load(sys.argv[1] + "/"+ sys.argv[2])
    out = sys.argv[3]
    thresh = int(sys.argv[4])
    img = im_file.get_fdata()

    img = img > 0
    if thresh > 0:
        img = post_processing(img,thresh)
        img = post_processing(img, thresh)
    img = img.astype(np.int8)
    nif = nib.Nifti1Image(img, im_file.affine)
    nib.save(nif, work_path + '/' + out)
