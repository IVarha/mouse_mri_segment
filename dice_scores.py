import sys

import nibabel as nib
import csv


def dice_coeff(mask_im, img2):
    overlap = mask_im & img2

    s1 = overlap.sum()
    s2 = mask_im.sum()
    s3 = img2.sum()

    return 2*s1/(s2 + s3)

def jaccard_coeff(mask_im, img2):
    overlap = mask_im & img2
    nov = mask_im | img2
    s1 = overlap.sum()
    s = nov.sum()
    return s1/s


def sensitivity_coeff(label_im, test_im ):
    overlap = label_im & test_im
    nov = label_im
    s1 = overlap.sum()
    s = nov.sum()
    return s1/s

def specificity_coeff(label_im, test_im ):
    overlap = (~ label_im) & (~ test_im)
    nov = ~ label_im
    s1 = overlap.sum()
    s = nov.sum()
    return s1/s

if __name__ == "__main__":
    work_path = sys.argv[1]
    manual_mask = nib.load(sys.argv[1] + "/" + sys.argv[2])
    im2_file = nib.load(sys.argv[1] + "/" + sys.argv[3])
    file_name = sys.argv[4]
    out_path = sys.argv[5]

    im1 = manual_mask.get_fdata() > 0
    im2 = im2_file.get_fdata() > 0
    dice = dice_coeff(im1,im2)
    jac = jaccard_coeff(im1,im2)
    sens = sensitivity_coeff(label_im=im1,test_im=im2)
    specificity = specificity_coeff(label_im=im1,test_im=im2)

    file_object = open(out_path + "/" + file_name , 'a')
    wr = csv.writer(file_object)
    wr.writerow([dice,jac,sens,specificity])
    file_object.close()