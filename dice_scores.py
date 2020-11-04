import sys

import nibabel as nib


def dice_coeff(img1, img2):
    overlap = img1 & img2

    s1 = overlap.sum()
    s2 = img1.sum()
    s3 = img2.sum()

    return 2*s1/(s2 + s3)

def jaccard_coeff(img1, img2):
    overlap = img1 & img2
    nov = img1 | img2
    s1 = overlap.sum()
    s = nov.sum()
    return s1/s

if __name__ == "__main__":
    work_path = sys.argv[1]
    im1_file = nib.load(sys.argv[1] + "/"+ sys.argv[2])
    im2_file = nib.load(sys.argv[1] + "/" + sys.argv[3])
    file_name = sys.argv[4]
    out_path = sys.argv[5]

    im1 = im1_file.get_fdata() > 0
    im2 = im2_file.get_fdata() > 0
    dice = dice_coeff(im1,im2)
    jac = jaccard_coeff(im1,im2)
    file_object = open(out_path + "/" + file_name , 'a')
    file_object.write(str(dice) +"," + str(jac)+ "\n")
    file_object.close()