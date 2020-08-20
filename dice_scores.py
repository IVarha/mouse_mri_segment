import sys

import nibabel as nib


def dice_coeff(img1, img2):
    overlap = img1 & img2

    s1 = overlap.sum()
    s2 = img1.sum()
    s3 = img2.sum()

    return 2*s1/(s2 + s3)


if __name__ == "__main__":
    work_path = sys.argv[1]
    im1_file = nib.load(sys.argv[1] + "/"+ sys.argv[2])
    im2_file = nib.load(sys.argv[1] + "/" + sys.argv[3])
    out_path = sys.argv[4]

    im1 = im1_file.get_fdata() > 0
    im2 = im2_file.get_fdata() > 0
    dice = dice_coeff(im1,im2)

    file_object = open(out_path + "/dice.csv" , 'a')
    file_object.write(str(dice) + "\n")
    file_object.close()