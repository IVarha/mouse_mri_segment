work_dir=$1
refer=$3
labels=$4
prefix_subdirs=$2
im_name=$work_dir/names
#ssss
#while  IFS= read -r line
#do
#    echo "$line"
#    name=$line
#    break
#done < "$im_name"
name="mri.nii.gz"
# segment alex data
for d in $work_dir//$prefix_subdirs*; do
    echo "$d"
    echo "$d/$name"
    #
    python3 filter.py $d/mri.nii.gz $d mri2.nii.gz 3
#    RESULT MASK FOR Bias field correction

    python3 remove_backgr_mouse.py $d/mri2.nii.gz $d mri2.nii.gz mri_bfc_mask_test.nii.gz none

    python3 remove_backgr_mouse.py $d/mri.nii.gz $d mri2.nii.gz none none
#    BIAS FIELD CORRECTION
    ./bfc.sh $d $d/mri2.nii.gz $d/mri_bfc_mask_test.nii.gz
#
#    SEGMENTS IMAGE
    python3 segment_mouse.py $d/bfc.nii.gz $d/bfc.nii.gz $d 0.5
    # graph_cut segmentation
    python3 gc_model.py $d/bfc.nii.gz $d gc.nii.gz
#    FILL HOLES
    python3 fill_holes.py $d output.nii.gz output2.nii.gz 10
#    CALCULATE DICE SCORES
    python3 dice_scores.py $d mask.nii.gz output2.nii.gz dice_ac.csv $work_dir

    python3 dice_scores.py $d mask.nii.gz gc.nii.gz dice_gc.csv $work_dir
done