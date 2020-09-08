work_dir=$1
#refer=$3
#labels=$4
#prefix_subdirs=$2
#im_name=$work_dir/names
#ssss
#while  IFS= read -r line
#do
#    echo "$line"
#    name=$line
#    break
#done < "$im_name"
name="mri.nii.gz"

for d in $work_dir//*; do
    echo "$d"

    arr=()
#    FILL ARRAY
    for fil in $d/*; do
#      echo $fil
      arr+=($fil)
#
    done

    for img in "${arr[@]}";do
      echo $img
      mkdir $d/work_dir
  #    RESULT MASK FOR Bias field correction
      python3 remove_backgr_mouse.py $img $d/work_dir none
  #    BIAS FIELD CORRECTION
      ./bfc.sh $d/work_dir $d/work_dir/mri2.nii.gz $d/work_dir/mri_bfc_mask_test.nii.gz
  #
  #    SEGMENTS IMAGE
      python3 segment_mouse.py $d/work_dir/bfc.nii.gz $d/work_dir/bfc.nii.gz $d/work_dir
  #    FILL HOLES
      python3 fill_holes.py $d/work_dir output.nii.gz output2.nii.gz 1
#      CALCULATE DICE SCORES
      stri=${img%.nii}
#      echo $stri
      mv -f $d/work_dir $stri

    done
#    echo ${arr[@]}


done