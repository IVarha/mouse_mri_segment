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

for d in $work_dir//$prefix_subdirs*; do
    echo "$d"
    echo "$d/$name"

#    ./bfc.sh $d $d/$name
#    ./preregister.sh $d bfc.nii.gz $refer $labels
    python3 remove_backgr_mouse.py $d/mri.nii.gz $d 0.12
#    python3 segment_mouse.py $d/bfc.nii.gz $d/bfc.nii.gz $d
done