work_dir=$1
struct_image=$2
refer=$3
labels=$4
#echo $struct_image
echo $work_dir
echo $refer
#echo $labels
echo "flirt -in $work_dir/$struct_image -ref $refer -out $work_dir/1_stage -omat $work_dir/affine_1.mat -dof 12"
flirt -in $work_dir/$struct_image -ref $refer -out $work_dir/1_stage -omat $work_dir/affine_1.mat -dof 12
flirt -in $work_dir/1_stage.nii.gz -ref $refer -out $work_dir/2_stage -omat $work_dir/affine_2.mat -dof 12
convert_xfm -omat $work_dir/combine_affine_t1.mat -concat $work_dir/affine_2.mat $work_dir/affine_1.mat
convert_xfm -omat $work_dir/combined_affine_reverse.mat -inverse $work_dir/combine_affine_t1.mat
flirt -in $labels -out $work_dir/out_label -ref $work_dir/$struct_image -applyxfm -init $work_dir/combined_affine_reverse.mat
