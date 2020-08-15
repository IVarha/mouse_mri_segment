string="/scripts/log/mount_hello_kitty.log";
prefix="/scripts/log/mount_";
string=${string#$prefix}; #Remove prefix
suffix=".log";
string=${string%$suffix}; #Remove suffix
echo $string; #Prints "hello_kitty"
work_dir=$1
#ssss
a=1
rare=$work_dir/RARE/
for d in $work_dir/RARE/*; do
    echo "$d"
    stri=${d#$rare}
    stri=${stri%.nii.gz}
#    stri=${c#rare}
    echo $stri
#
    mkdir "$work_dir/sub$a"
    cp $work_dir/Masks_ARN/${stri}* "$work_dir/sub${a}/mask.nii.gz"
    cp $work_dir/RARE/${stri}* "$work_dir/sub${a}/mri.nii.gz"
    let "a += 1"
done



