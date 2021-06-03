
<b>Prerequisites</b>

N4biasFieldCorrection should be accessible for user (be in PATH)

python 3.7+ 

To install all requirenments
> pip install -r requirenments.txt

<b>Run</b> 

All data should be placed in folders like

rootdir/"prefix""subject_id"/*
And mri files which need to be segmented should be named mri.nii.gz

To segment all subjects you need to run

> sh segment_all.sh "rootdir" "prefix"
