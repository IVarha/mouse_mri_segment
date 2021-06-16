
<b>Prerequisites</b>

N4biasFieldCorrection should be accessible for user (be in PATH)

python 3.7+ 

To install all requirenments
> pip install -r requirenments.txt

Further step is to install library into the virtual environment

Cmake 3.10+ needs to be accessible for user
> pip install "project-dir/"

<b>Run</b> 

All data should be placed in folders like

rootdir/"prefix""subject_id"/*
And mri files which need to be segmented should be named mri.nii.gz

To segment all subjects you need to run

> sh segment_all.sh "rootdir" "prefix"

To select different thresholding (CV script) after segment_all was runned
> python threshold_seg.py "folder_with combined.nii.gz()" "threshord from 0 to 1" "iterations of binary dilation-erosions"

Another possible option is to change it inside of the segment_all.sh in segment_mouse part