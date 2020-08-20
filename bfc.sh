
struct_image=$2
workdir=$1
mask=$3
echo "--image-dimensionality 3  --input-image $struct_image --output $workdir/bfc.nii.gz --shrink-factor 4 -b --convergence [70x70x70x70,0.000001]  -x $mask"

N4BiasFieldCorrection --image-dimensionality 3  --input-image $struct_image --output $workdir/bfc.nii.gz --shrink-factor 4 -b --convergence [70x70x70x70,0.000001] -x $mask