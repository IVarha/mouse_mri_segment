
struct_image=$2
workdir=$1
echo "--image-dimensionality 3  --input-image $struct_image --output $workdir/bfc.nii.gz --shrink-factor 4 -b --convergence [70x70x70x70,0.000001]"

N4BiasFieldCorrection --image-dimensionality 3  --input-image $struct_image --output $workdir/bfc.nii.gz --shrink-factor 4 -b --convergence [70x70x70x70,0.000001]