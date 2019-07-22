#!/bin/bash

export SUBJECTS_DIR=/fs/pool/pool-ruben/Maria/workspace/github/my_tests_output/comparison_to_others/test_freesurfer_output
cd $SUBJECTS_DIR

for SUBJECT in noisy_sphere smooth_sphere noisy_torus smooth_torus noisy_cylinder smooth_cylinder  # is a folder
do
	cd $SUBJECT
	mkdir surf
	for file in *.vtk
	do
        echo $file
        BASE=${file%".surface.vtk"}

        # Convert VTK surface file to FreeSurfer surface:
        mris_convert $file surf/$BASE.surface

        # Calculate curvatures, including principal curvatures:
        mris_curvature_stats -f surface -m -o ${BASE}.stats -G --signedPrincipals --writeCurvatureFiles $SUBJECT $BASE

        # Convert the desired output curvatures to VTK:
        mris_convert -c surf/unknown.$BASE.surface.K1.crv surf/$BASE.surface $BASE.surface.K1.vtk
        mris_convert -c surf/unknown.$BASE.surface.K2.crv surf/$BASE.surface $BASE.surface.K2.vtk
        mris_convert -c surf/unknown.$BASE.surface.H.crv surf/$BASE.surface $BASE.surface.H.vtk

        # Remove "unknown." and "surface." in all the output files:
        for out_file in surf/unknown*
        do
            new_out_file=`echo $out_file | sed -e "s/unknown\.//" | sed -e "s/surface\.//"`
            mv $out_file $new_out_file
        done
	done
	cd ..
done
