import glob, os
import nrrd
import nibabel as nib
import numpy as np

data_path="/media/guus/Secondary/MICCAI"

for i, subject_dir in enumerate(glob.glob(os.path.join(data_path, "*"))):
    print("Converting patient", i, " of ", len(glob.glob(os.path.join(data_path, "*"))))

    ct_img, ct_data = nrrd.read(os.path.join(subject_dir, "img.nrrd"))
    truth_img, truth_data = nrrd.read(os.path.join(subject_dir, "truth.nrrd"))
    spacing = ct_data["space directions"].diagonal()


    affine = np.array([(spacing[0],0,0,0), (0,-1*spacing[1],0,0), (0,0,spacing[2],0), (0,0,0,1)] )
    #affine = np.array([(spacing[0],0,0,0), (0, spacing[1],0,0), (0,0,spacing[2],0), (0,0,0,1)] )

    ct_img = nib.Nifti1Image(ct_img, affine)
    truth_img = nib.Nifti1Image(truth_img, affine)


    nib.save(ct_img, os.path.join(subject_dir, "ct.nii.gz"))
    nib.save(truth_img, os.path.join(subject_dir, "truth.nii.gz"))
