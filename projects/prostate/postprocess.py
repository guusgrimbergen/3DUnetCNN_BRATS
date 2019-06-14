import os, glob
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import label

from unet3d.utils.path_utils import get_filename_without_extension

from projects.prostate.config import config, config_unet, config_dict

data_path="/media/guus/Secondary/3DUnetCNN_BRATS/projects/prostate/database" 


prediction_dir="prediction/\
prostate_2018_is-256-256-128_crop-0_bias-0_denoise-0_norm-z_hist-0_ps-256-256-1_unet2d_crf-0_d-4_nb-16_loss-weighted_aug-1_model"
#prediction_dir="prediction_seperate/test"

prediction_path=os.path.join(data_path, prediction_dir)

def biggest_region_3D(im):
    struct=np.full((3,3,3),1)
    c=0
    lab, num_reg=label(im,structure=struct)
    h=np.zeros(num_reg+1)
    for i in range(num_reg):
        z=np.where(lab==(i+1),1,0)
        h[i+1]=z.sum()
        if h[i+1]==h.max():
            c=i+1
    lab=np.where(lab==c,1,0)
    return lab

for i, subject_dir in enumerate(glob.glob(os.path.join(prediction_path, "*"))):
    print("Postprocessing subject {} of {}".format(i, len(glob.glob(os.path.join(prediction_path, "*")))))


    ct_im=nib.load(os.path.join(subject_dir, "data_ct.nii.gz"))
    affine=ct_im.affine

    truth_post=np.zeros((256, 256, 128), dtype=np.uint8)

    prost_im=nib.load(os.path.join(subject_dir, "prediction_1.nii.gz")).get_fdata()
    blad_im=nib.load(os.path.join(subject_dir, "prediction_2.nii.gz")).get_fdata()
    rect_im=nib.load(os.path.join(subject_dir, "prediction_3.nii.gz")).get_fdata()

    #binarize the images
    prost_im=np.where(prost_im>=0.5, 1, 0)
    blad_im=np.where(blad_im>=0.5, 1, 0)
    rect_im=np.where(rect_im>=0.5, 1, 0)

    #dilate bladder to get rid of prostate "border" around bladder
    blad_im=ndimage.binary_dilation(blad_im, structure=np.ones((1,1,1))).astype(blad_im.dtype)

    #isolate the biggest connected volume from each image
    prost_im=biggest_region_3D(prost_im)
    blad_im=biggest_region_3D(blad_im)
    rect_im=biggest_region_3D(rect_im)

    #assign to single multilabel image; if prostate and bladder overlap, keep bladder only
    truth_post[prost_im==1]=1
    truth_post[(prost_im==1) & (blad_im==1)]=2
    truth_post[rect_im==1]=3

    truth_post=nib.Nifti1Image(truth_post, affine)
    nib.save(truth_post, os.path.join(subject_dir, "prediction.nii.gz"))
