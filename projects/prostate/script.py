import nibabel as nib
from nilearn.image import smooth_img, resample_img, crop_img
import numpy as np
import glob, os

from unet3d.utils.volume import get_bounding_box
from unet3d.utils.utils import resize
from unet3d.utils.utils import pad

import multiprocessing as mp 

data_path="/media/guus/Secondary/Data"

writing_path="/media/guus/Secondary/Data_prostate_256_256_128"



def preprocess(subject_dir):

    patient_name=os.path.basename(subject_dir)
    writing_dir=os.path.join(writing_path, patient_name)
    os.makedirs(writing_dir, exist_ok=True)

    ct_name=os.path.join(subject_dir, "ct.nii.gz")
    ct_img=nib.load(ct_name)
    ct_data=ct_img.header

    shape=ct_data.get_data_shape()
    spacing=ct_data.get_zooms()

    resample_shape=[shape[0]*spacing[0], shape[1]*spacing[1], (shape[2]*spacing[2])/2.5]
    padding_shape=[700, 700, 201]
    target_shape=[256, 256, 128]

    img_names=[ct_name, os.path.join(subject_dir, "truth.nii.gz")]

    for img in img_names:

        image_name=os.path.basename(img)

        if image_name == "ct.nii.gz":
            interpolation="linear"
        else:
            interpolation="nearest"
        
        image=nib.load(img)

        #resample to equal voxel size
        resampled=resize(image, resample_shape, interpolation=interpolation)
        #nib.save(resampled, os.path.join(writing_dir, "resampled_" + image_name))

        #pad to same size
        padded=pad(resampled, padding_shape, interpolation=interpolation)
        #nib.save(padded, os.path.join(writing_dir, "padded_" + image_name))

        #downsample to 256  
        downsampled=resize(padded, target_shape, interpolation=interpolation)
        nib.save(downsampled, os.path.join(writing_dir, image_name))
    

pool=mp.Pool(mp.cpu_count())

pool.map(preprocess, [subject_dir for i, subject_dir in enumerate(glob.glob(os.path.join(data_path, "*")))])

pool.close()
'''
for i, subject_dir in enumerate(glob.glob(os.path.join(data_path, "*"))):
    preprocess(subject_dir)
'''
