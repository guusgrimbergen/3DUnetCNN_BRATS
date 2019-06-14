import nibabel as nib
import nrrd
from nilearn.image import smooth_img, resample_img, crop_img
import numpy as np
import glob, os

from unet3d.utils.volume import get_bounding_box

spacings=[]
shapes=[]
resample_shapes=[]

data_path="/media/guus/Secondary/MICCAI"

for i, subject_dir in enumerate(glob.glob(os.path.join(data_path, "*"))):
    ct_name=os.path.join(subject_dir, "ct.nii.gz")
    ct_img=nib.load(ct_name)
    ct_data=ct_img.header

    shape=ct_data.get_data_shape()
    spacing=ct_data.get_zooms()

    resample_shape=[shape[0]*spacing[0], shape[1]*spacing[1], (shape[2]*spacing[2])/2.5]

    shapes.append(shape)
    spacings.append(spacing)
    resample_shapes.append(resample_shape)


resample_shapes=np.array(resample_shapes)
print(np.amin(resample_shapes, axis=0))
print(np.mean(resample_shapes,axis=0))
print(np.amax(resample_shapes,axis=0))

spacings=np.array(spacings)
print(np.amin(spacings, axis=0))
print(np.mean(spacings,axis=0))
print(np.amax(spacings,axis=0))

shapes=np.array(shapes)
print(np.amin(shapes, axis=0))
print(np.mean(shapes,axis=0))
print(np.amax(shapes,axis=0))

