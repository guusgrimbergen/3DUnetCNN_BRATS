import nibabel as nib
from nilearn.image import smooth_img, resample_img, crop_img
import numpy as np
import glob, os

from unet3d.utils.volume import get_bounding_box

shapes=[]
spacings=[]

data_path="/media/guus/Secondary/Data"

for i, subject_dir in enumerate(glob.glob(os.path.join(data_path, "*"))):
    ct_name=os.path.join(subject_dir, "ct.nii.gz")
    ct_img=nib.load(ct_name)
    ct_data=ct_img.header

    shape=ct_data.get_data_shape()
    spacing=ct_data.get_zooms()

    

    shapes.append(shape)
    spacings.append(spacing)

spacings=np.array(spacings)
print(np.amin(spacings, axis=0))
print(np.mean(spacings,axis=0))
print(np.amax(spacings,axis=0))

shapes=np.array(shapes)
print(np.amin(shapes, axis=0))
print(np.mean(shapes,axis=0))
print(np.amax(shapes,axis=0))


