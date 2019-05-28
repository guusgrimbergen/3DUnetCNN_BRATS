import nibabel as nib
import os, glob
import numpy as np

from keras.models import load_model

data_path="/media/guus/Secondary/Data_subset"
model_path="/media/guus/Secondary/Networks/AlexNet0.hdf5"

model=load_model(model_path)


'''
for subject_dir in glob.glob(os.path.join(data_path, "*")):
    ct=nib.load(os.path.join(subject_dir, "ct.nii.gz"))
    ct=ct.get_fdata()
    prost=nib.load(os.path.join(subject_dir, "Prostate.nii.gz"))
    prost=prost.get_fdata()
'''

subject_dir=os.path.join(data_path, "0a2c62cd7028 Brandi Purvis")
ct=nib.load(os.path.join(subject_dir, "ct.nii.gz"))
ct=ct.get_fdata()
prost=nib.load(os.path.join(subject_dir, "Prostate.nii.gz"))
prost=prost.get_fdata()

input=np.array([ct, prost])






print("Done")