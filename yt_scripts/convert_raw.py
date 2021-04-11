import nibabel as nib
import numpy as np

img = nib.load("../data/3d_sag_t1_bravo_nq.nii.gz")
arr = img.get_fdata()
arr = arr.reshape((arr.shape[2], arr.shape[1], arr.shape[0]))
def rescale(arr):
    mi = arr.min()
    ma = arr.max()
    arr_scaled = arr * 255 / ma
    return arr_scaled.astype('uint8')

arr_8bit = rescale(arr)
print(arr_8bit.max())
print(arr_8bit.min())
print(arr_8bit.shape)
arr_8bit.tofile("mri_head_uint8_256x256x166_resx1p2_resy1_resz1_mm.raw")

