import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

plt.rcParams['figure.figsize'] = (10.0, 10.0) # make the inline plots bigger


def overlay_segmentation(image, segmentation):

    overlay = np.ma.masked_where(segmentation == 0, segmentation)
    plt.imshow(image, cmap="bone", vmin=-1000, vmax=150)
    plt.imshow(overlay, cmap="tab10", vmin=0, vmax=1)
    plt.savefig("overlay.png")


image = sitk.ReadImage('example_images/RADCURE3304.nrrd')
segmentation = sitk.ReadImage('example_images/RADCURE3304_mask.nrrd')

image = sitk.GetArrayFromImage(image)
segmentation = sitk.GetArrayFromImage(segmentation)

nonzero_slices = np.where(segmentation.sum((1,2))>0)[0]

middle_slice = image[nonzero_slices[len(nonzero_slices)//2]]
middle_seg_slice = segmentation[nonzero_slices[len(nonzero_slices)//2]]
print("HEY")

overlay_segmentation(middle_slice, middle_seg_slice)