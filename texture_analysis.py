from skimage.feature import greycomatrix, greycoprops
import numpy as np
import cv2
import SimpleITK as sitk
import radiomics

import os  # needed navigate the system to get the input data
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import radiomics
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape, shape2D ,gldm,ngtdm
import numpy as np
import six
import matplotlib.pyplot as plt
from PIL import Image
import csv

directory_name = r"C:\Users\Buslab_GG\Desktop\space\white_127"
mask = "C:\\Users\\Buslab_GG\\Desktop\\Texture\\mask.JPG"
output = directory_name + "output.csv"

tmp_number = []
mask = sitk.ReadImage(mask,sitk.sitkInt8)
ndmask = sitk.GetArrayFromImage(mask)

settings = {}
settings['binWidth'] = 25
settings['Normalize'] = True
# If enabled, resample image (resampled image is automatically cropped.
settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
settings['interpolator'] = sitk.sitkBSpline
settings['label'] = 1 #Since the mask area has a pixel value of 1 (otherwise it is 0).


for filename in os.listdir(directory_name):

    image = sitk.ReadImage(directory_name + "/" + filename,sitk.sitkInt8)

    ndImg = sitk.GetArrayFromImage(image)




    bb, correctedMask = imageoperations.checkMask(image, mask)
    if correctedMask is not None:
        mask = correctedMask
    image, mask = imageoperations.cropToTumorMask(image, mask, bb)


    ############
    tmp = []
    tmp_name = []

    glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
    glcmFeatures.enableAllFeatures()
    results = glcmFeatures.execute()
    for (key, val) in six.iteritems(results):
        #print(val,end=" ")
        tmp_name.append(key)
        tmp.append(val)
    glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **settings)
    glrlmFeatures.enableAllFeatures()

    results = glrlmFeatures.execute()

    for (key, val) in six.iteritems(results):
        #print(val,end=" ")
        tmp_name.append(key)
        tmp.append(val)
    glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **settings)
    glszmFeatures.enableAllFeatures()

    results = glszmFeatures.execute()
    for (key, val) in six.iteritems(results):
        #print(val,end=" ")
        tmp_name.append(key)
        tmp.append(val)
    gldmFeatures = gldm.RadiomicsGLDM(image, mask, **settings)
    gldmFeatures.enableAllFeatures()

    results = gldmFeatures.execute()
    for (key, val) in six.iteritems(results):
        #print(val,end=" ")
        tmp_name.append(key)
        tmp.append(val)
    ngtdmFeatures = ngtdm.RadiomicsNGTDM(image, mask, **settings)
    ngtdmFeatures.enableAllFeatures()


    results = ngtdmFeatures.execute()

    for (key, val) in six.iteritems(results):
        #print(val,end=" ")
        tmp_name.append(key)
        tmp.append(val)
    tmp_name.append("name")
    tmp.append(filename)
    tmp_number.append(tmp)
with open(output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(tmp_name)
    writer.writerows(tmp_number)