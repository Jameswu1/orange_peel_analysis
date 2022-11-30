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


def main():
    #dataset
    directory_name = r"C:\Users\Buslab_GG\Desktop\orange_peel_analysis\data_normalize"
    mask = r"C:\Users\Buslab_GG\Desktop\orange_peel_analysis\mask.JPG"
    output = r"C:\Users\Buslab_GG\Desktop\orange_peel_analysis\data_texture.csv"

    tmp_number = []
    mask = sitk.ReadImage(mask,sitk.sitkInt8)
    ndmask = sitk.GetArrayFromImage(mask)
    
    #radiomics basic setting
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


        #tmp_space
        tmp = []
        tmp_name = []

        #texture_analysis

        #GLCM
        glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
        glcmFeatures.enableAllFeatures()
        results = glcmFeatures.execute()
        for (key, val) in six.iteritems(results):
            tmp_name.append(key)
            tmp.append(val)

        #GLRLM
        glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **settings)
        glrlmFeatures.enableAllFeatures()
        results = glrlmFeatures.execute()
        for (key, val) in six.iteritems(results):
            tmp_name.append(key)
            tmp.append(val)

        #GLSZM
        glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **settings)
        glszmFeatures.enableAllFeatures()
        results = glszmFeatures.execute()
        for (key, val) in six.iteritems(results):
            tmp_name.append(key)
            tmp.append(val)

        #GLDM
        gldmFeatures = gldm.RadiomicsGLDM(image, mask, **settings)
        gldmFeatures.enableAllFeatures()
        results = gldmFeatures.execute()
        for (key, val) in six.iteritems(results):
            tmp_name.append(key)
            tmp.append(val)


        #NGTDM
        ngtdmFeatures = ngtdm.RadiomicsNGTDM(image, mask, **settings)
        ngtdmFeatures.enableAllFeatures()
        results = ngtdmFeatures.execute()
        for (key, val) in six.iteritems(results):
            tmp_name.append(key)
            tmp.append(val)


        tmp_name.append("name")
        tmp.append(filename)
        tmp_number.append(tmp)

    #save to the csv
    with open(output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(tmp_name)
        writer.writerows(tmp_number)

if __name__ == '__main__':
    main()
