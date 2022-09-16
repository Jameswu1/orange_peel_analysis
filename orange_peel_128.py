import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import glob


directory_name = r"C:\Users\Buslab_GG\Desktop\orangepeel_new\15-03-52-ora"
output = r"C:\Users\Buslab_GG\Desktop\space"


if not os.path.isdir(output):
    os.mkdir(output)

for filename in os.listdir(directory_name):
    print(directory_name + "/" + filename)
    img = cv2.imread(directory_name + "/" + filename)
    img = cv2.bilateralFilter(img, 23, 50, 50)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret , dst = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    dst = cv2.erode(dst, kernel, iterations = 1)  
    x = cv2.Sobel(dst, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(dst, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)# 轉回uint8
    absY = cv2.convertScaleAbs(y)
    dst1 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    kernel1 = np.ones((7,7), np.uint8)
    dst1 = cv2.dilate(dst1, kernel1, iterations = 2)
    dst1[dst1>60] = 255
    #cv2.imwrite(output + "/" + "1" + filename,dst1)
    counter , hierarcy = cv2.findContours(dst1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    tmp2 = 1
    for i in counter:
        if cv2.contourArea(i) > 500:
            M = cv2.moments(i)
            x = int( M['m10'] / M['m00'] )
            y = int( M['m01'] / M['m00'] )
            if y + 100 > 2048 or x + 100 > 2448 or x - 100 < 0 or y - 100 < 0 :
                continue
            test = img[ y-100 : y+100 , x-100 : x+100 ]
            test = cv2.resize(test, (128, 128), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output + "/" + filename,test)

    #img = img[548:686,148:349]
    #img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    #cv2.imwrite(output + "/" + filename,img)
