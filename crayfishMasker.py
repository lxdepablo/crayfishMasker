import numpy as np
import cv2
import os

i = 1

#filepath of video file
vidName = "videos/056_L_1Cropped.avi"

#read in video file
vid = cv2.VideoCapture(vidName)

#loop over and generate mask for each frame of video
while vid.isOpened():
    #read current frame as image
    ret, image = vid.read()

    if image is None:
        break

    #convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #remove noise (including suspended PIV beads)
    denoise = cv2.fastNlMeansDenoising(grayscale, None, 30, 7, 21)

    #blur image
    blur = cv2.medianBlur(denoise, 5)

    # threshold image
    #find dark colors at center of objects
    thresh1 = cv2.threshold(blur,25,255,cv2.THRESH_BINARY_INV)[1]
    #block out edges of image
    h,w = thresh1.shape
    thresh1 = cv2.rectangle(thresh1, (0,int(h*3/4)), (w,h), 0, -1)
    thresh1 = cv2.rectangle(thresh1, (0,0), (int(w/4),h), 0, -1)
    thresh1 = cv2.rectangle(thresh1, (int(w*3/4),0), (w,h), 0, -1)
    thresh1 = cv2.rectangle(thresh1, (0,0), (w,int(h/8)), 0, -1)

    #find light colors on outlines of objects
    #thresh2 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY)[1]
    thresh2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    #combine light and dark for complete objects
    thresh = cv2.add(thresh1,thresh2)

    #mask out platform
    #find platform
    h,w = thresh.shape
    currH = int(h*15/16)
    currPix = thresh[currH-1, int(w/2)]
    while currPix == 0:
        currH-=1
        currPix = thresh[currH-1, int(w/2)]
    while currPix != 0:
        currH-=1
        currPix = thresh[currH-1, int(w/2)]
    #check if found platform is in bottom of image
    if currH<(h/2):
        currH = h
    #mask out platform
    thresh = cv2.rectangle(thresh, (0,currH-5), (w,h), 0, -1)

    #do contour detection
    contours, hier = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    for c in contours:
        area = cv2.contourArea(c)

        # Fill very small contours with zero (erase small contours).
        if area < 20:
            cv2.fillPoly(thresh, pts=[c], color=0)
            continue

    #close gaps
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)), iterations = 2)
    contours, hier = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #save mask as tif
    mask = np.zeros(grayscale.shape, np.uint8)
    #find and draw biggest contour
    maxArea = max(contours, key=cv2.contourArea)

    cv2.drawContours(image=mask, contours=[maxArea], contourIdx=-1, color=(255), thickness=-1, lineType=cv2.LINE_AA)
    #cv2.drawContours(image=mask, contours=contours, contourIdx=-1, color=(255), thickness=-1, lineType=cv2.LINE_AA)
    mask = cv2.rectangle(mask, (0,currH+1), (w-1,h), 255, -1)


    imgName = 'masks/' + bin(i) + '.tif'
    i+=1
    cv2.imwrite(imgName, mask)

vid.release()
