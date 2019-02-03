# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 06:32:39 2016

@author: celia
"""

import numpy as np
import cv2
import glob

def gimp2cv(color):
    h, s, v = color
    return h/2, int(s * 2.55), int(v * 2.55) 

def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(vis, (int(x), int(y)), 20, color)

def get_pos(contour):
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx, cy

# Load webcam calibration values for undistort()
# calibration values calculated using cv2.calibrateCamera() previously
# for our webcam
calfile=np.load('webcam_calibration_data.npz')

newcameramtx=calfile['newcameramtx']
roi=calfile['roi']
mtx=calfile['mtx']
dist=calfile['dist']

blobsNotFound = []
images = glob.glob('test_images\\*.jpg')
# images = glob.glob('images\\*.jpg')

for fname in images:
    print fname
    orig_img = cv2.imread(fname)
    
    # undistort and crop
    dst = cv2.undistort(orig_img, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    crop_frame = dst[y:y+h, x:x+w]    

    # Blur image to remove noise
    frame=cv2.GaussianBlur(crop_frame, (3, 3), 0)

    # Switch image from BGR colorspace to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of purple color in HSV
    # colorMin = (90, 153, 180)
    # colorMax = (180, 255, 255)
    colorMin = (160, 0, 240)
    colorMax = (180, 70, 255)

    # Sets pixels to white if in purple range, else will be set to black
    mask = cv2.inRange(hsv, colorMin, colorMax)
        
    # Bitwise-AND of mask and purple only image - only used for display
    res = cv2.bitwise_and(frame, frame, mask= mask)

#    mask = cv2.erode(mask, None, iterations=1)
    # commented out erode call, detection more accurate without it

    # dilate makes the in range areas larger
    mask = cv2.dilate(mask, None, iterations=1)


    contour,hier = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)   
    contour.sort(key=lambda c:cv2.contourArea(c), reverse=True)
    for cnt in contour[2:]:
        cv2.drawContours(mask,[cnt],0,0,-1)
    for i in range(2):
        cnt = contour[i]
        x, y = get_pos(cnt)
        cv2.circle(frame, (x, y), 20, (0, 255, 255), 3)
 
    # Draw green circles around detected blobs
    # im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # draw_keypoints(frame, keypoints)
        
    # open windows with original image, mask, res, and image with keypoints marked
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    # cv2.imshow("keypoints", im_with_keypoints)            
        
    k = cv2.waitKey(20000)
    if k & 0xFF is ord('q'):
        break
cv2.destroyAllWindows()
