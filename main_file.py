import keyboard
import cv2
import numpy as np
import math
import os
import time
import pyautogui
from point import p_crop
from hand import h_crop
from fin import f_crop
from thumbdown import t_crop

#os.chdir("C:/Users/preks/OneDrive/Desktop/CV_Project-master/CV_Project-master")
h1_cascade=cv2.CascadeClassifier('hand.xml')
point_cascade = cv2.CascadeClassifier('point1.xml')
fin_cascade=cv2.CascadeClassifier('fin_2.xml')
#fist_cascade=cv2.CascadeClassifier('fist.xml')
thumbdown_cascade = cv2.CascadeClassifier('thumbdown.xml')
cap = cv2.VideoCapture(0)
#ca=0

while(cap.isOpened()):    
        _,img=cap.read()
        
        cv2.rectangle(img, (300,300), (100,100), (0,255,0),0)
        crop_img = img[100:300, 100:300]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        point=point_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,flags=0, minSize=(100,80))
        fin=fin_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,flags=0, minSize=(100,80))
        hand=h1_cascade.detectMultiScale(gray,1.1, 5)
        #fist=fist_cascade.detectMultiScale(gray,1.3, 5)
        thumbdown=thumbdown_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,flags=0, minSize=(100,80))
        #print("cascading done")
       
                
        for (x,y,w,h) in point:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                #crop_img=img[y:y+h,x:x+w]
                ha=2
                p_crop(crop_img,img)
                #time.sleep(0.5)
                print("Slide Zoomed In!")

        for (x,y,w,h) in fin:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                #crop_img=img[y:y+h,x:x+w]
                f_crop(crop_img,img)
                time.sleep(0.5)
                print("Slide Moved Forward!")

        for (x,y,w,h) in thumbdown:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                #crop_img=img[y:y+h,x:x+w]
                ha=5
                t_crop(crop_img,img)
                time.sleep(0.5)
                print("Slide Moved Backwards!")

        for (x,y,w,h) in hand:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                #crop_img=img[y:y+h,x:x+w]
                h_crop(crop_img,img)
                time.sleep(0.5)
                print("Slide Zoomed Out!")

        '''for (x,y,w,h) in fist:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_color=img[y:y+h,x:x+w]
                #crop_img=img[y:y+h,x:x+w]
                h_crop(crop_img,img)
                time.sleep(0.5)
                print("out of FIST")'''


                
        cv2.imshow('Feed',img)
        
        k=cv2.waitKey(10)
        if k==27:
                break
cv2.destroyAllWindows()

    
