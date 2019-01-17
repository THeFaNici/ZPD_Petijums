# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:26:18 2019

@author: Nils
"""
#Attēli:Attels1 Attels2 Attels3 Attels4 Attels5
import cv2
import matplotlib.pyplot as plt
import time

#note time before detection 
t1 = time.time()  

#ielādē attēlu
colored_img = cv2.imread("Attels5.jpg")
#kaskādes klasifikators
f_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def faces_detected_img(f_cascade, colored_img, scaleFactor = 1.2):
    img_copy = colored_img.copy()
          
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY) 
         
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=4);          

    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img_copy  

detect_faces = faces_detected_img(f_cascade, colored_img)
plt.imshow(detect_faces)


#note time after detection 
t2 = time.time() 
#calculate time difference 
dt1 = t2 - t1 
#print the time difference

print("Haar: "+str(dt1))