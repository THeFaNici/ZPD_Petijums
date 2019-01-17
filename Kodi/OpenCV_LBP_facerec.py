# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:25:09 2019

@author: Nils
"""
#Attēli:Attels1 Attels2 Attels3 Attels4 Attels5

#Tiek import modeļi
import cv2
import matplotlib.pyplot as plt
import time

#Sāk laiku skaitīt
l1 = time.time()  

def sejas_atrast_img(lbp_face_cascade, colored_img, scaleFactor = 1.2):
    img_copy = colored_img.copy()
          
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY) 
         
    faces = lbp_face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=4);          

    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img_copy 

#Ielādē uztrenetu kaskadesanas klasifikatoru 
lbp_face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')  
 
#ielādēt attēlu
test2 = cv2.imread('Attels5.jpg')
 
#call our function to detect faces 
detect_faces = sejas_atrast_img(lbp_face_cascade, test2)
 
#convert image to RGB and show image
plt.imshow(detect_faces)


#beidz laiku skaitīt
l2 = time.time() 
#laika diference
ld1 = l2 - l1 
#izvada laika diferenci
print("LBP: "+str(ld1))