import matplotlib.pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'L:\Program Files\Tesseract-OCR\tesseract.exe'
import pyautogui
import time
from tkinter import *
import cv2
import numpy as np
import statistics


def sct(pos_origin,pos_end):   
    pos_diff = tuple(map(lambda i, j: i - j, pos_end, pos_origin))
    return pyautogui.screenshot(region=(pos_origin + pos_diff))

def process_rgb2bgr(im) :
    im = np.array(im)
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def process_red(im) :
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask = cv2.inRange(im, lower_red, upper_red)
    return cv2.bitwise_and(im, im, mask=mask)

def process_hsv2gray(im) :
    im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)



def masked(im) :
    height,width,depth = im.shape
    circle_img = np.zeros((height,width), np.uint8)
    cv2.circle(circle_img,(18,18),16,1,thickness=-1)
    return cv2.bitwise_and(im, im, mask=circle_img)

def pastemap(map,champ,x,y) :
    img1 = map
    img2 = champ

    img_2_shape = img2.shape
    roi = img1[y:y+img_2_shape[0],x:x+img_2_shape[1]]

    height,width,depth = img2.shape
    mask = np.zeros((height,width), np.uint8)
    cv2.circle(mask,(18,18),16,1,thickness=-1)

    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)

    img1[y:y+img_2_shape[0], x:x+img_2_shape[1]] = dst
    return img1

def matchim(champ_detected_masked,list_champ,line_counter) :
    result = cv2.matchTemplate(champ_detected_masked, list_champ[line_counter][0], cv2.TM_SQDIFF_NORMED)
    avg = statistics.mean(result[0]) #moyenne image value
    return 1 - avg



