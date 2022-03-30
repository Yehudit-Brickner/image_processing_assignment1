"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 328601018



def NormalizeData(data):
    """
    return the array normalized to numbers between 0 and 1
    :param data:
    :return:
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def Normalize255(data):
    """
    we will run the function to normalize the array - just to make sure that it is in between  and 1
    and than we will * by 255 so that we can get values in the range of 0 and 255
    :param data:
    :return:
    """
    data=NormalizeData(data)
    return (data*255)

def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    """
    we will see if the image is grayscale or color
    """
    try:
        if representation == 1: # check if grayscale or rgb
            img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # read the image
            img_gray = NormalizeData(img_gray)
            return img_gray
        else:
            img_color = cv2.imread(filename) # read the image
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB) # change the image from BGR to RGB
            img_color = NormalizeData(img_color)
            return img_color
    except:
        print("wrong file type")


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    try:
        img = imReadAndConvert(filename, representation) #get the array of the image
        if representation == 1: # check if grayscale or rgb
            # By default, matplotlib use a colormap which maps intensities to colors. so without the cmap='gray' the bic whold be blueish yellow
            plt.imshow(img, cmap='gray')
            plt.show()
        else:
            plt.imshow(img)
            plt.show()
    except:
        print("caught on exception")


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    try:
        imgRGB = NormalizeData(imgRGB) # normalize the array
        # split into the R,G,B channels
        r = imgRGB[:, :, 0]
        g = imgRGB[:, :, 1]
        b = imgRGB[:, :, 2]
        # create the new channels
        y = r * 0.299 + g * 0.587 + b * 0.114
        i = r * 0.596 + g * -0.275 + b * -0.321
        q = r * 0.212 + g * -0.523 + b * 0.311

        yiq_img = imgRGB.copy() # create an array that is the same as the original so that it will be the exact same size
        # put the Y,I,Q channels into the yiq_img
        yiq_img[:, :, 0] = y
        yiq_img[:, :, 1] = i
        yiq_img[:, :, 2] = q
        yiq_img=NormalizeData(yiq_img) # normalize the array

        return yiq_img
    except:
        print("was mot given an array")


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    try:
        imgYIQ=NormalizeData(imgYIQ) # normalize the array
        # split into the Y,I,Q channels
        y = imgYIQ[:, :, 0]
        i = imgYIQ[:, :, 1]
        q = imgYIQ[:, :, 2]
        # create the new channels
        r = y + i * 0.956 + q * 0.619
        g = y + i * -0.272 + q * -0.647
        b = y + i * -1.106 + q * 1.703
        # normalize the new channels
        r = NormalizeData(r)
        g = NormalizeData(g)
        b = NormalizeData(b)

        rgb_img = imgYIQ.copy() # create an array that is the same as the original so that it will be the exact same size
        # put the R,G,B channels into the rgb_img
        rgb_img[:, :, 0] = r
        rgb_img[:, :, 1] = g
        rgb_img[:, :, 2] = b

        return rgb_img
    except:
        print("was mot given an array")


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    try:
        shape = imgOrig.shape # gets the shape of the array Ex. (300,600)=grayscale or(300,600,3)=rgb
        countnum = len(shape) # gets the amount of values in the shape
        pixel_num = shape[0] * shape[1] # the amount of pixels in the picture

        if (countnum == 2): #checks if the pic is grayscale or rgb
            # print("gray scale image")
            imgOrig_norm255 = Normalize255(imgOrig) # noramalize the picture to values between 0 and 255
            histOrg, edges = np.histogram(imgOrig_norm255, 256, [0, 256]) # create a histogram of the image with 256 bins
            cumsumOrig = np.cumsum(histOrg) # create the cumsum of the image

            # create a look up table
            lut = cumsumOrig * 255 / pixel_num
            lut = np.ceil(lut)

            imEq = imgOrig_norm255.copy() # create an array that is the same as the original so that it will be the exact same size
            for num in range(256):
                imEq[imgOrig_norm255 == num] = int(lut[num])

            histEQ, edges = np.histogram(imEq, 256, [0, 256]) # create a new histogram of the new image
            imEq = NormalizeData(imEq) # normalize the image
            return (imEq, histOrg, histEQ)

        else:
            yiq = transformRGB2YIQ(imgOrig) # transform the image to YIQ so that we can do the histogram equalization only on the y channel
            y = yiq[:, :, 0]
            y255 = Normalize255(y) # normalize
            histOrg, edges = np.histogram(y255, 256, [0, 256]) #create a histogram of the y channel with 256 bins
            cumsumOrig = np.cumsum(histOrg) # create the cumsum of the y channel
            #create lut
            lut = cumsumOrig * 255 / pixel_num
            lut = np.ceil(lut)

            y_new = y255.copy() # create an array that is the same as the original so that it will be the exact same size
            y255=y255.astype(int)
            # change the values of the y channel by the lut
            for num in range(256):
                y_new[y255 == num] = int(lut[num])

            histEQ, edges = np.histogram(y_new, 256, [0, 256]) # create a new histogram of the new y channel

            y_new = NormalizeData(y_new) # normalize the y channel
            yiq[:, :, 0] = y_new # put in the new y channel
            imEq = transformYIQ2RGB(yiq) # transform the image back to rgb
            return (imEq, histOrg, histEQ)
    except:
        print("was not given an array")


def upper(x, arr, num_pixel):
    y_val = arr[x]
    return x*y_val / num_pixel

def lower(x, arr, num_pixel):
    y_val = arr[x]
    return y_val / num_pixel

def find_new_z(z, q):
    new_z =z.copy()
    for i in range(len(q) - 1):
        new_z[i + 1] = (q[i] + q[i + 1]) / 2
    return new_z

def find_new_q(z, q, hist, num_pixel):
    new_q = q.copy()
    j = 0
    for zs in range(len(z) - 1):
        res1 = 0
        res2 = 0
        if (z[zs] == z[zs + 1]):
            new_q[j] = z[zs]
            j = j + 1
        else:
            for num in range(z[zs], z[zs + 1]):
                res1 += upper(num, hist, num_pixel)
                res2 += lower(num, hist, num_pixel)
            # print( "res1 and 2",res1, res2)
            if res1 == 0 or res2 == 0:
                new_q[j] = z[zs]
                j = j + 1
            else:
                qi = res1 / res2
                new_q[j] = qi
                j = j + 1
    # print("new q",new_q)
    # new_q=q.copy()
    # for zs in range(len(z) - 1):
    #     # print(z[zs], z[zs+1])
    #     part_hist = hist[z[zs]:z[zs+1]]
    #     idx = range(len(part_hist))
    #     weightedMean = (part_hist * idx).sum() / np.sum(part_hist)
    #     new_q[zs]=weightedMean+z[zs]
    # # new_q = new_q.astype(int)
    # print("new q", new_q)
    return new_q

def newpic(imOrig255, nQuant, z, q):
    new_img = imOrig255.copy()
    for num in range(nQuant):
        new_img[imOrig255>z[num]] =q[num]
    return new_img

def calc_mse(imgOrig, newImg):
    mse=((imgOrig- newImg) ** 2).mean(axis=None)
    #mse=math.sqrt(mse)
    return mse

def find_orig_z(pixel_num, nQuant, cumsum, z):
    new_z = z.copy()
    bound1 = pixel_num / nQuant
    bound = bound1
    i = 1
    for x in range(255):
        if (cumsum[x] >= bound):
            new_z[i] = x
            i = i + 1
            bound = bound + bound1
    while (i < len(z) - 1):
        new_z[i] = 255
        i = i + 1
    new_z[0] = 0
    new_z[len(z) - 1] = 255
    new_z = new_z.astype(int)
    # new_z=z.copy()
    # for i in range(nQuant+1):
    #     new_z[i]=(i)/nQuant*255
    # new_z=new_z.astype(int)
    # print("first z",new_z)
    return new_z


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    try:
        z = np.zeros(nQuant + 1)  # bourders for the areas nquant+1 numbers in the list
        q = np.zeros(nQuant)  # the inteseties we wnat nquant
        list_of_pic = []
        list_mse = []
        shape = imOrig.shape
        pixel_num = (float) (shape[0] * shape[1])
        count = 0

        if len(shape) == 2: # if the picture is grayscale
            imOrig255 = Normalize255(imOrig) # normalize
            hist, edges = np.histogram(imOrig255, 256, [0, 256])  # create a histogram of the images with 256 bins
            cumsum = np.cumsum(hist) # create the cumsum of the image

            z = find_orig_z(pixel_num, nQuant, cumsum, z) # find original z by pixels weight

            for k in range(0, nIter): # make a loop that will run nIter times unless told to stop
                if count > 10:
                    return list_of_pic, list_mse # if count is bigger than 10. then 10 times the diference between the current and last mse was smaller than 0.001, that means that we have converged
                # print("iteration ", k)
                q = find_new_q(z, q, hist, pixel_num)   # find new q
                z = find_new_z(z, q) # find new z

                new_img = newpic(imOrig255, nQuant, z, q) # create the new img using z and q
                mse_new = calc_mse(imOrig255,new_img) # calculate the mse

                if k > 0: # if we are on the first iteration we cant compare to anything so we will only compare from the 2nd iteration and on.
                    if list_mse[-1] - mse_new < 0.001: # check the difference between the last and current mse
                        count = count+1
                    if (list_mse[-1]+1< mse_new): # if the last mse+1 is bigger than the current mse we will stop
                        print("mse got bigger")
                        return list_of_pic, list_mse
                list_mse.append(mse_new) # add the mse to the mse_list
                new_img = NormalizeData(new_img) # noramlize the image
                list_of_pic.append(new_img) # add the image to the img_list
            return list_of_pic, list_mse
        else: # image is rgb
            yiq = transformRGB2YIQ(imOrig) # transform the image to YIQ
            y = yiq[:, :, 0]
            y255 = Normalize255(y) # normalize
            hist, edges = np.histogram(y255, 256, [0, 256]) # create a histogram of the y channel
            cumsum = np.cumsum(hist) # create the cumsum of the y channel
            z = find_orig_z(pixel_num, nQuant, cumsum, z)  # find the original z

            for k in range(0, nIter): # make a loop that will run nIter times unless told to stop
                if count > 10:
                    return list_of_pic, list_mse # if count is bigger than 10. then 10 times the diference between the current and last mse was smaller than 0.001, that means that we have converged
                q = find_new_q(z, q, hist, pixel_num) # find new q
                z = find_new_z(z, q) # find new z

                new_img = newpic(y255, nQuant, z, q)  # create new y channel
                mse_new = calc_mse(y255,new_img) # calculate the mse

                if(k>0): # if we are on the first iteration we cant compare to anything so we will only compare from the 2nd iteration and on.
                    if list_mse[-1] - mse_new < 0.001: # check the difference between the last and current mse
                        count=count+1
                    if (list_mse[-1] +1< mse_new): # if the last mse+1 is bigger than the current mse we will stop
                        print("mse got bigger")
                        return list_of_pic, list_mse
                list_mse.append(mse_new) # add the mse to the mse_list
                y_new = NormalizeData(new_img) # normalize the y channel
                yiq[:, :, 0] =y_new # put in the new y channel
                imnew = transformYIQ2RGB(yiq) # transform the image back to rgb
                imnew = NormalizeData(imnew) # normalize the image
                list_of_pic.append(imnew) # add the image to the image_list

            return list_of_pic, list_mse
    except:
        print("caught an error")


