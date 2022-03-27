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

    if representation == 1: # check if grayscale or rgb
        img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # read the image
        img_gray = NormalizeData(img_gray)
        return img_gray
    else:
        img_color = cv2.imread(filename) # read the image
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB) # change the image from BGR to RGB
        img_color = NormalizeData(img_color)
        return img_color
    # pass


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation) #get the array of the image
    if representation == 1: # check if grayscale or rgb
        # By default, matplotlib use a colormap which maps intensities to colors. so without the cmap='gray' the bic whold be blueish yellow
        plt.imshow(img, cmap='gray')
        plt.show()
    else:
        plt.imshow(img)
        plt.show()



def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    imgRGB = NormalizeData(imgRGB) # normalize the array
    # split into the R,G,B channels
    r = imgRGB[:, :, 0]
    g = imgRGB[:, :, 1]
    b = imgRGB[:, :, 2]
    # create the new channels
    y = r * 0.299 + g * 0.587 + b * 0.114
    i = r * 0.596 + g * -0.275 + b * -0.321
    q = r * 0.212 + g * -0.523 + b * 0.311

    yiq_img = imgRGB # create an array that is the same as the original so that it will be the exact same size
    # put the Y,I,Q channels into the yiq_img
    yiq_img[:, :, 0] = y
    yiq_img[:, :, 1] = i
    yiq_img[:, :, 2] = q
    yiq_img=NormalizeData(yiq_img) # normalize the array

    return yiq_img



def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
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

    rgb_img = imgYIQ # create an array that is the same as the original so that it will be the exact same size
    # put the R,G,B channels into the rgb_img
    rgb_img[:, :, 0] = r
    rgb_img[:, :, 1] = g
    rgb_img[:, :, 2] = b

    return rgb_img
    


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    shape = imgOrig.shape
    countnum = len(shape)
    pixel_num = shape[0] * shape[1]

    if (countnum == 2):
        print("gray scale image")
        imgOrig_norm255 = Normalize255(imgOrig)
        histOrg, edges = np.histogram(imgOrig_norm255, 256, [0, 256])
        cumsumOrig = np.cumsum(histOrg)

        lut = cumsumOrig * 255 / pixel_num
        lut = np.ceil(lut)

        imEq = imgOrig_norm255
        for row in range(0, shape[0]):
            for col in range(0, shape[1]):
                x = imgOrig_norm255[row][col]
                x = int(x)
                y = lut[x]
                imEq[row][col] = y

        histEQ, edges = np.histogram(imEq, 256, [0, 256])
        imEq = NormalizeData(imEq)
        return (imEq, histOrg, histEQ)

    else:
        print("color image")
        yiq = transformRGB2YIQ(imgOrig)
        y = yiq[:, :, 0]
        y255 = Normalize255(y)
        print(y255)
        histOrg, edges = np.histogram(y255, 256, [0, 256])
        cumsumOrig = np.cumsum(histOrg)

        lut = cumsumOrig * 255 / pixel_num
        lut = np.ceil(lut)

        y_new = y255
        for row in range(0, shape[0]):
            for col in range(0, shape[1]):
                x = y255[row][col]
                x = int(x)
                num = lut[x]
                y_new[row][col] = num

        histEQ, edges = np.histogram(y_new, 256, [0, 256])

        y_new = NormalizeData(y_new)
        yiq[:, :, 0] = y_new
        imEq = transformYIQ2RGB(yiq)
        imEq = NormalizeData(imEq)
        return (imEq, histOrg, histEQ)

    # pass

# function to help with the quantization

def upperold(x, arr,num_pixel):
    y_val = arr[x]
    return y_val/num_pixel * x


def lowerold(x, arr,num_pixel):
    y_val = arr[x]
    return y_val/num_pixel


def find_new_z(z, q):
    #     print("z old=",z)
    #     print("q=",q)
    for i in range(len(q) - 1):
        z[i + 1] = (q[i] + q[i + 1]) / 2
    return z


def find_new_q(z, q, hist,num_pixel):
    ans = 1
    new_q = q
    j = 0;
    for zs in range(len(z) - 1):
        res1 = 0
        res2 = 0
        for num in range(z[zs], z[zs + 1]):
            res1 += upperold(num, hist,num_pixel)
            res2 +=lowerold(num, hist,num_pixel)

        if res2 != 0:
            qi = int(res1 / res2)
            q[j] = qi
            j = j + 1
        else:
            ans = 0
            #             print("res 2 = 0")
            #             q[j]=int((z[zs]+z[zs+1])/2)
            new_q = q
            return new_q, ans
    new_q = np.ceil(new_q)
    return new_q, ans


def newpic(imOrig255, hist, nQuant, z, q):
    shape = imOrig255.shape
    new_img = imOrig255
    for row in range(0, shape[0]):
        for col in range(0, shape[1]):
            x = imOrig255[row][col]
            x = int(x)
            for num in range(nQuant):
                if x >= z[num] and x <= z[num + 1]:
                    new_img[row][col] = q[num]
    return new_img


def calc_mse(nQuant, z, q, hist,pixel_num):
    mse = 0
    for i in range(nQuant):
        mse1 = 0
        for j in range(z[i], z[i + 1]):
            mse1 += (q[i] - j) * (q[i] - j) * hist[j]/pixel_num
        mse += mse1
    return mse


def find_orig_z(pixel_num, nQuant, cumsum, z):
    bound1 = pixel_num / nQuant
    bound = bound1
    i = 1
    for x in range(256):
        if (cumsum[x] >= bound):
            z[i] = x
            i = i + 1
            bound = bound + bound1

    z = z.astype(int)
    return z








def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    z = np.zeros(nQuant + 1)  # bourders for the areas nquant+1 numbers in the list
    q = np.zeros(nQuant)  # the inteseties we wnat nquant
    list_of_pic = []
    list_mse = []
    shape = imOrig.shape
    pixel_num = shape[0] * shape[1]

    if len(shape) == 2:
        imOrig255 = Normalize255(imOrig)
        hist, edges = np.histogram(imOrig255, 256, [0, 256])

        cumsum = np.cumsum(hist)

        z = find_orig_z(pixel_num, nQuant, cumsum, z)
        q, ans = find_new_q(z, q, hist,pixel_num)
        print("first q",q)
        print("first z",z)

        new_img = newpic(imOrig255, hist, nQuant, z, q)
        list_of_pic.append(new_img)
        mse1 = calc_mse(nQuant, z, q, hist,pixel_num)
        list_mse.append(mse1)

        mse_old = mse1
        mse_new = 0
        old_img = new_img
        q_old = q
        for k in range(1, nIter):
            z = find_new_z(z, q)
            q, ans = find_new_q(z, q, hist,pixel_num)
            if np.array_equal(q, q_old) and ans == 0:
                break

            new_img = newpic(imOrig255, hist, nQuant, z, q)
            mse_new = calc_mse(nQuant, z, q, hist,pixel_num)
            if abs(mse_old - mse_new) < 0.001:
                break
            if (mse_new > mse_old):
                new_img = old_img
                print("mse got bigger")
                break
            list_mse.append(mse_new)
            list_of_pic.append(new_img)
            mse_old = mse_new
            old_img = new_img
            q_old = q
        # plt.plot(list_mse)
        # plt.show()

        # plt.imshow(new_img, cmap='gray')
        # plt.show()

        print(list_mse)
        print("z=",z)
        print("q=",q)
        return list_of_pic, list_mse
    else:
        yiq = transformRGB2YIQ(imOrig)
        y = yiq[:, :, 0]
        y255 = Normalize255(y)
        # print(y255)
        hist, edges = np.histogram(y255, 256, [0, 256])
        cumsum = np.cumsum(hist)
        z = find_orig_z(pixel_num, nQuant, cumsum, z)
        q, ans = find_new_q(z, q, hist,pixel_num)
        print("first q", q)
        print("first z", z)

        new_img = newpic(y255, hist, nQuant, z, q)
        y_new = NormalizeData(new_img)
        yiq[:, :, 0] = y_new
        imnew = transformYIQ2RGB(yiq)
        imnew = NormalizeData(imnew)
        list_of_pic.append(imnew)
        mse1 = calc_mse(nQuant, z, q, hist,pixel_num)
        list_mse.append(mse1)

        mse_old = mse1
        mse_new = 0
        old_img = new_img
        q_old = q
        for k in range(1, nIter):
            z = find_new_z(z, q)
            q, ans = find_new_q(z, q, hist,pixel_num)
            if np.array_equal(q, q_old) and ans == 0:
                break

            new_img = newpic(y255, hist, nQuant, z, q)
            mse_new = calc_mse(nQuant, z, q, hist,pixel_num)
            if abs(mse_old - mse_new) < 0.001:
                break
            if (mse_new > mse_old):
                new_img = old_img
                print("mse got bigger")
                break
            list_mse.append(mse_new)
            y_new = NormalizeData(new_img)
            yiq[:, :, 0] = y_new
            imnew = transformYIQ2RGB(yiq)
            imnew = NormalizeData(imnew)
            list_of_pic.append(imnew)
            mse_old = mse_new
            old_img = new_img
            q_old = q
        # plt.plot(list_mse)
        # plt.show()
        print(list_mse)
        print("z=",z)
        print("q=",q)
        return list_of_pic, list_mse





