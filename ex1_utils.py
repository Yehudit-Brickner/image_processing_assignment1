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

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 328601018



def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def Normalize255(data):
    return (data*255)

def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # print("reading img")
    if representation == 1:
        img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img_gray = NormalizeData(img_gray)
        return img_gray
    else:
        img_color = cv2.imread(filename)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
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
    img = imReadAndConvert(filename, representation)
    if representation == 1:
        # By default, matplotlib use a colormap which maps intensities to colors. so without the cmap='gray' the bic whold be blueish yellow
        plt.imshow(img, cmap='gray')
        plt.show()
    else:
        plt.imshow(img)
        plt.show()
    # pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    imgRGB = NormalizeData(imgRGB)
    r = imgRGB[:, :, 0]
    g = imgRGB[:, :, 1]
    b = imgRGB[:, :, 2]
    # y = r * 0.299 + g * 0.587 + b * 0.114
    # i = r * 0.595879 + g * -0.274133 + b * -0.321746
    # q = r * 0.211205 + g * -0.523083 + b * 0.311878
    y = r*0.299 +g*0.587 +b*0.114
    i = r*0.596 +g*-0.275 +b*-0.321
    q = r*0.212 +g*-0.523 +b*0.311
    yiq_img = imgRGB
    yiq_img[:, :, 0] = y
    yiq_img[:, :, 1] = i
    yiq_img[:, :, 2] = q
    yiq_img=NormalizeData(yiq_img)
    # plt.imshow(yiq_img)
    # plt.show()
    return yiq_img
    # pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    imgYIQ=NormalizeData(imgYIQ)
    y = imgYIQ[:, :, 0]
    i = imgYIQ[:, :, 1]
    q = imgYIQ[:, :, 2]
    r = y + i * 0.956 + q * 0.619
    g = y + i * -0.272 + q * -0.647
    b = y + i * -1.106 + q * 1.703
    rgb_img = imgYIQ
    rgb_img[:, :, 0] = r
    rgb_img[:, :, 1] = g
    rgb_img[:, :, 2] = b
    rgb_img=NormalizeData(rgb_img)
    # plt.imshow(rgb_img)
    # plt.show()
    return rgb_img
    # pass


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


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
