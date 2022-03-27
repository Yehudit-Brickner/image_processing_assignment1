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
from ex1_utils import LOAD_GRAY_SCALE
import numpy as np
import matplotlib.pyplot as plt
import cv2


def NormalizeData(data):
    """
    return the array normalized to numbers between 0 and 1
    :param data:
    :return:
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def nothing():
    pass

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    # create a separate window named 'image' for trackbar
    cv2.namedWindow('image')
    # create trackbar in 'image' window with name 'gamma''
    cv2.createTrackbar('gamma', 'image', 0, 200,nothing)
    while (1):
        gamma= cv2.getTrackbarPos('gamma', 'image')/100.0

        if rep == 1:
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_gray = (img_gray - np.min(img_gray)) / (np.max(img_gray) - np.min(img_gray))
            img_gray_copy = img_gray ** gamma
            cv2.imshow('image', img_gray_copy)
        else:
            img_color = cv2.imread(img_path)
            img_color = (img_color - np.min(img_color)) / (np.max(img_color) - np.min(img_color))
            img_color_copy = img_color ** gamma
            cv2.imshow('image',img_color_copy)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    # pass






def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
