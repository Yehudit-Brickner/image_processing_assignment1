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
import cv2


def NormalizeData(data):
    """
    return the array normalized to numbers between 0 and 1
    :param data:
    :return:
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def nothing(x): # this function does nothing it is used for creating the trackbar
    pass

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    try:
        if rep == 1: # check if the image is grayscale or rgb
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_gray = NormalizeData((img_gray))
        else:
            img_color = cv2.imread(img_path)
            img_color = NormalizeData((img_color))

        # i am using an inner try and except to catch errors that will stop the main from continuing to run
        try:
            # create a separate window named 'image' for trackbar
            cv2.namedWindow('image')
            # create trackbar in 'image' window with name 'gamma''
            cv2.createTrackbar('gamma', 'image', 0, 200,nothing)

            while (1):

                gamma= cv2.getTrackbarPos('gamma', 'image')/100.0
                # the gamma value will br the value of thr traker divided by 100.0 so that tha gamma value will actuallly be netwen 0 and 1

                if rep == 1:
                    img_gray_copy = img_gray ** gamma # create a copy of the image array raised to the power of the gamma
                    cv2.imshow('image', img_gray_copy)
                else:
                    img_color_copy = img_color ** gamma # create a copy of the image array raised to the power of the gamma
                    cv2.imshow('image',img_color_copy)
                cv2.waitKey(1) # wait 1 milli sec before going back to the beginning of the while
            cv2.destroyAllWindows() # destroys the window
        except:
            print("an error was caught")
    except:
        print ("Could not open/read file:", img_path)





def main():
   gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
