# image_processing_assignment1


this project was done in pycharm
<br> python version 3.10.0

there are 3 files of code:
- ex1_utils.py
- gamma.py
- ex1_main.py

In this assignment we had to write 7 functions:
- imReadAndConvert = read an image and return the numpy array of the image.
- imDisplay = display the image.
- transformRGB2YIQ = transform the image from RGB color space to YIQ color space and return the new array.
- transformYIQ2RGB = transform the image from YIQ color space to RGB color space and return the new array.
- hsitogramEqualize = take an image and spread out the histogram so that there is more contrast in the image and return the new image the original histogran array and the changed histogram array.
- quantizeImage = takes the image and chnages it so that it only has x colors, returning a list of the images array, and a list of the mse.
- gammaDisplay = dispaly an image and a trackbar that will show the gamma transfmation.



<br>
<br>


<b>imReadAndConvert</b>
<br> this function is surrounded by a try and except in case we get the wrong file name for the image.
<br> we will read the image.
<br> if the image is in grayscale we will normalize the array and return it.
<br> if the image is in color we will change it from bgr to rgb  normalize the array and return it.
<br>
<br>
<b>imDisplay</b>
<br> this function is surrounded by a try and except in case we get the wrong file name for the image.
<br> we will call the imReadAndConvert function to get the array of the image.
<br> if the image is in grayscale we will show it using "cmap =gray" so that we see the image in grayscale and not a map of itenseties.
<br> if the image is in color we will show it.
<br>
<br>
<b>transformRGB2YIQ</b>
<br> this function is surrounded by a try and except in case we get an empty array.
<br> we will take the array of the image and split it into 3 arrays r g b.
<br> we will create 3 new arrays y i q that are scalar multiples of rgb, with the correct scalars.
<br> we will combine them together again into a 3d array
<br> we will normalize the arrays.
<br> and return the new array
<br>
<br>
<b>transformYIQ2RGB</b>
<br> this function is surrounded by a try and except in case we get an empty array.
<br> we will take the array of the image and split it into 3 arrays y i q.
<br> we will create 3 new arrays r g b that are scalar multiples of yiq, with the correct scalars.
<br> we will normalize the arrays.
<br> we will combine them together again into a 3d array
<br> and return the new array
<br>
<br>
<b>hsitogramEqualize</b>
<br>
<br>
<b>quantizeImage</b>
<br>
<br>
<b>gammaDisplay</b>
<br> this function is surrounded by a try and except in case we get the wrong file name for the image.
<br> we will read the image and normalize it.
<br> we will surround the next part in a try and excempt, because when we close the pop up window it throws an error.
<br> we will create a new window and a sliding track bar with the values 0-200
<br> we will create a while loop that will always stay true.
<br> we will get the gamma value from the trackbar and divide by 100.0.
<br> we will show the image raised to the power of gamma.


