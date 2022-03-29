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


<br><br>
<br> all of the function below use a function to normalize an array to values between 0 and 1.
<br> we also used a function to take an array that was normalized and make the values be between 0 and 255.

<br> <br>
<b>imReadAndConvert</b>
<br> this function is surrounded by a try and except in case we get the wrong file name for the image.
<br> we will read the image.
<br> if the image is in grayscale we will normalize the array and return it.
<br> if the image is in color we will change it from bgr to rgb  normalize the array and return it.

<br><br>
<b>imDisplay</b>
<br> this function is surrounded by a try and except in case we get the wrong file name for the image.
<br> we will call the imReadAndConvert function to get the array of the image.
<br> if the image is in grayscale we will show it using "cmap =gray" so that we see the image in grayscale and not a map of itenseties.
<br> if the image is in color we will show it.

<br><br>
<b>transformRGB2YIQ</b>
<br> this function is surrounded by a try and except in case we get an empty array.
<br> we will take the array of the image and split it into 3 arrays r g b.
<br> we will create 3 new arrays y i q that are scalar multiples of rgb, with the correct scalars.
<br> we will combine them together again into a 3d array
<br> we will normalize the arrays.
<br> and return the new array

<br><br>
<b>transformYIQ2RGB</b>
<br> this function is surrounded by a try and except in case we get an empty array.
<br> we will take the array of the image and split it into 3 arrays y i q.
<br> we will create 3 new arrays r g b that are scalar multiples of yiq, with the correct scalars.
<br> we will normalize the arrays.
<br> we will combine them together again into a 3d array
<br> and return the new array

<br><br>
<b>hsitogramEqualize</b>
<br> this function is surrounded by a try and except in case we get an empty array.
<br> we will check if the image is grayscale or rgb
<br> if the image is gray scale we will do the following:
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will take the normalized array and normalize it so that it is between 0 and 255
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create a histogram of the intenseties of the image
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create the cumsum of the histogram
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create a look up table -  this will tell us how to change the picture
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will make a copy of the original array and change the values of it using the cumsum
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will make a histogram of the new image 
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will normalize the image
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will return the new image and both of the histograms
<br> if the image is rgb we will do the following:
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will transform the image to yiq
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will take the y channel and normalize it so that it is between 0 and 255
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create a histogram of the intenseties of the y channel of the image
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create the cumsum of the histogram
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create a look up table -  this will tell us how to change the picture
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will make a copy of the y channel array and change the values of it using the cumsum 
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will make a histogram of the new y channel
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will normalize the new y channel
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will put the new y channel in the original yiq image
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will transform the picture back to rgb
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will return the new image and both of the histograms

<br><br>
<b>quantizeImage</b>
<br> this function is surrounded by a try and except in case we get an empty array.
<br> we will create a few constants that we will use
<br> we wil ckeck if the image is grayscale or rgb
<br> if the picture is in grayscale we will do the following
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will take the normalized array and normalize it so that it is between 0 and 255
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create a histogram of the intenseties of the image
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create the cumsum of the histogram
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will use the cumsum to find the original z array (boarders)
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create a loop that will run nIter times or will stop if it converges.
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will check if count is bigger then 10 if so the image has converged and we can return the image_list and mse_list
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will find the q array based on the z array
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will find the new z array base on the q array
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create a new image using the z and q arrays 
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will calculate the mse
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will check and see if the last mse - the new mse is smaller than 0.001, if so we will increment count by 1.
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will check if the mse is smaller than the last mse+1
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if the new mse is bigger we will stop the loop and return the image_list and mse_list
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will add the mse to the mse_list
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will normalize the image
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will add the image to the image_list
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will return the image_list and the mse_list
<br> if the image is rgb we will do the following:
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will transform the image to yiq
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will take the y channel and normalize it so that it is between 0 and 255
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create a histogram of the intenseties of the y channel
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create the cumsum of the histogram
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will use the cumsum to find the original z array (boarders)
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create a loop that will run nIter times or will stop if it converges.
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will check if count is bigger then 10 if so, the image has converged and we can return the image_list and mse_list
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will find the q array based on the z array
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will find the new z array base on the q array
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will create a new y channel using the z and q  arrays 
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will calculate the mse
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will check and see if the last mse - the new mse is smaller than 0.001, if so we will increment count by 1.
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will check if the mse is smaller than the last mse+1
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; if the new mse is bigger we will stop the loop and return the image_list and mse_list
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will add the mse to the mse_list
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will normalize the y channel
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will put the new y channel in the original yiq image
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will transform the image back to rgb and normalize
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will add the image to the image_list
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; we will return the image_list and the mse_list
<br> this functin used 7 helping functions:
<br> a function to find the the original z
<br> upper and lower- functions for the numerater and denominater of the equation to find the new q.
<br> a function to find the new q array
<br> a function to find the new z
<br> a function to create the new array/image
<br> a function to calculate the mse
<br><br>
<b>gammaDisplay</b>
<br> this function is surrounded by a try and except in case we get the wrong file name for the image.
<br> we will read the image and normalize it.
<br> we will surround the next part in a try and excempt, because when we close the pop up window it throws an error.
<br> we will create a new window and a sliding track bar with the values 0-200
<br> we will create a while loop that will always stay true.
<br> we will get the gamma value from the trackbar and divide by 100.0.
<br> we will show the image raised to the power of gamma.




