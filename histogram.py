import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

inputf = sys.argv[1]
outputf = sys.argv[2]

# reading the greyscale image through OpenCV
input_1 = cv2.imread(inputf,0)

print("Input1: Gray level values in pixels")
for i in range (input_1.shape[0]): #traverses through height of the image
    for j in range (input_1.shape[1]): #traverses through width of the image
        print(input_1[i][j], end="\t")
    print("\t")

# calculating the cumulative distribution function (cdf)
def cdf_calc(image):
    # generating the histogram by flattening the image through NumPy
    hist,bins = np.histogram(image.flatten(),256,[0,256])

    # Cumulative Distribution function of the Histogram
    cdf = hist.cumsum()

    # Calculating the normalized cumulative histogram
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    return cdf, cdf_normalized

# cdf and normalized cdf for the input image    
cdf, cdf_normalized = cdf_calc(input_1)

# plotting the histogram of the input image
plt.subplot(121)
plt.title("Input Image 1")
plt.plot(cdf_normalized, color = 'b')
plt.hist(input_1.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper right')

# normalized cumulative histogram (cdf) is used as a mapping function
# utilized the mapping arrays of NumPy for this purpose
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

# output image gray values are calculated putting the input image inside the new cdf
output_1 = cdf[input_1]
# the new image is written out as result_1.png
cv2.imwrite(outputf, output_1)

# calculating the cdf and cdf_normalized values for plotting the histogram of output image
cdf1, cdf_normalized1 = cdf_calc(output_1)

print("Output1: Gray level values in pixels")
for i in range (output_1.shape[0]): #traverses through height of the image
    for j in range (output_1.shape[1]): #traverses through width of the image
        print(output_1[i][j], end="\t")
    print("\t")

# plotting the histogram of the output image
plt.subplot(122)
plt.title("Output Image 1")
plt.plot(cdf_normalized1, color = 'b')
plt.hist(output_1.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper right')

# printing the final graphs
plt.show()

