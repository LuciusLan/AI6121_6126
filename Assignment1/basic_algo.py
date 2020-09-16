import cv2
import numpy as np
from collections import OrderedDict

filename = 'sample01.jpg'
img = cv2.imread(filename)
#Convert RGB (BGR in opencv) to YCbCr (YCrCb in opencv)
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

x, y, z = img.shape
total_pix = x * y
#Get the Y channel (Grayscale)
img_y = img_ycrcb[:, :, 0]

#Flatten the grayscale img
flat = img_y.reshape(x*y)
tmp_hist = {}
#Get the histogram
for pixel in flat:
    if pixel in tmp_hist.keys():
        tmp_hist[pixel] += 1
    else:
        tmp_hist[pixel] = 1

#Order the histogram by Y value
hist = OrderedDict(sorted(tmp_hist.items()))
freq_hist = hist.copy()
for k in freq_hist.keys():
    freq_hist[k] /= total_pix
hist_item = list(freq_hist.keys())

#Define the pixel value transform function
def transform_y(pix):
    sum_freq = 0
    pos = hist_item.index(pix)
    #Sum the frequency of Y less or equal to pix's Y
    for i in range(pos+1):
        sum_freq += freq_hist[hist_item[i]]
    return 255 * sum_freq

#Perform the transform
for xx in range(x):
    for yy in range(y):
        img_y[xx, yy] = round(transform_y(img_y[xx, yy]))

output_img = img_ycrcb.copy()
#Replace the Y channel
output_img[:, :, 0] = img_y
output_img = cv2.cvtColor(output_img, cv2.COLOR_YCrCb2BGR)
cv2.imshow('Original image', img)
cv2.imshow('Equalized image', output_img)
cv2.waitKey(0)
cv2.imwrite(f'Equalized_{filename}', output_img)

print()