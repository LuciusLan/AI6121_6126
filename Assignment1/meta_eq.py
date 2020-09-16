import cv2
import numpy as np
from collections import OrderedDict

filename = 'sample05.jpeg'
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

#Create the histogram of histogram. Preserving the original key (Y value)
hist_hist = {}
for _, (k, v) in enumerate(hist.items()):
    if v in hist_hist.keys():
        hist_hist[v]['v'] += 1
    else:
        hist_hist[v] = {'v': 1, 'k': k}
hist_hist = OrderedDict(sorted(hist_hist.items()))

#Calculate the frequency in histogram of histogram
freq_hh = hist_hist.copy()
for k in freq_hh.keys():
    freq_hh[k]['v'] /= 256
hh_item = list(freq_hh.keys())

#Define the histogram value transform 
def transform_h(f):
    sum_freq = 0
    pos = hh_item.index(f)
    #Sum the frequency of F less or equal to input F
    for i in range(pos+1):
        sum_freq += freq_hh[hh_item[i]]['v']
    return 255 * sum_freq
h2 = hist.copy()
#Perform the histogram transform
for _, (k ,v) in enumerate(h2.items()):
    h2[k] = round(transform_h(v))

#Form the equalized histogram. The total number for frequency calculation is total number of frequencies instead of pixels
freq_hist = h2.copy()
for k in freq_hist.keys():
    freq_hist[k] /= np.array(list(h2.values())).sum()
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
cv2.imwrite(f'MEqualized_{filename}', output_img)

print()