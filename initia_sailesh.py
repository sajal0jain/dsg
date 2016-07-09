# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 01:56:14 2016

@author: Sailesh Mohanty
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import glob, os
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import matplotlib as mpl
import matplotlib.pyplot as plt
from subprocess import check_output
from skimage.transform import resize
from skimage import io
from skimage.color import rgb2gray
from skimage import data, img_as_float
from skimage import exposure
from skimage import color
from scipy import ndimage as ndi
from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from matplotlib import cm
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage.feature import daisy
from skimage import transform as tf
from skimage.feature import CENSURE



mpl.rcParams['font.size'] = 8

def colorize(image, hue, saturation=1):
    """ Add color of the given hue to an RGB image.

    By default, set the saturation to 1 so that the colors pop!
    """
    hsv = color.rgb2hsv(image)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return color.hsv2rgb(hsv)

def plot_img_and_hist(img, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    img = img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    ax_img.set_adjustable('box-forced')

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


train = pd.read_csv('/media/sajal/47DE7E8B160B07C9/dsg/id_train.csv')
train_id = train['Id'].astype(str)
train['label'] = train['label'].astype(str)
train['length'] = 0
train['breadth'] = 0
'''
for i in range(len(train_id)):
    img_name = "DSG/train_resize/"+train_id.iloc[i]+".jpg"
    img = plt.imread(img_name)
    dim = np.shape(img)
    train['length'][i] = dim[0]
    train['breadth'][i] = dim[1]
    '''
print("1")
img_name = io.imread("DSG/train_resize/"+train_id.iloc[5000]+".jpg")
img_name_gs = rgb2gray(img_name)

plt.imshow(img_name)
plt.imshow(img_name_gs)

# Load an example image
img = img_name_gs

# Contrast stretching
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

# Equalization
img_eq = exposure.equalize_hist(img)

# Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2,4), dtype=np.object)
axes[0,0] = fig.add_subplot(2, 4, 1)
for i in range(1,4):
    axes[0,i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
for i in range(0,4):
    axes[1,i] = fig.add_subplot(2, 4, 5+i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
ax_img.set_title('Histogram equalization')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
ax_img.set_title('Adaptive equalization')

ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 5))

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()

hue_rotations = np.linspace(0, 1, 6)

fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)

for ax, hue in zip(axes.flat, hue_rotations):
    # Turn down th e saturation to give it that vintage look.
    tinted_image = colorize(img_name, hue, saturation=0.3)
    ax.imshow(tinted_image, vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_adjustable('box-forced')
fig.tight_layout()

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(img_name_gs)
edges2 = feature.canny(img_name_gs, sigma=3.2)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)
ax1.imshow(img_name_gs, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)
ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)
ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)
fig.tight_layout()
plt.show()

edge_roberts = roberts(img_name_gs)
edge_sobel = sobel(img_name_gs)

fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

ax0.imshow(edge_roberts, cmap=plt.cm.gray)
ax0.set_title('Roberts Edge Detection')
ax0.axis('off')

ax1.imshow(edge_sobel, cmap=plt.cm.gray)
ax1.set_title('Sobel Edge Detection')
ax1.axis('off')
plt.tight_layout()


edge_sobel = sobel(img_name_gs)
edge_scharr = scharr(img_name_gs)
edge_prewitt = prewitt(img_name_gs)

diff_scharr_prewitt = edge_scharr - edge_prewitt
diff_scharr_sobel = edge_scharr - edge_sobel
max_diff = np.max(np.maximum(diff_scharr_prewitt, diff_scharr_sobel))

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

ax0.imshow(img_name_gs, cmap=plt.cm.gray)
ax0.set_title('Original image')
ax0.axis('off')

ax1.imshow(edge_scharr, cmap=plt.cm.gray)
ax1.set_title('Scharr Edge Detection')
ax1.axis('off')

ax2.imshow(diff_scharr_prewitt, cmap=plt.cm.gray, vmax=max_diff)
ax2.set_title('Scharr - Prewitt')
ax2.axis('off')

ax3.imshow(diff_scharr_sobel, cmap=plt.cm.gray, vmax=max_diff)
ax3.set_title('Scharr - Sobel')
ax3.axis('off')

plt.tight_layout()
plt.show()





