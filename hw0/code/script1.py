import numpy as np
import matplotlib
from scipy.misc import toimage	
from alignChannels import alignChannels
from PIL import Image

# Problem 1: Image Alignment

# 1. Load images (all 3 channels)
red = np.load('../data/red.npy')
green = np.load('../data/green.npy')
blue = np.load('../data/blue.npy')


# print (green.shape)
# scipy.misc.imsave('red.jpg',red)
# toimage(green).show()
# a=red[0,:]
# print(len(red[0,:]))
# print(a[20])
# rgbArray=np.zeros((810,943,3),'uint8')
# rgbArray[..., 0] = red
# rgbArray[..., 1] = green
# rgbArray[..., 2] = blue
# img = Image.fromarray(rgbArray)
# img.save('rgb.jpg')


# 2. Find best alignment
rgbResult = alignChannels(red, green, blue)
rgbResult.save('rgbResult.jpg')

# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)
