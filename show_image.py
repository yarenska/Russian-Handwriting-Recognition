import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def rgb2gray(rgb):
     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread('C:\\Users\\sgunes\\Downloads\\Cyrillic\\Д\\5a087d4c45e5b.png')
gray = rgb2gray(img)
plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.show()