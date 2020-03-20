import pandas as pd
import cv2
import os
import glob
import numpy as np
from sklearn.utils import shuffle

def convert_data(pic_path):
    img = cv2.imread(pic_path, cv2.IMREAD_UNCHANGED)
    img_gray = 255 - img[:, :, 3]
    img_gray = img_gray.reshape(1, 784)
    pixel_string = ""
    for i in range(0, len(img_gray)):
        pixel_string = pixel_string + str(img_gray[i])
    pixel_string = pixel_string.replace("[", "")
    pixel_string = pixel_string.replace("]", "")
    return pixel_string

path = "C:\\Users\\asuss\\Desktop\\Cyrillic" # Cyrillic dizinine giden yolun path'i
directory = os.chdir(path)
df = pd.DataFrame(columns=['Pixel', 'Letter'])
i = 0
for x in os.walk(path):
     if i == 0:
         i = i+1
         continue;
     else:
         print(x[0])
         for png in glob.glob(x[0] + "\\" + "*.png"):
            pixel_string = convert_data(png)
            letter = png.split('\\')[5]
            df = df.append(pd.Series([pixel_string,letter], index=df.columns), ignore_index=True)

df = shuffle(df)
df.to_csv (r'C:\Users\asuss\Desktop\letters.csv', index = None, header=True, encoding='utf-8')