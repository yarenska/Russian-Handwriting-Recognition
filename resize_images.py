from PIL import Image
import os
import glob

path = "C:\\Users\\asuss\\Desktop\\Cyrillic" # Cyrillic dizinine giden yolun path'i
directory = os.chdir(path)
i = 0
for x in os.walk(path):
    if i == 0:
        i = i+1
        continue;
    else:
        print(x[0])
        size = (28, 28)
        for png in glob.glob(x[0] + "\\" + "*.png"): # Dizinin içindeki her bir png dosyası
            im = Image.open(png)
            im = im.resize(size)
            im.save(png, dpi=(96, 96))