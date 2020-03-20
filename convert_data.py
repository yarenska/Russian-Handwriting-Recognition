import cv2
import matplotlib.pyplot as plt

def convert_data(pic_path):
    img = cv2.imread(pic_path, cv2.IMREAD_UNCHANGED)
    img_gray = 255 - img[:, :, 3]
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    plt.show()
    print(img_gray)
    print(img_gray.shape)

pic_path= 'C:\Users\asuss\Desktop\Cyrillic\A\58bf1323ef917.png'.encode('unicode_escape')
convert_data(pic_path)