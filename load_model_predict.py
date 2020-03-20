from keras.models import load_model
import cv2

def convert_data(pic_path):
    img = cv2.imread(pic_path, cv2.IMREAD_UNCHANGED)
    img_gray = 255 - img[:, :, 3]
    return img_gray

pic_path= 'C:\\Users\\asuss\\Desktop\\Cyrillic\\SH\\5a08661ed938b.png'
img_gray = convert_data(pic_path).reshape(1, 28, 28, 1)
print(img_gray.shape)
# Returns a compiled model identical to the previous one
model = load_model('russian_letters.h5')
# model.summary()
result = model.predict_classes(img_gray)
print(result)