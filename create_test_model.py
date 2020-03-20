from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import traceback
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras import backend as K

my_dict = {"ii": "И",
           "A": "А",
           "B": "Б",
           "V": "В",
           "G": "Г",
           "D": "Д",
           "YE": "Е",
           "YO": "Ё",
           "J": "Ж",
           "Z": "З",
           "Y": "Й",
           "K": "К",
           "L": "Л",
           "M": "М",
           "N": "Н",
           "O": "О",
           "P": "П",
           "R": "Р",
           "S": "С",
           "T": "Т",
           "U": "У",
           "F": "Ф",
           "H": "Х",
           "TS": "Ц",
           "CH": "Ч",
           "SH": "Ш",
           "SHCH": "Щ",
           "Tviordiy znak": "Ъ",
           "I": "Ы",
           "Myagkiy znak": "Ь",
           "E": "Э",
           "YU": "Ю",
           "YA": "Я"
           }

letter_to_num = {"И": 0,
                 "А": 1,
                 "Б": 2,
                 "В": 3,
                 "Г": 4,
                 "Д": 5,
                 "Е": 6,
                 "Ё": 7,
                 "Ж": 8,
                 "З": 9,
                 "Й": 10,
                 "К": 11,
                 "Л": 12,
                 "М": 13,
                 "Н": 14,
                 "О": 15,
                 "П": 16,
                 "Р": 17,
                 "С": 18,
                 "Т": 19,
                 "У": 20,
                 "Ф": 21,
                 "Х": 22,
                 "Ц": 23,
                 "Ч": 24,
                 "Ш": 25,
                 "Щ": 26,
                 "Ъ": 27,
                 "Ы": 28,
                 "Ь": 29,
                 "Э": 30,
                 "Ю": 31,
                 "Я": 32
             }

batch_size = 128
num_classes = 33
epochs = 20

img_rows, img_cols = 28, 28

# Read data from csv
data = pd.read_csv(r"C:\Users\asuss\Desktop\letters.csv")

# Print the head of the data
# print(data.head())

# Split the data into train and test
print(data.shape)


pixels = []
letters = []
number = data.shape[0]
for i in range(0,number):
    try:
        pixel_string = data['Pixel'][i]
        pixels.append(np.fromstring(pixel_string, dtype=int, sep=' ').reshape(28, 28))
        letters.append(data['Letter'][i])
    except Exception as e:
        pass

print(np.array(pixels).shape)
print(np.asarray(letters).shape)
pixels = np.array(pixels)
letters = np.asarray(letters)

X_train = pixels[:12000]
y_train = letters[:12000]
X_test = pixels[12001:15233]
y_test = letters[12001:15233]

for i in range(0,len(y_train)):
    y_train[i] = letter_to_num[my_dict[y_train[i]]]
for i in range(0,len(y_test)):
    y_test[i] = letter_to_num[my_dict[y_test[i]]]

# I looked my keras.json file and I found out that "image_data_format": "channels_last" so I reshaped my data according
# to the keras documentation
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train[0])
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(y_train)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('russian_letters.h5')



