import os, cv2
import GUI as gui
import ColorDetector as cd
from PIL import Image
import numpy as np
from pylab import rcParams

rcParams['figure.figsize'] = 20, 10


def getLabel(id):
    return ['Civic', 'Corolla', 'Mehran', 'Other'][id]


def loadData():
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D

    input_shape = (128, 128, 3)
    num_classes = 4
    model = Sequential()

    # Feature Extraction
    model.add(Convolution2D(6, 5, 5, input_shape=input_shape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(120, 5, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # TEST
    model.load_weights('model_weights.h5')
    return model


def imgReader(fn, model):
    testimg_data_list = []
    test_img = cv2.imread(fn, True)
    test_img_resize = cv2.resize(test_img, (128, 128))
    testimg_data_list.append(test_img_resize)
    testimg_data = np.array(testimg_data_list)
    testimg_data = testimg_data.astype('float32')
    testimg_data = testimg_data / 255
    testimg_data.shape()
    testimg_data_list.clear()

    results = model.predict_classes(testimg_data)

    cv2.imshow('window-name', test_img)
    cv2.waitKey(0)

    return getLabel(results[0])


if __name__ == "__main__":
    model = loadData()
    app = gui.Root()
    app.mainloop()  # this will run until it closes
    if app.fileName is not None and app.fileName != '':
        fileName = app.fileName
        print("car :", imgReader(fileName, model))
        image = Image.open(fileName)
        final_colors = cd.process_image(image)
        highest = 0
        for strength in final_colors.items():
            split = int(str(strength).split('.')[2].split(", ")[1])
            if split > highest:
                highest = split

        for color, strength in final_colors.items():
            # print("x", color.__name__,strength)
            split = int(str(strength).split('.')[0])
            if split == highest:
                print(color.__name__, strength)
    else:
        print("No File!")
