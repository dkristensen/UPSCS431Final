from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from PIL import Image, ImageDraw
import numpy
import dotFinder
import os


# dimensions of our images.
img_width, img_height = 64, 64 # change to 128?
#
train_data_dir = 'Examples'
validation_data_dir = 'data/test'
nb_train_samples = int((len(os.listdir("Examples/sealion"))+len(os.listdir("Examples/background")))*0.8)
nb_validation_samples = int((len(os.listdir("data/test/sealion"))+len(os.listdir("data/test/background"))))
epochs = 80
batch_size = 50

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
a = Activation('relu')
model.add(a)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
a = Activation('relu')
model.add(a)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
a = Activation('relu')
model.add(a)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
a = Activation('relu')
model.add(a)
model.add(Dense(128))
a = Activation('relu')
model.add(a)
model.add(Dense(1))
a = Activation('sigmoid')
model.add(a)
#
# def fscore(y_true, y_pred):
#     return (2*y_true*y_pred / (y_true+y_pred) )
#
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# # this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale= 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.4,
    rotation_range = 90,
    height_shift_range = 8,
    width_shift_range = 8,
    horizontal_flip=True)
#
# # # this is the augmentation configuration we will use for testing:
# # # only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)
# # model.load_weights('model_values/firstFull.h5')
# # model.save('models/Binary_5C3H1O_80.h5')
# # model = load_model('models/firstModel.h5')

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

currentModel = 'Binary_3C3H2_64.h5'
model.save_weights('model_values/'+currentModel)
model.save('models/'+currentModel)
model = load_model('models/'+currentModel)
# print(model.metrics_names)
# quit()
# print(model.evaluate_generator(
#     validation_generator,
#     steps = nb_train_samples // batch_size))

def test_on_image(filename):
    model = load_model('models/'+currentModel)
    imageUsed = filename
    print(filename)
    a = dotFinder.getImageArray(imageUsed)
    b = dotFinder.imageArrayToTensor(a)
    c = numpy.asarray(b)
    classification = ["Male", "Female", "Juvenile", "Pup", "Subadult Male"]

    batchVal = model.predict(c, verbose = 1)
    im = Image.open(imageUsed)
    filename = filename.replace(".jpg","")
    filename = filename.replace("Train/","")
    dotFinder.drawOutputGrid(batchVal, im, filename)
    return
