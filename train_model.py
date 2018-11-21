from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt 
from net import Net

img_width, img_height = 32, 32
no_of_channels = 3
train_data_dir = 'data/train/' 
validation_data_dir = 'data/test/' 
epochs = 80
batch_size = 32

model = Net.build(width = img_width, height = img_height, depth = no_of_channels)
rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer=rms,
              metrics=['accuracy'])

train_data_model = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=False)

# this is the augmentation configuration used for testing, only rescaling
test_datagen = ImageDataGenerator(featurewise_center=True, 
                        featurewise_std_normalization=True,
                        rescale=1. / 255,
                        shear_range=0.1,
                        zoom_range=0.1,
                        rotation_range=5,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        horizontal_flip=False)

train_generator = train_data_model.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# fit the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / batch_size, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / batch_size)  

# evaluate  validation dataset and save weights in a file
model.evaluate_generator(validation_generator,validation_generator.samples/batch_size,)
model.save_weights('trained_weights.h5') 
