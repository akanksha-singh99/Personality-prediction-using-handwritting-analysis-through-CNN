# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 14:48:12 2020

@author: Akanksha singh
"""

# Independent variables here are pixels. So traditional method won't work.

# Building the CNN

# Squential is used to initialize ANN
# Convolution,Pooling,Flatenning have their regular uses
# Dense is used to add fully connected layers to ANN
''' Part A'''
'''A.a'''
#Importing the necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
#from tensorflow.keras.models import load_model


'''A. b'''

# For augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,#Scaling pixels values b/w 0 to 1
        #rotation_range=45,
        horizontal_flip=True,
        shear_range=0.2,# Random shear
        zoom_range=0.2,# Random zoom
        )# Flipping of images horizontaly is allowed.

# Ensuring the size of test data b/w 0 to 1 
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        shuffle=True,
        target_size=(600, 600),#64 and 64 specified above
        batch_size=3,# Number of images in a batch
        class_mode='sparse')#Since we have the multiple calss

augmented_images = [training_set[0][0][0] for i in range(5)]

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 3, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
plotImages(augmented_images)

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(600, 600),
        batch_size=32,
        class_mode='sparse')


'''Part A.c'''
sample_training_images, _ = next(training_set)#The next function returns a batch from the dataset
    
plotImages(sample_training_images[:5])

''' Part B'''

'''B. a'''

model = Sequential()


# filters = 32 i.e. the number of feature detectors/kernel/filters are 32 (Use 64 if you're on GPU)
# kernel_size = 3, 3 i.e. the size of these kernels is 3x3
# input_shape = (3, size, size) is the format for theano backend. Since we're using tensorflow backend we'll use
#               (size, size, 3). 3 is the number of channels. RGB channels so 3 for a colored image. (256, 256 size on a GPU)
# activation = 'relu' is Rectfied Linear Units method for removing the linearity. (Rectifier used as activation on neuron)
model.add(Convolution2D(4, 3, 3, input_shape=(600, 600, 3), activation='relu'))

# pool_size = (2,2) ie. the size of the pool that will check for max values is 2x2
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Convolution2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))


model.add(Flatten())

# Here we have 32 feature detectors, each with many pixels  then after all the steps of convolution, pooling and 
# flattening we'll still get a high number of nodes consisting vector, that'll act as input layer for fully connected layer.
# output_dim here represents the number of nodes in the hidden layer. These should be b/w the input and output nodes.
# Generally it is observed that choosing around 100 helps. It's not too compute intensive nor too small.

model.add(Dense(128, activation = 'relu'))

# This one is for the output layer. Here output_dim = 1 since it's either a Diseased or a Healthy.
# Since outcome is binary so we'll use 'sigmoid' as the activation function
model.add(Dense(5, activation = 'sigmoid'))


# Compiling CNN
# Optimizer = 'adam' means the stochastic gradient decent
# loss = 'binary_crossentropy' is the loss/cost function. If it was more than 2 outcomes then would've chosen categorical_crossentropy.
# Here outcome is binary ie. either Diseased or Healthy.
# metrics=['accuracy'] is performance metric over which it will be evaulated

model.summary()

'''B. b'''
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# Preparing Data
# Here we have used augmentation of images as we were not having quite large number of images.


history = model.fit_generator(
        training_set,
        steps_per_epoch=86,#364 images of Diseased and 388 images of Healthy
        epochs=10,# After which we'll be updating weights
        validation_data=test_set,#evaluating over the test_set
            )#Images in test set. 60 Diseased and 60 Healthy



'''B. c'''

#model.save('my_model.h5')
#model = load_model('my_model.h5')
# For individual testing

''' Part C'''
    # Accuracy can be seen right after the epochs have been completed.
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=10
epochs_range = range(epochs)

plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


"""   
''' Part D '''
path = 'dataset/single_prediction/diseased_or_healthy_2.jpg'
test_image = image.load_img(path, target_size=(128,128)) #Dimensions already specified
# Now since the data that we need to have should be a 3D, 3rd being the 3 for RGB
test_image = image.img_to_array(test_image)
# The data for the prediction needs to be 4D. 4th being the batch. Even though it is single but predict function
# expects the data to be in form of batches
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
print(result)
# training_set.class_indices can be used to check the category encoding
animal = ""
if result==0:
    animal = "Diseased"
else:
    animal = "Healthy"
    
from PIL import Image, ImageDraw, ImageFont
image = Image.open(path)
font_type = ImageFont.truetype('arial.ttf',30)
draw = ImageDraw.Draw(image)
draw.text(xy=(0,0),text=animal,fill=(0,0,0),font=font_type)
image.show()
"""    
""" To improve accuracy, one more convulation layer can be added with either same or different parameters.
    Add it after the max pooling and before the flattening.
    Next layer won't be having input_shape parameter. Keras will automatically accept the pool of previous layer."""
