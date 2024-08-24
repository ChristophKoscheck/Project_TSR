import os
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage import io, transform
from skimage.transform import resize


#
#    Load Data
#

def load_data(data_dir, target_size=(32, 32)):
    images = []
    labels = []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for f in os.listdir(class_path):
                if f.endswith('.ppm'):
                    image = io.imread(os.path.join(class_path, f))
                    # Resize the image to the target size
                    image_resized = resize(image, target_size, mode='constant')
                    images.append(image_resized)
                    labels.append(int(class_dir))
    return np.array(images), np.array(labels)



# Load training and testing datasets.
ROOT_PATH = ""
train_data_dir = os.path.join(ROOT_PATH, "Training")
test_data_dir = os.path.join(ROOT_PATH, "Testing")

train_images, train_labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)

# Load Data
images, labels = load_data(train_data_dir)

#
#	Transform images to 32x32 pixels
#


# Training Data Generator

training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    zoom_range=0.2,
    width_shift_range=0.3,
    height_shift_range=0.2,
    shear_range=0.25,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode='nearest'
)
train_generator = training_datagen.flow(train_images,train_labels,batch_size=32)

# Model training
def conv_net(train_images_dims,num_of_classes,filter_size = 2, num_convolutions=64,num_strides=2):

     # pre process image dimensions
  if (len(train_images_dims) == 3):    # Channel Last
    train_images_dims = (train_images_dims[1],train_images_dims[2])
  elif (len(train_images_dims) == 4):
    train_images_dims = (train_images_dims[1],train_images_dims[2],train_images_dims[3])

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(num_convolutions, (filter_size, filter_size), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=num_strides))

    model.add(tf.keras.layers.Conv2D(num_convolutions * 2, (filter_size, filter_size), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=num_strides))

    model.add(tf.keras.layers.Conv2D(num_convolutions * 4, (filter_size, filter_size), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides= num_strides))

    model.add(tf.keras.layers.Conv2D(num_convolutions * 8, (filter_size, filter_size), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides= num_strides))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Dense(num_of_classes, activation='softmax'))

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    return model


monitor = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 8,restore_best_weights = True, min_delta = 0.01)

model_regularized = conv_net(train_images.shape,len(set(train_labels)),filter_size=2,num_convolutions=512)

model_regularized.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
model_regularized.summary()


history = model_regularized.fit(train_generator, validation_data=train_generator,steps_per_epoch=(len(train_images) / 32),epochs = 15,verbose=1,callbacks=[monitor])  # 32 = batch size

# Get training and test loss histories
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, validation_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# SAVE THE MODEL AS H5 File (e.g. to use in an app)
model_regularized.save('final_model.h5')

# First step is always to initialize all variables. 
# We don't care about the return value, though. It's None.
# Load validation data
val_data_dir = os.path.join(ROOT_PATH, "Validation")
val_images, val_labels = load_data(val_data_dir)
