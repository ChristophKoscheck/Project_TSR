import os
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage import io, transform
from skimage.transform import resize
from sklearn.preprocessing import normalize
from canvasapi import Canvas
from skimage.color import rgb2gray
from keras.callbacks import TensorBoard
import datetime



##Inforamtionen zur Codeversion und der Modellversion
#Aenderungen hier eingeben:
modell_nummer = 5

# Auflösung
resolution = 64
batch_size = 32
epochs = 15


## Code to ensure an GPU is used when avaiblabe to process the Data (Works for AMD, Intel and, Nvidia GPUs)
# Ensure TensorFlow-DirectML is being used (Native Linux or Windows Subystem for Linux)
print("TensorFlow version:", tf.__version__)

# Check for available GPUs to train the model ## Note if more than 1 GPU is available the first GPU will be used
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set TensorFlow to use the first GPU if available
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        # Catch and display the error if the GPU setting fails
        print(e)
else:
    print("No GPU available, using CPU instead.")

#Load Data

def load_data(data_dir, target_size=(resolution, resolution)):  # You can adjust the target size as needed
    images = []
    labels = []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for f in os.listdir(class_path):
                if f.endswith('.ppm'):
                    image = io.imread(os.path.join(class_path, f))
                    image_resized = resize(image, target_size, anti_aliasing=True)
                    images.append(image_resized)
                    labels.append(int(class_dir))
    return np.array(images), np.array(labels)


# Load training and testing datasets
ROOT_PATH = "/home/paul/TSR"                                	# Use in Windows Subsystem for Linux
# ROOT_PATH = ""                                                # Use this line if you are running the code in the same directory as the dataset
train_data_dir = os.path.join(ROOT_PATH, "Training")
test_data_dir = os.path.join(ROOT_PATH, "Testing")
train_images, train_labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)


# Create an instance of ImageDataGenerator for augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,           # Normalize the image
    rotation_range=20,        # Random rotation between -20 to +20 degrees
    width_shift_range=0.2,    # Random horizontal shift
    height_shift_range=0.2,   # Random vertical shift
    shear_range=0.2,          # Shear transformations
    zoom_range=0.2,           # Random zoom
    horizontal_flip=False,    # Traffic signs should not be flipped horizontally
    fill_mode='nearest'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)  # Only rescale for testing

# Load images from directories and apply transformations
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(resolution, resolution),  # Adjust depending on your model input
    batch_size=batch_size,
    class_mode='categorical'  # Use 'categorical' for multi-class labels
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(resolution, resolution),
    batch_size=batch_size,
    class_mode='categorical'
)


# Berechne die Summe aller Trainingsbilder
total_train_images = len(train_images)

# Log the training process for visualization in TensorBoard
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Convolutional Neural Network

def conv_net(train_images_dims, num_classes, batch_size=batch_size, filter_size = 16, pool_size=(2, 2)):
    # Preprocess image dimensions
        if len(train_images_dims) == 3:  # Assuming channel last format
            input_shape = (train_images_dims[0], train_images_dims[1], train_images_dims[2])
        elif len(train_images_dims) == 4:
            input_shape = (train_images_dims[1], train_images_dims[2], train_images_dims[3])
        else:
            raise ValueError("Invalid train image dimensions")

        model = tf.keras.Sequential([
        # Data Augmentation Layers
        #tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='reflect', interpolation='bilinear'),
        #tf.keras.layers.RandomRotation(factor=(0, 1), fill_mode='reflect', interpolation='bilinear'),
        #tf.keras.layers.RandomBrightness(factor=(0.5, 0.7), value_range=(0, 255)),  # Adjusted
        #tf.keras.layers.RandomContrast(factor=(0, 1)),  # Adjusted
        #tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2, fill_mode='reflect', interpolation='bilinear'),
        # Convolutional Layers
        tf.keras.layers.Conv2D(64, (7, 7), activation='relu', input_shape=train_images_dims, padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        # Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model



# Create the model

monitored = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

model_regulation = conv_net(train_images[0].shape, num_classes=61, batch_size=batch_size)

model_regulation.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_regulation.summary()


# Model training
history = model_regulation.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.n // test_generator.batch_size, 
    callbacks=[monitored, tensorboard_callback]
)


# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.get_current_fig_manager().set_window_title('Loss History for Model {}'.format(modell_nummer))
plt.title('Loss History for Model {}'.format(modell_nummer))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Get training and test accuracy histories
training_accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']

# Visualize accuracy history
plt.plot(epoch_count, training_accuracy, 'r--')
plt.plot(epoch_count, test_accuracy, 'b-')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.get_current_fig_manager().set_window_title('Accuracy History for Model {}'.format(modell_nummer))
plt.title('Accuracy History for Model {}'.format(modell_nummer))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Evaluate the model
test_loss, test_accuracy = model_regulation.evaluate(test_images, test_labels, verbose=2)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# Save the model
savepath = "/home/paul/TSR"
model_regulation.save(os.path.join(savepath, 'Test_Model_{}.h5'.format(modell_nummer)))