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


##Inforamtionen zur Codeversion und der Modellversion
#Aenderungen hier eingeben:
modell_nummer = 2

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

# Display the first image in the training dataset
plt.imshow(train_images[0])
plt.get_current_fig_manager().set_window_title('Train Image Modell {}'.format(modell_nummer))
plt.title('Train Image Modell {}'.format(modell_nummer))
plt.show()
print("Image shape:", train_images[0].shape)
print("Label:", train_labels[0])

# Display the first image in the testing dataset
plt.imshow(test_images[0])
plt.get_current_fig_manager().set_window_title('Test Image Model {}'.format(modell_nummer))
plt.title('Test Image Model {}'.format(modell_nummer))
plt.show()
print("Image shape:", test_images[0].shape)
print("Label:", test_labels[0])

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convolutional Neural Network

def conv_net(train_images_dims, num_classes, batch_size=batch_size):
    # Preprocess image dimensions
        if len(train_images_dims) == 3:  # Assuming channel last format
            input_shape = (train_images_dims[0], train_images_dims[1], train_images_dims[2])
        elif len(train_images_dims) == 4:
            input_shape = (train_images_dims[1], train_images_dims[2], train_images_dims[3])
        else:
            raise ValueError("Invalid train image dimensions")

        # Define the model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

# Create the model

monitored = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

model_regulation = conv_net(train_images[0].shape, len(np.unique(train_labels)))

model_regulation.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_regulation.summary()

history = model_regulation.fit(train_images, train_labels, validation_data=(test_images, test_labels),steps_per_epoch=(len(train_images) / batch_size), epochs=epochs, callbacks=[monitored])

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