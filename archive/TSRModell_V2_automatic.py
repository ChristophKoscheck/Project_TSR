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
import pandas as pd
import logging

save_dir = '/home/paul/TSR'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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


# Setting up the parameters
epochs = np.array([8, 16, 32])
batch_sizes = np.array([8, 16, 32])
resolutions = np.array([32, 64, 128])

# Load and preprocess data
def load_data(data_dir, target_size=(64, 64)):
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

# Function to create and compile the model
def create_model(input_shape, num_classes):
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
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Root paths
ROOT_PATH = "/home/paul/TSR"
train_data_dir = os.path.join(ROOT_PATH, "Training")
test_data_dir = os.path.join(ROOT_PATH, "Testing")

# DataFrame to store results
results_df = pd.DataFrame(columns=['Epochs', 'Batch Size', 'Resolution', 'Loss', 'Accuracy', 'Val Loss', 'Val Accuracy', 'Test Loss', 'Test Accuracy'])

# Run the model with different configurations
# Setup basic configuration for logging
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

for epoch in epochs:
    for batch_size in batch_sizes:
        for resolution in resolutions:
            try:
                # Load data with current resolution
                train_images, train_labels = load_data(train_data_dir, target_size=(resolution, resolution))
                test_images, test_labels = load_data(test_data_dir, target_size=(resolution, resolution))
                
                # Normalize the images
                train_images = train_images / 255.0
                test_images = test_images / 255.0

                # Get input shape and number of classes
                input_shape = train_images.shape[1:]
                num_classes = len(np.unique(train_labels))
                
                # Create and train model
                model = create_model(input_shape, num_classes)
                history = model.fit(train_images, train_labels, batch_size=batch_size,
                                    epochs=epoch, validation_data=(test_images, test_labels), verbose=1)
                
                # Evaluate the model on test data
                test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

                # Extract final epoch data
                final_epoch = history.history
                last_loss = final_epoch['loss'][-1]
                last_accuracy = final_epoch['accuracy'][-1]
                last_val_loss = final_epoch['val_loss'][-1]
                last_val_accuracy = final_epoch['val_accuracy'][-1]
                
                # Prepare new row for DataFrame
                new_row = pd.DataFrame({
                             'Epochs': [epoch], 
                             'Batch Size': [batch_size], 
                             'Resolution': [resolution],
                             'Loss': [last_loss], 
                             'Accuracy': [last_accuracy],
                             'Val Loss': [last_val_loss], 
                             'Val Accuracy': [last_val_accuracy],
                             'Test Loss': [test_loss], 
                             'Test Accuracy': [test_accuracy]
                            })

                results_df = pd.concat([results_df, new_row], ignore_index=True)
            except Exception as e:
                logging.error(f"Error processing configuration with Epochs={epoch}, Batch Size={batch_size}, Resolution={resolution}: {str(e)}")

results_df.to_csv(os.path.join(save_dir, 'model_results.csv'), index=False)
logging.info("All configurations processed successfully and results saved.")
