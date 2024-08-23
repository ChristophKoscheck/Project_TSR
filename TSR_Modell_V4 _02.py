import os
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage import transform
from skimage.transform import resize
from sklearn.preprocessing import normalize
from keras.callbacks import TensorBoard
import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping
from skimage.io import imread
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import itertools
from PIL import Image
import seaborn as sns
import io

##Inforamtionen zur Codeversion und der Modellversion
#Aenderungen hier eingeben:
modell_nummer = 4

# Auflösung
resolution = 64
batch_size = 48
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
                    image = imread(os.path.join(class_path, f))
                    image_resized = resize(image, target_size, anti_aliasing=True)
                    images.append(image_resized)
                    labels.append(int(class_dir))
    return np.array(images), np.array(labels)



# Load training and testing datasets
ROOT_PATH = "/home/paul/TSR"                                	# Use in Windows Subsystem for Linux
# ROOT_PATH = ""                                                # Use this line if you are running the code in the same directory as the dataset
#train_data_dir = os.path.join(ROOT_PATH, "Training")
#test_data_dir = os.path.join(ROOT_PATH, "Testing")
train_data_dir = os.path.join(ROOT_PATH, "TSR_Data_Train")
test_data_dir = os.path.join(ROOT_PATH, "TSR_Data_Test")
train_images, train_labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)




# Berechne die Summe aller Trainingsbilder
total_train_images = len(train_images)

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convolutional Neural Network

def conv_net(train_images_dims, num_classes, batch_size=batch_size, filter_size = 32,):
    # Preprocess image dimensions
        if len(train_images_dims) == 3:  # Assuming channel last format
            input_shape = (train_images_dims[0], train_images_dims[1], train_images_dims[2])
        elif len(train_images_dims) == 4:
            input_shape = (train_images_dims[1], train_images_dims[2], train_images_dims[3])
        else:
            raise ValueError("Invalid train image dimensions")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D((64),(7,7),activation='relu',input_shape= train_images_dims),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D((64),(5,5),activation='relu',input_shape= train_images_dims),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D((128),(3,3),activation='relu',input_shape= train_images_dims),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

# Create the model
# Path to save the best model
save_path = os.path.join("/home/paul/TSR", f'Test_Model_{modell_nummer}.h5')

#monitored = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    save_path, 
    monitor='val_loss', 
    save_best_only=True, 
    verbose=1
)
model_regulation = conv_net(train_images[0].shape, len(np.unique(train_labels)))

#model_regulation.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_regulation.summary()

history = model_regulation.fit(
    train_images, train_labels, 
    validation_data=(test_images, test_labels),
    steps_per_epoch=(len(train_images) / batch_size), 
    epochs=epochs,
    batch_size= batch_size,
    callbacks =[early_stopping, model_checkpoint, TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))])

# Save the model
model_regulation.save(os.path.join(save_path, 'Test_Model_{}.h5'.format(modell_nummer)))

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


predictions = model_regulation.predict(train_images)
predicted_classes = np.argmax(predictions, axis=1)

cm = confusion_matrix(train_labels, predicted_classes)
class_names = [str(i) for i in np.unique(train_labels)]
               
def plot_confusion_matrix(cm, class_names, normalize=False):
    """
    Plots the confusion matrix.
    
    Parameters:
        cm (array, shape = [n, n]): confusion matrix
        class_names (array, shape = [n]): list of class names
        normalize (bool): whether to normalize the matrix
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps)  # Add epsilon to avoid division by zero
        cm = np.nan_to_num(cm)  # Replace any NaN values with 0
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

log_dir = "logs/fit/"  # Replace with your log directory

# Plot the confusion matrix
cm_fig = plot_confusion_matrix(cm, class_names)
cm_image = io.BytesIO()
plt.savefig(cm_image, format='png')
plt.close(cm_fig)
cm_image.seek(0)

# Convert to tensor
image_tensor = tf.image.decode_png(cm_image.getvalue(), channels=4)
image_tensor = tf.expand_dims(image_tensor, 0)  # Add batch dimension

# Log to TensorBoard
with tf.summary.create_file_writer(log_dir).as_default():
    tf.summary.image("Confusion Matrix", image_tensor, step=0)

# Log the classification report
report = classification_report(train_labels, predicted_classes, target_names=class_names)
with tf.summary.create_file_writer(log_dir).as_default():
    tf.summary.text("Classification Report", report, step=0)

# Log histograms for model layers
for layer in model_regulation.layers:
    for weight in layer.weights:
        tf.summary.histogram(weight.name, weight, step=0)
