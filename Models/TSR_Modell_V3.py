##############################################################################
#                                  Modell V3                                 #
##############################################################################
#  Autoren   : Christoph Koscheck, Paul Smidt                                #
#  Vorlesung : Künstliche Intelligenz, Verkehrszeichenerkennung              #
#  Datum     : 23. August 2024                                               #
##############################################################################
# ----------------------------------------------------------------------------
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from skimage.io import imread
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import io
# ----------------------------------------------------------------------------

##Inforamtionen zur Codeversion und der Modellversion
modell_nummer = 3

# Parameter
resolution = 64
batch_size = 32
epochs = 15

# Code, der sicherstellt, dass eine GPU verwendet wird, wenn sie verfügbar ist, um die Daten zu verarbeiten (funktioniert für AMD, Intel und Nvidia GPUs)
print("TensorFlow version:", tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, using CPU instead.")

# Daten einlesen
def load_data(data_dir, target_size=(resolution, resolution)):  
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

# Lade die Trainings- und Testdaten
ROOT_PATH = ""                                                
train_data_dir = os.path.join(ROOT_PATH, "Training")
test_data_dir = os.path.join(ROOT_PATH, "Testing")
train_images, train_labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)

# Normalisiere die Bilder
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convolutional Neural Network
def conv_net(train_images_dims, num_classes, batch_size=batch_size):
         # Dimensionen der Trainingsbilder
        if len(train_images_dims) == 3:
            input_shape = (train_images_dims[0], train_images_dims[1], train_images_dims[2])
        elif len(train_images_dims) == 4:
            input_shape = (train_images_dims[1], train_images_dims[2], train_images_dims[3])
        else:
            raise ValueError("Invalid train image dimensions")

        # Modeldefinition
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        # Modelkompilierung
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

# Modelerstellung
monitored = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

model_regulation = conv_net(train_images[0].shape, len(np.unique(train_labels)))

model_regulation.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_regulation.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model_regulation = conv_net(train_images[0].shape, len(np.unique(train_labels)))

history = model_regulation.fit(
    train_images, train_labels, 
    validation_data=(test_images, test_labels),
    steps_per_epoch=(len(train_images) / batch_size), 
    epochs=epochs,
    batch_size= batch_size,
    callbacks =[early_stopping, TensorBoard(log_dir="logs/fit/" + '{}'.format(modell_nummer))])

# Speichern des Modells
save_path = os.path.join("Models")
model_regulation.save(os.path.join(save_path, 'Test_Model_{}.h5'.format(modell_nummer)))

# Trainings- und Testverlust
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Epochenanzahl
epoch_count = range(1, len(training_loss) + 1)

# Trainings- und Testgenauigkeit
training_accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']

# Modellevaluierung
test_loss, test_accuracy = model_regulation.evaluate(test_images, test_labels, verbose=2)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


predictions = model_regulation.predict(train_images)
predicted_classes = np.argmax(predictions, axis=1)

cm = confusion_matrix(train_labels, predicted_classes)
class_names = [str(i) for i in np.unique(train_labels)]
               
def plot_confusion_matrix(cm, class_names, normalize=False):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps)
        cm = np.nan_to_num(cm) 
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

if not os.path.exists("logs/fit/"):
    os.makedirs("logs/fit/")
log_dir = "logs/fit/"

# Konfusionsmatrix als Bild
cm_fig = plot_confusion_matrix(cm, class_names)
cm_image = io.BytesIO()
plt.savefig(cm_image, format='png')
plt.close(cm_fig)
cm_image.seek(0)

# Tensorkonvertierung
image_tensor = tf.image.decode_png(cm_image.getvalue(), channels=4)
image_tensor = tf.expand_dims(image_tensor, 0)  # Add batch dimension

# Tensorboard-Loggings
with tf.summary.create_file_writer(log_dir).as_default():
    tf.summary.image("Confusion Matrix", image_tensor, step=0)

# Klassifikationsbericht
report = classification_report(train_labels, predicted_classes, target_names=class_names)
with tf.summary.create_file_writer(log_dir).as_default():
    tf.summary.text("Classification Report", report, step=0)

# Loggen der Gewichtungen
for layer in model_regulation.layers:
    for weight in layer.weights:
        tf.summary.histogram(weight.name, weight, step=0)