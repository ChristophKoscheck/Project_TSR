##############################################################################
#                          Evaluierung des Modells                           #
##############################################################################
#  Autoren   : Christoph Koscheck, Paul Smidt                                #
#  Vorlesung : Künstliche Intelligenz, Verkehrszeichenerkennung              #
#  Datum     : 23. August 2024                                               #
##############################################################################
# ----------------------------------------------------------------------------
import numpy as np
import os
from skimage import io, transform
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import csv
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
# ----------------------------------------------------------------------------

# Initialisierung
resolution = 64
modell_nummer = 4 # nur Modell 4 ist als .h5 vortrainiert in der Abgabe enthalten, für alle anderen Modell müssen zunächst die entsprechenden Pythonskripte ausgeführt werden
model_path = f'Models/Test_Model_{modell_nummer}.h5'

# Laden des trainierten Modells
model = load_model(model_path)

# Pfad zum eignen Datensatz-Bildverzeichnis
image_dir = 'Validation'  
raw_image_dir = 'Validation' # Pfad zum Rohdatenverzeichnis

# Laden der Daten
def load_data(data_dir, target_size=(resolution, resolution)):
    images = []
    labels = []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for f in os.listdir(class_path):
                if f.endswith('.jpg'):
                    image = io.imread(os.path.join(class_path, f))
                    image_resized = transform.resize(image, target_size, anti_aliasing=True)
                    images.append(image_resized)
                    labels.append(int(class_dir))
    return np.array(images), np.array(labels)

images, labels = load_data(raw_image_dir)

# Laden der Bilder und Anpassen der Auflösung
def load_and_preprocess_image(image_path, target_size=(resolution, resolution)):
    image = io.imread(image_path)
    image = transform.resize(image, target_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Vorhersage der Verkehrsschilder
def predict_traffic_sign(image_path, model):
    image = load_and_preprocess_image(image_path)
    prediction_images = model.predict(image)
    predicted_class_index = np.argmax(prediction_images, axis=1)[0]
    return predicted_class_index

# Lesen der Labels und Vorhersagen
def read_labels_and_predict(image_dir, model, csv_file_path):
    label_map = {}
    with open(csv_file_path, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            filename, label = row
            label_map[filename] = int(label)

    predictions = []
    true_labels = []
    image_paths = []

    for filename, true_label in label_map.items():
        image_path = os.path.join(image_dir, filename)
        if os.path.exists(image_path):
            predicted_label = predict_traffic_sign(image_path, model)
            image_paths.append(image_path)
            predictions.append(predicted_label)
            true_labels.append(true_label)

    return true_labels, predictions, image_paths

# Klassenlabel, Label für Klasse 42, 43 und 55 nachträglich hinzugefügt
class_labels = {
    0 :'Warning for a bad road surface',
    1 :'Warning for a speed bump',
    2 :'Warning for a slippery road surface',
    3 :'Warning for a curve to the left',
    4 :'Warning for a curve to the right',
    5 :'Warning for a double curve, first left then right',                                                    # Merge Classes 5 & 6 later
    6 :'Warning for a double curve, first left then right',
    7 :'Watch out for children ahead',
    8 :'Watch out for  cyclists',
    9 :'Watch out for cattle on the road',
    10:'Watch out for roadwork ahead',
    11:'Traffic light ahead',
    12:'Watch out for railroad crossing with barriers ahead',
    13:'Watch out ahead for unknown danger',
    14:'Warning for a road narrowing',
    15:'Warning for a road narrowing on the left',
    16:'Warning for a road narrowing on the right',
    17:'Warning for side road on the right',
    18:'Warning for an uncontrolled crossroad',
    19:'Give way to all drivers',
    20:'Road narrowing, give way to oncoming drivers',
    21:'Stop and give way to all drivers',
    22:'Entry prohibited (road with one-way traffic)',
    23:'Cyclists prohibited',
    24:'Vehicles heavier than indicated prohibited',
    25:'Trucks prohibited',
    26:'Vehicles wider than indicated prohibited',
    27:'Vehicles higher than indicated prohibited',
    28:'Entry prohibited',
    29:'Turning left prohibited',
    30:'Turning right prohibited',
    31:'Overtaking prohibited',
    32:'Driving faster than indicated prohibited (speed limit)',
    33:'Mandatory shared path for pedestrians and cyclists',
    34:'Driving straight ahead mandatory',
    35:'Mandatory left',
    36:'Driving straight ahead or turning right mandatory',
    37:'Mandatory direction of the roundabout',
    38:'Mandatory path for cyclists',
    39:'Mandatory divided path for pedestrians and cyclists',
    40:'Parking prohibited',
    41:'Parking and stopping prohibited',
    42:'Parking forbidden from the 1st till 15th day of the month',
    43:'Parking forbidden from the 16th till last day of the month',
    44:'Road narrowing, oncoming drivers have to give way',
    45:'Parking is allowed',
    46:'parking for handicapped',
    47:'Parking for motor cars',
    48:'Parking for goods vehicles',
    49:'Parking for buses',
    50:'Parking only allowed on the sidewalk',
    51:'Begin of a residential area',
    52:'End of the residential area',
    53:'Road with one-way traffic',
    54:'Dead end street',
    55:'End of roadworks',
    56:'Crossing for pedestrians',
    57:'Crossing for cyclists',
    58:'Parking lot',
    59:'Information Sign : Speed bump',
    60:'End of the priority road',
    61:'Begin of a priority road'
}

# Pfad zur CSV-Datei mit den Labels des eigenen Datensatzes
label_csv_path = 'Validation/Labels.csv'  # Adjust this to the actual path

# Labels des eignen Datensatzes einlesen und Vorhersagen
true_labels, predictions, image_paths = read_labels_and_predict(image_dir, model, label_csv_path)

# Lables sortieren
labels = sorted(set(true_labels))

# Classification Report
class_names = [class_labels[label].strip() for label in labels] 
print(classification_report(true_labels, predictions, labels=labels))

# Vorbereiten der Daten für die Confusion Matrix
unique_labels = sorted(set(class_labels.keys())) 
class_names_matrix = [class_labels[label].strip() for label in unique_labels]

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(45, 45), cmap=plt.cm.Blues,
                                show_absolute=True, show_normed=True, class_names=class_names_matrix)
plt.rcParams.update({'font.size': 24})
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Wahre Klasse')
plt.title('Confusion Matrix')
if not os.path.exists("evaluation"):
    os.makedirs("evaluation")
plt.savefig('evaluation/confusion_matrix.png')

# Fehlerhafte Klassifikationen
def misclassified_images(targets, predicted_targets, image_paths, class_names):
    count = 0
    for i, target in enumerate(targets):
        if target != predicted_targets[i] and count < 10:
            # Create a directory for misclassifications if it doesn't exist
            if not os.path.exists("fehlklassifikationen"):
                os.makedirs("fehlklassifikationen")
            
            # Get the class names and images
            true_class = class_names[targets[i]]
            predicted_class = class_names[predicted_targets[i]]
            true_img = Image.open(image_paths[i])
            true_img = true_img.resize((resolution, resolution))
            img_true=Image.open("./Icons/TSR/{}.png".format(targets[i]))
            img_pred=Image.open("./Icons/TSR/{}.png".format(predicted_targets[i]))

            # Create a subplot with the true class, predicted class, and the true image
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img_true)
            axs[0].set_title("Wahre Klasse:\n" + true_class, fontsize=8)
            axs[0].axis("off")
            axs[1].imshow(img_pred)
            axs[1].set_title("Vorhergesagte Klasse:\n" + predicted_class, fontsize=8)
            axs[1].axis("off")
            axs[2].imshow(true_img)
            axs[2].set_title("Testbild", fontsize=8)
            axs[2].axis("off")
            
            # Save the subplot as an image in the misclassifications directory
            plt.savefig("fehlklassifikationen/misclassification_{}.png".format(count))
            plt.close(fig)
            
            count += 1

misclassified_images(true_labels, predictions, image_paths, class_names)

# Absoluter Pfad zum Bild für das die Feature Maps erstellt werden sollen
img_path = 'Validation/21/21_0.jpg'

# Erstellen des Modells, welches die Feature Maps für jedes Layer berechnet
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

# Laden des Bildes und auf Auflösung anpassen
img = load_img(img_path, target_size=(resolution, resolution, 3))

# Umwandeln des Bildes in ein Numpy Array
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
x /= 255.0

# Berechnen der Feature Maps
successive_feature_maps = visualization_model.predict(x)

# Namen der Layer des Modells
layer_names = [layer.name for layer in model.layers]

plt.figure(figsize=(20, 20))

image_files = []

# Code und Methode zur Visualisierung inspiriert von:
# https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c
for i, (layer_name, feature_map) in enumerate(zip(layer_names, successive_feature_maps)):
    if len(feature_map.shape) == 4: # Nur Convolutional / Pooling Layer
        n_features = feature_map.shape[-1]  # Anzahl der Features in der Feature Map
        size_y = feature_map.shape[1] # Größe der Feature Map in y-Richtung
        size_x = feature_map.shape[2] # Größe der Feature Map in x-Richtung
        
        display_grid = np.zeros((size_y, size_x * n_features)) # Leere Matrix für die Feature Maps

        for j in range(n_features):
            x = feature_map[0, :, :, j] 
            x -= x.mean()
            if x.std() != 0: 
                x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8') # Normalisierung der Feature Map

            display_grid[:, j * size_x: (j + 1) * size_x] = x # Hinzufügen der Feature Map zur Matrix

        plt.figure(figsize=(display_grid.shape[1] / 100, display_grid.shape[0] / 100)) 
        plt.imshow(display_grid, aspect='auto', cmap='viridis') 
        plt.title(layer_name)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        if not os.path.exists("feature_map"):
            os.makedirs("feature_map")

        filename = f"feature_map/feature_map_{i}_{layer_name}.png"
        plt.savefig(filename)
        image_files.append(filename)
        plt.close()

# Kombinieren der Feature Maps in einem Plot
n_layers = len(image_files)

plt.figure(figsize=(20, n_layers * 1.1))  # Anpassen der Größe des Plots an die Anzahl der Feature Maps

for i, filename in enumerate(image_files):
    img = mpimg.imread(filename)
    ax = plt.subplot(n_layers, 1, i + 1)
    ax.imshow(img) 
    ax.axis('off')  
    ax.set_title(layer_names[i])

plt.tight_layout()
plt.savefig('feature_map/combined_feature_maps.png')  # Speichern der kombinierten Feature Maps
plt.close()