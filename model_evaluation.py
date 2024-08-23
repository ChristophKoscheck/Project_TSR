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
# ----------------------------------------------------------------------------

# Initialisierung
resolution = 64
modell_nummer = 6
# model_path = f'/home/paul/TSR/Test_Model_{modell_nummer}.h5'
model_path = f'F:/TSR/Test_Model_{modell_nummer}.h5'

# Laden des trainierten Modells
model = load_model(model_path)

# Pfad zum eignen Datensatz-Bildverzeichnis
# image_dir = '/home/paul/TSR/TryTest'  
image_dir = 'F:\\TSR\\TryTest'  
raw_image_dir = 'F:\\TSR\\RawTryTest' # Pfad zum Rohdatenverzeichnis

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
label_csv_path = 'F:/TSR/TryTest/Labels.csv'  # Adjust this to the actual path

# Labels des eignen Datensatzes einlesen und Vorhersagen
true_labels, predictions, image_paths = read_labels_and_predict(image_dir, model, label_csv_path)
print(f"Vorhersagen:{predictions}")

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
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(80, 80), cmap=plt.cm.Blues,
                                show_absolute=True, show_normed=True, class_names=class_names_matrix)
plt.rcParams.update({'font.size': 12})
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Wahre Klasse')
plt.title('Confusion Matrix')
plt.show()

# Fehlerhafte Klassifikationen
def misclassified_images(targets, predicted_targets, data, class_names):
    count = 0
    for i, e in enumerate(targets):
        if targets[i] != predicted_targets[i] and count < 20:
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

misclassified_images(true_labels, predictions, images, class_names)

# Create a directory to save the filter images if it doesn't exist
output_dir = "filters"
os.makedirs(output_dir, exist_ok=True)

# Code und Methode inspiriert von: https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
for layer in model.layers:
    if 'conv' in layer.name:
        weights, bias = layer.get_weights()
        
        # Normalisieren der Gewichte
        f_min, f_max = weights.min(), weights.max()
        filters = (weights - f_min) / (f_max - f_min) 
        
        print(layer.name, filters.shape)
        
        n_filters = filters.shape[3]
        n_channels = filters.shape[2]
        
        fig, axs = plt.subplots(n_filters, n_channels, figsize=(n_channels, n_filters))
        
        for i in range(n_filters):
            for j in range(n_channels):
                ax = axs[i, j]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(filters[:, :, j, i], cmap='viridis')
        
        plt.savefig(os.path.join(output_dir, f"{layer.name}_filters.png"), bbox_inches='tight', pad_inches=0)
        plt.close(fig)


