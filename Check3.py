import tensorflow as tf
import numpy as np
import os
from skimage import io, transform
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import csv
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from skimage.transform import resize


# Define Parameters
resolution = 64
modell_nummer = 4
model_path = f'/home/paul/TSR/Test_Model_{modell_nummer}.h5'

# Load your trained model
model = load_model(model_path)

# Define the directory containing the images
image_dir = '/home/paul/TSR/TryTest'  # Update this path to your image directory


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

images, labels = load_data(image_dir)


def load_and_preprocess_image(image_path, target_size=(resolution, resolution)):
    image = io.imread(image_path)
    image = transform.resize(image, target_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_traffic_sign(image_path, model):
    image = load_and_preprocess_image(image_path)
    prediction_images = model.predict(image)
    predicted_class_index = np.argmax(prediction_images, axis=1)[0]
    return predicted_class_index

def read_labels_and_predict(image_dir, model, csv_file_path):
    label_map = {}
    with open(csv_file_path, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            filename, label = row
            label_map[filename] = int(label)

    predictions = []
    true_labels = []

    for filename, true_label in label_map.items():
        image_path = os.path.join(image_dir, filename)
        if os.path.exists(image_path):
            predicted_label = predict_traffic_sign(image_path, model)
            predictions.append(predicted_label)
            true_labels.append(true_label)

    return true_labels, predictions,



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
              42:'',
              43:'',
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
              55:'',
              56:'Crossing for pedestrians',
              57:'Crossing for cyclists',
              58:'Parking exit',
              59:'Information Sign : Speed bump',
              60:'End of the priority road',
              61:'Begin of a priority road'
    }


# Path to the CSV file containing true labels
label_csv_path = '/home/paul/TSR/TryTest/Labels.csv'  # Adjust this to the actual path

# Get true labels and predictions
true_labels, predictions = read_labels_and_predict(image_dir, model, label_csv_path)
print(f"Predictions:{predictions}")

# Define the labels parameter for the classification report
labels = sorted(set(true_labels))

# Classification Report
#Generate and print the classification report using the defined class labels
class_names = [class_labels[label].strip() for label in labels]  # Create a list of class names based on the sorted labels
print(classification_report( true_labels, predictions, labels=labels))


# Prepare for the confusion matrix
unique_labels = sorted(set(class_labels.keys()))  # All possible labels from the model training
class_names_matrix = [class_labels[label].strip() for label in unique_labels]


# Confusion Matrix
# Generate the confusion matrix
cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
# Plot the confusion matrix
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(40, 40), cmap=plt.cm.Greens,
                                show_absolute=True, show_normed=True, class_names=class_names_matrix)
plt.rcParams.update({'font.size': 16})  # Adjust font size for better readability
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
