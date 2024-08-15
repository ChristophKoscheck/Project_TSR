import tensorflow as tf
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage.transform import resize
import matplotlib.pyplot as plt


# Define Parameters

resolution = 128
modell_nummer = 3

# Load your trained model
model = load_model('/home/paul/TSR/Test_Model_{}.h5'.format(modell_nummer))

def load_and_preprocess_image(image_path, target_size=(resolution, resolution)):
    # Load the image using skimage
    image = io.imread(image_path)

    # Resize the image to the target size
    image = transform.resize(image, target_size)

    # Normalize the image to the range [0, 1]
    image = image.astype('float32') / 255.0

    # Add a batch dimension
    image = np.expand_dims(image, axis=0)

    return image

def predict_traffic_sign(image_path, model, class_labels):
    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)
    
    # Predict the class probabilities
    predictions = model.predict(image)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Map the predicted index to the class label
    predicted_class_label = class_labels[predicted_class_index]
    
    return predicted_class_label

def test_accuracy_on_single_image(image_path, true_label, model, class_labels):
    predicted_label = predict_traffic_sign(image_path, model, class_labels)
    
    # Check if the prediction matches the true label
    is_correct = predicted_label == true_label
    
    # Output the results
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Prediction is {'correct' if is_correct else 'incorrect'}.")
    
    plt.imshow(io.imread(image_path))
    plt.title(predicted_label)
    plt.show()

    return is_correct

# Define the path to the image
image_path = '/home/paul/TSR/3.jpg' 

# Define the true label for the image
true_label = 'Stop Sign'  # Example: 'Stop Sign'

# Define the list of class labels
class_labels = {
              0 : 'Warning for a bad road surface',
              1 : 'Warning for a speed bump',
              2 : 'Warning for a slippery road surface',
              3 : 'Warning for a curve to the left',
              4 : 'Warning for a curve to the right',
              5 : 'Warning for a double curve, first left then right',                                                    # Merge Classes 5 & 6 later
              6 : 'Warning for a double curve, first left then right',
              7 : 'Watch out for children ahead',
              8 : 'Watch out for  cyclists',
              9 : 'Watch out for cattle on the road',
              10: 'Watch out for roadwork ahead',
              11: 'Traffic light ahead',
              12: 'Watch out for railroad crossing with barriers ahead',
              13: 'Watch out ahead for unknown danger',
              14: 'Warning for a road narrowing',
              15: 'Warning for a road narrowing on the left',
              16: 'Warning for a road narrowing on the right',
              17: 'Warning for side road on the right',
              18: 'Warning for an uncontrolled crossroad',
              19: 'Give way to all drivers',
              20: 'Road narrowing, give way to oncoming drivers',
              21: 'Stop and give way to all drivers',
              22: 'Entry prohibited (road with one-way traffic)',
              23: 'Cyclists prohibited',
              24: 'Vehicles heavier than indicated prohibited',
              25: 'Trucks prohibited',
              26: 'Vehicles wider than indicated prohibited',
              27: 'Vehicles higher than indicated prohibited',
              28: 'Entry prohibited',
              29: 'Turning left prohibited',
              30: 'Turning right prohibited',
              31: 'Overtaking prohibited',
              32: 'Driving faster than indicated prohibited (speed limit)',
              33: 'Mandatory shared path for pedestrians and cyclists',
              34: 'Driving straight ahead mandatory',
              35: 'Mandatory left',
              36: 'Driving straight ahead or turning right mandatory',
              37: 'Mandatory direction of the roundabout',
              38: 'Mandatory path for cyclists',
              39: 'Mandatory divided path for pedestrians and cyclists',
              40: 'Parking prohibited',
              41: 'Parking and stopping prohibited',
              42: '',
              43: '',
              44: 'Road narrowing, oncoming drivers have to give way',
              45: 'Parking is allowed',
              46: 'parking for handicapped',
              47: 'Parking for motor cars',
              48: 'Parking for goods vehicles',
              49: 'Parking for buses',
              50: 'Parking only allowed on the sidewalk',
              51: 'Begin of a residential area',
              52: 'End of the residential area',
              53: 'Road with one-way traffic',
              54: 'Dead end street',
              55: '',
              56: 'Crossing for pedestrians',
              57: 'Crossing for cyclists',
              58: 'Parking exit',
              59: 'Information Sign : Speed bump',
              60: 'End of the priority road',
              61: 'Begin of a priority road'
    }

# Test the accuracy on the provided image
test_accuracy_on_single_image(image_path, true_label, model, class_labels)

def evaluate_traffic_sign(image_path, true_label, model_path, class_labels):
    model = load_model(model_path)
    is_correct = test_accuracy_on_single_image(image_path, true_label, model, class_labels)
    return is_correct