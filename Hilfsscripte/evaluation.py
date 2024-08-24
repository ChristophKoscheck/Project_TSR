import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage import io, transform, feature
from skimage.transform import resize
from sklearn.preprocessing import normalize
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf
import pandas as pd
from skimage import io
from skimage.transform import resize
from skimage.exposure import equalize_adapthist
from matplotlib import pyplot as plt
from keras import datasets, layers, models
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from IPython.display import display




##Inforamtionen zur Codeversion und der Modellversion
#Aenderungen hier eingeben:
modell_nummer = 4

NUM_EPOCHS = 15 # Number of training epochs 
INIT_LR = 1e-3 # Initial Learning Rate for ADAM training
# BS = 64 # Size of minibatches
BS = 32  # Adjust the batch size to match the number of labels
# Auflösung
resolution = 32
batch_size = BS
epochs = NUM_EPOCHS


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
# ROOT_PATH = "/home/paul/TSR"                                	# Use in Windows Subsystem for Linux
ROOT_PATH = ""                                                # Use this line if you are running the code in the same directory as the dataset
train_data_dir = os.path.join(ROOT_PATH, "Training")
test_data_dir = os.path.join(ROOT_PATH, "Testing")
train_images, train_labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)

# Datagenerator
training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.3,
    height_shift_range=0.2,
    shear_range=0.25,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest')

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

# Create a DataFrame for labels and paths
labels_df = pd.DataFrame({
    'ClassId': np.concatenate([train_labels, test_labels])
})

df_signnames = pd.DataFrame({'ClassId': test_labels})

# Put Generator into Model
training_generator = training_datagen.flow(train_images, train_labels, batch_size=batch_size)
validation_datagen = validation_datagen.flow(test_images, test_labels, batch_size=batch_size)

# Berechne die Summe aller Trainingsbilder
total_train_images = len(train_images)

# Normalize the images
train_images_norm = train_images / 255.0
test_images_norm = test_images / 255.0

# Kopie der nicht encodierten Daten erstellen
trainY_old = train_labels
trainX = train_images
testY_old = test_labels
testX = test_images

# Daten one-hot-encodieren
trainY = tf.keras.utils.to_categorical(train_labels)
testY = tf.keras.utils.to_categorical(test_labels)

def generateCNN(width, height, depth, classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(8,(5,5),input_shape=(width, height, depth), padding="same", name="conv_1"))
    model.add(tf.keras.layers.Activation('relu', name="activation_1"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_1"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="max_pooling2d_1"))
    
    model.add(tf.keras.layers.Conv2D(16,(3,3), padding="same", name="conv_2"))
    model.add(tf.keras.layers.Activation('relu', name="activation_2"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_2"))

    model.add(tf.keras.layers.Conv2D(16,(3,3), padding="same", name="conv_3"))
    model.add(tf.keras.layers.Activation('relu', name="activation_3"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_3"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="max_pooling2d_2"))

    model.add(tf.keras.layers.Conv2D(32,(3,3), padding="same", name="conv_4"))
    model.add(tf.keras.layers.Activation('relu', name="activation_4"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_4"))

    model.add(tf.keras.layers.Conv2D(32,(3,3), padding="same", name="conv_5"))
    model.add(tf.keras.layers.Activation('relu', name="activation_5"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="max_pooling2d_3"))
    model.add(tf.keras.layers.Flatten(name="flatten_1"))
    
    model.add(tf.keras.layers.Dense(128, activation='relu', name="dense_1"))
    model.add(tf.keras.layers.Activation('relu', name="activation_6"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_5"))
    
    model.add(tf.keras.layers.Dropout(.2, name="dropout_1"))
    
    model.add(tf.keras.layers.Dense(128, activation='relu', name="dense_2"))
    model.add(tf.keras.layers.Activation('relu', name="activation_7"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_6"))
    model.add(tf.keras.layers.Dropout(.2, name="dropout_2"))

    model.add(tf.keras.layers.Dense(classes, activation='relu', name="dense_3"))
    model.add(tf.keras.layers.Activation('softmax', name="activation_8"))
      
    return model
    
generateCNN(resolution,resolution,3,62).summary()

model=generateCNN(resolution,resolution,3,62)

opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5)) 
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# construct the image generator for data augmentation
aug = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest"
)

def get_classWeight(trainY_old):
    unique, frequency = np.unique(trainY_old, return_counts = True)

    classWeight={}
    sum_frequencies=frequency.sum()
    for i in unique:
        classWeight[i]=frequency[i]/sum_frequencies
    return classWeight

classWeight = get_classWeight(trainY_old)

# compile the model and train the network
print("[INFO] training network...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=trainX.shape[0] // BS,
    epochs=NUM_EPOCHS,
    class_weight=classWeight,
    verbose=1)

plt.plot(H.history.get('accuracy'))
plt.plot(H.history.get('val_accuracy'))
plt.ylabel('Accuracy')
plt.xlabel('Epoche')
plt.legend(['Training', 'Testing'])
plt.show()

plt.plot(H.history.get('loss'))
plt.plot(H.history.get('val_loss'))
plt.ylabel('Loss')
plt.xlabel('Epoche')
plt.legend(['Training', 'Testing'])
plt.show()

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

# Create a DataFrame for test labels and their corresponding sign names
df_signnames = pd.DataFrame({
    'ClassId': np.unique(test_labels)  # Use unique test labels only
})
df_signnames['SignName'] = df_signnames['ClassId'].map(class_labels)

# Predictions
predictions = model(testX).numpy()
predicted_targets = np.argmax(predictions, axis=1)

# Ensure that the ClassId matches the number of unique labels
assert len(np.unique(test_labels)) == len(df_signnames), "Mismatch between the number of unique labels and sign names"

# If this assertion fails, identify missing classes
missing_classes = set(np.unique(test_labels)) - set(df_signnames['ClassId'])
if missing_classes:
    print(f"Missing classes in sign names mapping: {missing_classes}")

# Ensure `target_names` is correctly aligned with actual labels in `testY_old`
unique_labels = sorted(np.unique(testY_old))

# Generate the classification report
print(classification_report(testY_old, predicted_targets, target_names=df_signnames.loc[df_signnames['ClassId'].isin(unique_labels), "SignName"].values))

cm = confusion_matrix(testY_old, predicted_targets)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(40, 40), cmap=plt.cm.Greens,show_absolute=True, show_normed=True, class_names=df_signnames['ClassId'])
plt.rcParams.update({'font.size': 16})
plt.xlabel('Predictions')
plt.ylabel('Actuals')
plt.title('Confusion Matrix')
plt.show()

def five_wrong_images(targets, predicted_targets, data, df_signnames):
    count = 0
    for i,e in enumerate(targets):
        if targets[i] != predicted_targets[i] and count < 5:  
            print("Richtige Klasse: ", df_signnames[df_signnames['ClassId']==targets[i]]['SignName'].values)
            img_true=Image.open("./Icons/TSR/{}.jpg".format(targets[i]))
            display(img_true)
            print("Vorhergesagte Klasse ", df_signnames[df_signnames['ClassId']==predicted_targets[i]]['SignName'].values)
            img_pre=Image.open("./Icons/TSR/{}.jpg".format(predicted_targets[i]))
            display(img_pre)
            img = data[i]
            plt.imshow(img)
            plt.show() 
            count+=1

five_wrong_images(testY_old, predicted_targets, testX, df_signnames)

#Iterate thru all the layers of the model
for layer in model.layers:
    if 'conv' in layer.name:
        weights, bias= layer.get_weights()
        
        #normalize filter values between  0 and 1 for visualization
        f_min, f_max = weights.min(), weights.max()
        filters = (weights - f_min) / (f_max - f_min) 
        
        print(layer.name, filters.shape)
        print(filters.shape[3])
        
        filter_cnt=1
        
        #plotting all the filters
        for i in range(filters.shape[3]):
            #get the filters
            filt=filters[:,:,:, i]
            #plotting each of the channel, color image RGB channels
            for j in range(3):
                ax= plt.subplot(filters.shape[3], filters.shape[0], filter_cnt  )
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(filt[:,:, j],cmap='gray')
                filter_cnt+=1
        plt.show()

img_path='Training/00000/01153_00000.ppm' 
# Define a new Model, Input= image 
# Output= intermediate representations for all layers in the  
# previous model after the first.
successive_outputs = [layer.output for layer in model.layers[1:]]#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)#Load the input image
img = load_img(img_path, target_size=(32,32,3))# Convert ht image to Array of dimension (32,32,3)
x   = img_to_array(img)                           
x   = x.reshape((1,) + x.shape)# Rescale by 1/255
x /= 255.0# Let's run input image through our vislauization network
# to obtain all intermediate representations for the image.
successive_feature_maps = visualization_model.predict(x)# Retrieve are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    print(feature_map.shape)
    if len(feature_map.shape) == 4:
    
        # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers
        n_features = feature_map.shape[-1]  # number of features in the feature map
        size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
    
        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))
    
        # Postprocess the feature to be visually palatable
        for i in range(n_features):
            x  = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std ()
            x *=  64
            x += 128
            x  = np.clip(x, 0, 255).astype('uint8')
          # Tile each filter into a horizontal grid
            display_grid[:, i * size : (i + 1) * size] = x# Display the grid
        scale = 20. / n_features
        plt.figure( figsize=(scale * n_features, scale) )
        plt.title ( layer_name )
        plt.grid  ( False )
        plt.imshow( display_grid, aspect='auto', cmap='gray' )



