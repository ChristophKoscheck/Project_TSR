##############################################################################
#                        Feature Map Visualisierung                          #
##############################################################################
#  Autoren   : Christoph Koscheck, Paul Smidt                                #
#  Vorlesung : Künstliche Intelligenz, Verkehrszeichenerkennung              #
#  Datum     : 23. August 2024                                               #
##############################################################################
# ----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.image as mpimg
# ----------------------------------------------------------------------------

modell_nummer = 6
model_path = f'F:/TSR/Test_Model_{modell_nummer}.h5'

# Laden des trainierten Modells
model = load_model(model_path)

# Absoluter Pfad zum Bild für das die Feature Maps erstellt werden sollen
img_path = 'F:/TSR/RawTryTest/21/21_0.jpg'

# Erstellen des Modells, welches die Feature Maps für jedes Layer berechnet
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

# Laden des Bildes und auf Auflösung anpassen
img = load_img(img_path, target_size=(64, 64, 3))

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

        filename = f"feature_map/feature_map_{i}_{layer_name}.png"
        plt.savefig(filename)
        image_files.append(filename)
        plt.close()

# Kombinieren der Feature Maps in einem Plot
n_layers = len(image_files)

plt.figure(figsize=(20, n_layers * 2))  # Anpassen der Größe des Plots an die Anzahl der Feature Maps

for i, filename in enumerate(image_files):
    img = mpimg.imread(filename) 
    ax = plt.subplot(n_layers, 1, i + 1)
    ax.imshow(img) 
    ax.axis('off')  
    ax.set_title(layer_names[i])

plt.tight_layout()
plt.savefig('feature_map/combined_feature_maps.png')  # Speichern der kombinierten Feature Maps
plt.show()
