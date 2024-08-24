import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io, transform
from keras.models import load_model
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
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from IPython.display import display

modell_nummer = 4
model_path = f'F:/TSR/Test_Model_{modell_nummer}.h5'

# Load your trained model
model = load_model(model_path)

img_path='Training/00000/01153_00000.ppm' 
# Define a new Model, Input= image 
# Output= intermediate representations for all layers in the  
# previous model after the first.
successive_outputs = [layer.output for layer in model.layers[1:]]#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)#Load the input image
img = load_img(img_path, target_size=(128,128,3))# Convert ht image to Array of dimension (32,32,3)
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
            if x.std() != 0:
                x /= x.std()
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
        plt.savefig(f'{layer_name}.png')
