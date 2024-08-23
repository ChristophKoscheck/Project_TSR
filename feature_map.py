import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

modell_nummer = 6
model_path = f'F:/TSR/Test_Model_{modell_nummer}.h5'

# Load your trained model
model = load_model(model_path)

img_path = 'F:/TSR/RawTryTest/21/21_0.jpg'

# Define a new Model, Input= image 
# Output= intermediate representations for all layers in the previous model after the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

# Load the input image
img = load_img(img_path, target_size=(64, 64, 3))

# Convert the image to Array of dimension (64, 64, 3)
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Rescale by 1/255
x /= 255.0

# Let's run the input image through our visualization network to obtain all intermediate representations for the image.
successive_feature_maps = visualization_model.predict(x)

# Retrieve the names of the layers
layer_names = [layer.name for layer in model.layers]

# Create a figure to hold all feature maps
plt.figure(figsize=(20, 20))

for i, (layer_name, feature_map) in enumerate(zip(layer_names, successive_feature_maps)):
    if len(feature_map.shape) == 4:  # Only plot conv/maxpool layers
        n_features = feature_map.shape[-1]  # number of features in the feature map
        size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)
        
        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))

        # Postprocess the feature to be visually palatable
        for j in range(n_features):
            x = feature_map[0, :, :, j]
            x -= x.mean()
            if x.std() != 0:
                x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            # Tile each filter into a horizontal grid
            display_grid[:, j * size: (j + 1) * size] = x

        # Plot each layer's feature maps
        ax = plt.subplot(len(successive_feature_maps), 1, i + 1)
        ax.set_title(layer_name)
        ax.grid(False)
        ax.imshow(display_grid, aspect='auto', cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])

plt.xlabel('Feature Maps')
plt.tight_layout()
plt.savefig('feature_maps_all_layers.png')
plt.show()
