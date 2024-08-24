import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
from skimage import transform
import pandas as pd

resolution = 64

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

train_images, train_labels = load_data("F:\TSR\RawTryTest")

labels_df_train = pd.DataFrame({
    'ClassId': train_labels
})

# Ausgabe der Klassenverteilung
print("\nVerteilung der Klassen im Datensatz:")
print(labels_df_train['ClassId'].value_counts())
print("Summe der Klassen: ", labels_df_train['ClassId'].value_counts().sum())

# Visualisierung der Klassenverteilung
def plot_class_distribution(labels_df, dataset_name):
    plt.figure(figsize=(16, 6))
    sns.countplot(x='ClassId', data=labels_df, palette='viridis')
    plt.title('Verteilung der Verkehrszeichenklassen {}'.format(dataset_name))
    plt.xlabel('Klassen-ID')
    plt.ylabel('Anzahl der Bilder')
    # plt.show()
    plt.savefig(f'exploration/class_distribution_{dataset_name}.png')
    plt.close()

plot_class_distribution(labels_df_train, "Eigener Datensatz")