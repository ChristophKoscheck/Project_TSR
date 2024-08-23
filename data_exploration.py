##############################################################################
#                        Exploration des Datensatzes                         #
##############################################################################
#  Autoren   : Christoph Koscheck, Paul Smidt                                #
#  Vorlesung : Künstliche Intelligenz, Verkehrszeichenerkennung              #
#  Datum     : 23. August 2024                                               #
##############################################################################
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from skimage.transform import resize
import seaborn as sns
import tensorflow as tf
from skimage import feature
# ----------------------------------------------------------------------------

resolution = 64
sample_image_num = 3

# Funktionsaufruf
def main():
    # Trainings- und Testdaten laden
    ROOT_PATH = "./"
    train_data_dir = os.path.join(ROOT_PATH, "Training")
    test_data_dir = os.path.join(ROOT_PATH, "Testing")

    train_images, train_labels, train_paths = load_data(train_data_dir)
    test_images, test_labels, test_paths = load_data(test_data_dir)

    # DataFrame für Pfade und ClassIds erstellen
    labels_df = pd.DataFrame({
        'Path': np.concatenate([train_paths, test_paths]),
        'ClassId': np.concatenate([train_labels, test_labels])
    })

    # Ausgabe der Klassenverteilung
    print("\nVerteilung der Klassen im Datensatz:")
    print(labels_df['ClassId'].value_counts())

    # Visualisierung der Klassenverteilung
    plot_class_distribution(labels_df)

    # Visualisierung von Stichproben der Bilder jeder Klasse
    plot_images_for_classes(labels_df, train_data_dir, test_data_dir, n_images=sample_image_num)

    # Berechnung und Ausgabe des Seitenverhältnisses der Bilder und Pixel-Anzahl
    dimensions_df = get_image_stats(train_data_dir, test_data_dir, labels_df)

    # Visualisierung der Verteilung der Bildgrößen und Seitenverhältnisse
    plot_image_stats(dimensions_df)

    # Berechnung der dominierenden Farben
    vibrant_colors = calculate_vibrant_colors(labels_df, train_data_dir, test_data_dir)
    labels_df['VibrantColor'] = vibrant_colors

    # Visualisierung von Stichproben mit ihren dominierenden Farben
    plot_vibrant_color_images(labels_df, train_data_dir, test_data_dir, vibrant_colors, dimensions_df)

    # Extrahieren und Visualisieren von Farbhistogrammen (nur zur Veranschaulichung)
    plot_color_histograms(labels_df, train_data_dir, test_data_dir)

    # Extrahieren und Visualisieren von Kantenbildern (nur zur Veranschaulichung)
    plot_edges(labels_df, train_data_dir, test_data_dir)

    # Visualisierung von augmentierten Bildern (nur zur Veranschaulichung)
    augment_and_plot_images(labels_df, train_data_dir, test_data_dir)

## Funktionen
# Laden der Trainings- und Testdaten
def load_data(data_dir, target_size=(resolution, resolution)):  
    images = []
    labels = []
    paths = []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for f in os.listdir(class_path):
                if f.endswith('.ppm'):
                    image = io.imread(os.path.join(class_path, f))
                    image_resized = resize(image, target_size, anti_aliasing=True)
                    images.append(image_resized)
                    labels.append(int(class_dir))
                    paths.append(os.path.join(class_dir, f))
    return np.array(images), np.array(labels), np.array(paths)

# Visualisierung der Klassenverteilung
def plot_class_distribution(labels_df):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='ClassId', data=labels_df, palette='viridis')
    plt.title('Verteilung der Verkehrszeichenklassen')
    plt.xlabel('Klassen-ID')
    plt.ylabel('Anzahl der Bilder')
    plt.show()

# Visualisierung von Stichproben der Bilder jeder Klasse
def plot_images_for_classes(labels_df, train_data_dir, test_data_dir, n_images):
    unique_classes = labels_df['ClassId'].unique()
    n_classes = len(unique_classes)
    n_cols = min(n_classes, 5*sample_image_num)
    n_rows = (n_classes * n_images + n_cols - 1) // n_cols + 1
    plt.figure(figsize=(15, n_rows * 2))

    for i, class_id in enumerate(unique_classes):
        class_images = labels_df[labels_df['ClassId'] == class_id]['Path'].values
        for j in range(min(len(class_images), n_images)):
            img_path_train = os.path.join(train_data_dir, class_images[j])
            img_path_test = os.path.join(test_data_dir, class_images[j])
            if os.path.exists(img_path_train):
                img_path = img_path_train
            elif os.path.exists(img_path_test):
                img_path = img_path_test
            else:
                print(f"Bild nicht gefunden: {class_images[j]}")
                continue
            try:
                img = Image.open(img_path)
                plt.subplot(n_rows, n_cols, i * n_images + j + 1)
                plt.imshow(img)
                plt.axis('off')
                if j == 0:
                    plt.title(f'Class {class_id}')
            except FileNotFoundError as e:
                print(f"Fehler beim Laden des Bildes: {e}")
                continue

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.0, left=0.2, right=0.8, hspace=0.87, wspace=0.2)
    plt.show()

# Berechnung und Ausgabe des Seitenverhältnisses der Bilder und Pixel-Anzahl
def get_image_stats(train_data_dir, test_data_dir, labels_df):
    dimensions = []
    for img_path in labels_df['Path'].values:
        img_path_train = os.path.join(train_data_dir, img_path)
        img_path_test = os.path.join(test_data_dir, img_path)
        if os.path.exists(img_path_train):
            full_img_path = img_path_train
        elif os.path.exists(img_path_test):
            full_img_path = img_path_test
        else:
            print(f"Bild nicht gefunden: {img_path}")
            continue
        try:
            img = Image.open(full_img_path)
            dimensions.append(img.size)
        except FileNotFoundError as e:
            print(f"Fehler beim Laden des Bildes: {e}")
            continue
    
    dimensions_df = pd.DataFrame(dimensions, columns=['Width', 'Height'])
    dimensions_df['AspectRatio'] = dimensions_df['Width'] / dimensions_df['Height']
    
    print("\nZusammenfassende Statistiken für Bilddimensionen:")
    print(dimensions_df.describe())
    
    return dimensions_df

# Visualisierung der Verteilung der Bildgrößen und Seitenverhältnisse
def plot_image_stats(dimensions_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    sns.histplot(dimensions_df['Width'], kde=True, label='Breite', ax=ax1)
    sns.histplot(dimensions_df['Height'], kde=True, label='Höhe', ax=ax1)
    ax1.set_xlabel('Dimension in Pixel')
    ax1.set_ylabel('Anzahl der Bilder')
    ax1.set_title('Verteilung von Bildbreite und -höhe')
    ax1.legend()

    sns.histplot(dimensions_df['AspectRatio'], kde=True, label='Bildseitenverhältnis', ax=ax2, color='r')
    ax2.set_xlabel('Bildseitenverhältnis')
    ax2.set_ylabel('Anzahl der Bilder')
    ax2.set_title('Verteilung des Bildseitenverhältnisses')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Berechnung der dominierenden Farben
def calculate_vibrant_colors(labels_df, train_data_dir, test_data_dir):
    def calculate_vibrant_color(image):
        image = image.convert('RGB')
        np_image = np.array(image)
        np_image = np_image.reshape(-1, 3)
        avg_color = np.mean(np_image, axis=0)
        max_color = np.argmax(avg_color)
        color_names = ['Rot', 'Grün', 'Blau']
        vibrant_color = color_names[max_color]
        return vibrant_color

    vibrant_colors = []
    for img_path in labels_df['Path']:
        img_path_train = os.path.join(train_data_dir, img_path)
        img_path_test = os.path.join(test_data_dir, img_path)
        if os.path.exists(img_path_train):
            full_img_path = img_path_train
        elif os.path.exists(img_path_test):
            full_img_path = img_path_test
        else:
            vibrant_colors.append('Unknown')
            continue
        try:
            img = Image.open(full_img_path)
            vibrant_color = calculate_vibrant_color(img)
            vibrant_colors.append(vibrant_color)
        except FileNotFoundError:
            vibrant_colors.append('Unknown')

    return vibrant_colors

# Visualisierung von Stichproben mit ihren dominierenden Farben
def plot_vibrant_color_images(labels_df, train_data_dir, test_data_dir, vibrant_colors, dimensions_df):
    sample_images = labels_df.sample(5)
    plt.figure(figsize=(12, 6))

    for idx, img_path in enumerate(sample_images['Path']):
        img_path_train = os.path.join(train_data_dir, img_path)
        img_path_test = os.path.join(test_data_dir, img_path)
        if os.path.exists(img_path_train):
            full_img_path = img_path_train
        elif os.path.exists(img_path_test):
            full_img_path = img_path_test
        else:
            print(f"Bild nicht gefunden: {img_path}")
            continue
        try:
            img = Image.open(full_img_path)
            vibrant_color = labels_df.loc[labels_df['Path'] == img_path, 'VibrantColor'].values[0]
            plt.subplot(2, 5, idx+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Dominierende Farbe: {vibrant_color}')
        except FileNotFoundError as e:
            print(f"Fehler beim Laden des Bildes: {e}")

    plt.tight_layout()
    plt.show()

    # Dominante Farben in numerische Werte umwandeln und zur DataFrame hinzufügen
    color_mapping = {'Rot': 0, 'Grün': 1, 'Blau': 2}
    labels_df['VibrantColor'] = labels_df['VibrantColor'].map(color_mapping)

    # Bild-Dimensionen hinzufügen
    labels_df['Width'] = dimensions_df['Width']
    labels_df['Height'] = dimensions_df['Height']

    # Berechnung des Seitenverhältnisses
    labels_df['AspectRatio'] = labels_df['Width'] / labels_df['Height']

    # Korrelationsmatrix der Merkmale
    correlation = labels_df[['ClassId', 'VibrantColor', 'Width', 'Height', 'AspectRatio']].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korrelationsmatrix der Merkmale')
    plt.xlabel('Merkmale')
    plt.ylabel('Merkmale')
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=['Klassen-ID', 'Farbe', 'Breite', 'Höhe', 'Seitenverhältnis'])
    plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=['Klassen-ID', 'Farbe', 'Breite', 'Höhe', 'Seitenverhältnis'])
    plt.show()

# Extrahieren und Visualisieren von Farbhistogrammen (nur zur Veranschaulichung)
def plot_color_histograms(labels_df, train_data_dir, test_data_dir):
    def extract_color_histogram(image, bins=(8, 8, 8)):
        image = image.convert('RGB')
        hist = np.histogramdd(np.array(image).reshape(-1, 3), bins=bins, range=((0, 256), (0, 256), (0, 256)))
        return hist[0]

    sample_images = labels_df.sample(5)
    plt.figure(figsize=(15, 12))

    for idx, img_path in enumerate(sample_images['Path']):
        img_path_train = os.path.join(train_data_dir, img_path)
        img_path_test = os.path.join(test_data_dir, img_path)
        if os.path.exists(img_path_train):
            full_img_path = img_path_train
        elif os.path.exists(img_path_test):
            full_img_path = img_path_test
        else:
            print(f"Bild nicht gefunden: {img_path}")
            continue
        try:
            img = Image.open(full_img_path)
            hist = extract_color_histogram(img)
            plt.subplot(2, 5, idx+1)
            plt.imshow(img)
            plt.axis('off')
            plt.subplot(2, 5, idx+6)
            colors = ['r', 'g', 'b']
            for i, color in enumerate(colors):
                plt.plot(hist[:, i].flatten(), color=color, label=f'{color.upper()} Kanal')
            plt.xlabel('Bin-Nummer')
            plt.ylabel('Häufigkeit')
            plt.title(f'Farbhistogram - Klasse {sample_images.iloc[idx]["ClassId"]}')
            plt.legend(loc='upper right')
        except FileNotFoundError as e:
            print(f"Fehler beim Laden des Bildes: {e}")
            continue

    plt.tight_layout()
    plt.show()

# Extrahieren und Visualisieren von Kantenbildern (nur zur Veranschaulichung)
def plot_edges(labels_df, train_data_dir, test_data_dir):
    def extract_edges(image):
        image = image.convert('L')
        image = np.array(image)
        edges = feature.canny(image)
        return edges

    sample_images = labels_df.sample(5)
    plt.figure(figsize=(12, 6))

    for idx, img_path in enumerate(sample_images['Path']):
        img_path_train = os.path.join(train_data_dir, img_path)
        img_path_test = os.path.join(test_data_dir, img_path)
        if os.path.exists(img_path_train):
            full_img_path = img_path_train
        elif os.path.exists(img_path_test):
            full_img_path = img_path_test
        else:
            print(f"Bild nicht gefunden: {img_path}")
            continue
        try:
            img = Image.open(full_img_path)
            edges = extract_edges(img)
            plt.subplot(2, 5, idx+1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.xlabel('Breite')
            plt.ylabel('Höhe')
            plt.subplot(2, 5, idx+6)
            plt.imshow(edges, cmap='binary')
            plt.title(f'Kanten - Klasse {sample_images.iloc[idx]["ClassId"]}')
            plt.xlabel('Breite')
            plt.ylabel('Höhe')
        except FileNotFoundError as e:
            print(f"Fehler beim Laden des Bildes: {e}")
            continue

    plt.tight_layout()
    plt.show()

# Visualisierung von augmentierten Bildern (nur zur Veranschaulichung)
def augment_and_plot_images(labels_df, train_data_dir, test_data_dir):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range = 12,    # Drehe Bilder zufällig um max. 12 Grad
        width_shift_range = 0.2,    # Verschiebe Bilder horizontal um max. 20%
        height_shift_range = 0.2,   # Verschiebe Bilder vertikal um max. 20%
        shear_range = 0.1,  # Schere Bilder zufällig um max. 10%
        zoom_range = 0.25,  # Zoome zufällig in Bilder hinein um max. 25%
        horizontal_flip = False,    # Nicht horizontal spiegeln
        vertical_flip = False,  # Nicht vertikal spiegeln
        fill_mode='nearest' # Fülle leere Pixel mit den nächsten Nachbarn
    )

    sample_image_train_path = os.path.join(train_data_dir, labels_df.iloc[0]['Path'])
    sample_image_test_path = os.path.join(test_data_dir, labels_df.iloc[0]['Path'])
    if os.path.exists(sample_image_train_path):
        sample_image_path = sample_image_train_path
    elif os.path.exists(sample_image_test_path):
        sample_image_path = sample_image_test_path
    else:
        raise FileNotFoundError(f"Beispielbild in keinem der Verzeichnisse gefunden.")

    sample_image = np.array(Image.open(sample_image_path))

    augmented_images = [datagen.random_transform(sample_image) for _ in range(5)]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 6, 1)
    plt.imshow(sample_image)
    plt.title('Original')
    plt.axis('off')

    for i, aug_img in enumerate(augmented_images):
        plt.subplot(1, 6, i+2)
        plt.imshow(aug_img)
        plt.title(f'Erweitert {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Hauptprogramm
if __name__ == "__main__":
    main()
