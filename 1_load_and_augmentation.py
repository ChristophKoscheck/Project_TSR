##############################################################################
#                        Erweiterung des Datensatzer                         #
##############################################################################
#  Autoren   : Christoph Koscheck, Paul Smidt                                #
#  Vorlesung : Künstliche Intelligenz, Verkehrszeichenerkennung              #
#  Datum     : 23. August 2024                                               #
##############################################################################
# ----------------------------------------------------------------------------
import os
import shutil
import cv2
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import requests
import zipfile
# ----------------------------------------------------------------------------

training_url = "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip"
testing_url = "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip"

# Download und Entpacken des Trainingsdatensatzes
training_zip = requests.get(training_url)
with open("BelgiumTSC_Training.zip", "wb") as file:
    file.write(training_zip.content)
with zipfile.ZipFile("BelgiumTSC_Training.zip", "r") as zip_ref:
    zip_ref.extractall()
    os.rename(zip_ref.namelist()[0], 'Training')

# Download und Entpacken des Testdatensatzes
testing_zip = requests.get(testing_url)
with open("BelgiumTSC_Testing.zip", "wb") as file:
    file.write(testing_zip.content)
with zipfile.ZipFile("BelgiumTSC_Testing.zip", "r") as zip_ref:
    zip_ref.extractall()
    os.rename(zip_ref.namelist()[0], 'Testing')

# Definiere den ImageDataGenerator für die Datenerweiterung
datagen = ImageDataGenerator(
        rotation_range=15,
        zca_epsilon=1e-06,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='nearest',
        shear_range=0.2,
        brightness_range=[0.5, 1.5])

# Quell- und Zielverzeichnisse
source_dir_train = 'Training'
target_dir_train = 'TSR_Data_Train'

source_dir_test = 'Testing'
target_dir_test = 'TSR_Data_Test'

# Funktion zum Erstellen von Unterverzeichnissen
def create_subdirectories(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir):
        # Ermittele den relativen Pfad des aktuellen Verzeichnisses
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)
        
        # Erstelle das Zielverzeichnis, wenn es nicht existiert
        if not os.path.exists(target_path):
            os.makedirs(target_path)

# Funktionsaufruf zum Erstellen von Unterverzeichnissen
create_subdirectories(source_dir_train, target_dir_train)
create_subdirectories(source_dir_test, target_dir_test)

# Funktion zum Laden eines Bildes
def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((1,) + img.shape)  # Reshape to (1, height, width, channels)
    return img

# Funktion zur Bestimmung der Anzahl der Erweiterungen abhängig von der Anzahl der vorhandenen Bilder
def determine_augmentation_count(num_images):
    if num_images < 10:
        return 6
    elif num_images < 20:
        return 3
    elif num_images < 75:
        return 2
    else:
        return 1
    
# Loop über Trainings- und Testverzeichnisse
for source_dir, target_dir in [(source_dir_train, target_dir_train), (source_dir_test, target_dir_test)]:
    for root, dirs, files in os.walk(source_dir):
        if any(file.endswith('.ppm') for file in files):  
            # Laden der Metadaten aus der CSV-Datei
            csv_path = os.path.join(root, f"GT-{os.path.basename(root)}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, sep=';')

                # Zielunterverzeichnis basierend auf dem relativen Pfad
                relative_path = os.path.relpath(root, source_dir)
                target_subdir = os.path.join(target_dir, relative_path)

                # Zähle die Anzahl der vorhandenen Bilder im Zielunterverzeichnis	
                existing_images = [f for f in os.listdir(target_subdir) if f.endswith('.ppm')]
                num_existing_images = len(existing_images)

                # Bestimme die Anzahl der Erweiterungen basierend auf der Anzahl der vorhandenen Bilder
                num_augmentations = determine_augmentation_count(num_existing_images)

                # Liste für die neuen Metadaten
                new_metadata_list = []

                # Verarbeite jedes Bild im aktuellen Verzeichnis
                for file in files:
                    if file.endswith('.ppm'):
                        img_path = os.path.join(root, file)
                        img = load_image(img_path)

                        new_img_path = os.path.join(target_dir, relative_path, file)
                        shutil.copy(img_path, new_img_path)

                        metadata = df[df['Filename'] == file]
                        new_metadata_list.append(metadata.copy())

                        aug_iter = datagen.flow(img, batch_size=1)

                        # Generieren und speichern der augmentierten Bilder
                        for i in range(num_augmentations):
                            img_aug = next(aug_iter)[0].astype('uint8')

                            new_filename = f"{os.path.splitext(file)[0]}_aug_{i}.ppm"
                            new_aug_img_path = os.path.join(target_dir, relative_path, new_filename)
                            cv2.imwrite(new_aug_img_path, cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))

                            new_metadata = metadata.copy()
                            new_metadata['Filename'] = new_filename
                            new_metadata_list.append(new_metadata)

                if new_metadata_list:
                    new_metadata_df = pd.concat(new_metadata_list, ignore_index=True)
                    df = pd.concat([df, new_metadata_df], ignore_index=True)

                # Speichern der neuen Metadaten in einer CSV-Datei
                new_csv_path = os.path.join(target_dir, os.path.relpath(csv_path, source_dir))
                df.to_csv(new_csv_path, index=False, sep=';')
