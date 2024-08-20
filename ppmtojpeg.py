import os
from PIL import Image

def convert_ppm_to_jpeg(source_directory, target_directory):
    # Stelle sicher, dass das Zielverzeichnis existiert
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if file.endswith('.ppm'):
                # Pfad der originalen PPM-Datei
                ppm_path = os.path.join(root, file)
                # Erstelle einen relativen Pfad für das Zielverzeichnis
                relative_path = os.path.relpath(root, source_directory)
                # Neues Zielverzeichnis, das die Verzeichnisstruktur beibehält
                new_target_dir = os.path.join(target_directory, relative_path)
                if not os.path.exists(new_target_dir):
                    os.makedirs(new_target_dir)
                # Pfad, an dem die JPEG-Datei gespeichert wird
                jpeg_path = os.path.join(new_target_dir, os.path.splitext(file)[0] + '.jpeg')

                # Bild laden und konvertieren
                with Image.open(ppm_path) as img:
                    img.convert('RGB').save(jpeg_path, 'JPEG')
                    
                print(f'Converted {ppm_path} to {jpeg_path}')

# Verzeichnis, in dem Ihre PPM-Dateien gespeichert sind
source_directory = '/home/paul/TSR/ppms'
# Zielverzeichnis, in dem die JPEG-Dateien gespeichert werden sollen
target_directory = '/home/paul/TSR/Jpegs'

convert_ppm_to_jpeg(source_directory, target_directory)
