import os
from PIL import Image

def convert_png_to_jpg(root_folder):
    # Durchlaufe alle Dateien und Ordner im angegebenen Verzeichnis
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.png'):
                # Kompletter Pfad der PNG-Datei
                png_file = os.path.join(foldername, filename)
                
                # Konvertiere die Datei nach JPG
                img = Image.open(png_file)
                jpg_file = os.path.splitext(png_file)[0] + '.jpg'
                img = img.convert('RGB')
                img.save(jpg_file, 'JPEG')
                
                print(f'Converted: {png_file} -> {jpg_file}')

if __name__ == "__main__":
    # Geben Sie hier den Pfad zum Hauptordner an
    root_folder = '/home/paul/TSR/OwnTestImages'
    convert_png_to_jpg(root_folder)
