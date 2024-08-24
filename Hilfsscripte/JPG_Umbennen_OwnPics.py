#import os
#
#def rename_and_delete_identifier_files(base_path):
#    # Gehe durch jeden Unterordner im Basisverzeichnis
#    for subdir in os.listdir(base_path):
#        subdir_path = os.path.join(base_path, subdir)
#        if os.path.isdir(subdir_path):  # Stelle sicher, dass es ein Verzeichnis ist
#            # Liste alle Dateien im Unterordner
#            files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
#            # Sortiere die Dateien (optional, um sicherzustellen, dass die Reihenfolge konsistent ist)
#            files.sort()
#            # Durchlaufe jede Datei im Unterordner
#            for i, filename in enumerate(files):
#                old_file_path = os.path.join(subdir_path, filename)
#                
#                # Prüfe, ob es sich um eine .Identifier-Datei handelt
#                if filename.endswith('.Identifier'):
#                    # Lösche die Datei
#                    os.remove(old_file_path)
#                    print(f"Deleted: {old_file_path}")
#                else:
#                    # Benenne die Datei um
#                    file_extension = os.path.splitext(filename)[1]  # Erhalte die Dateiendung (z.B. .jpg)
#                    new_file_name = f"{subdir}_{i}{file_extension}"  # Erstelle neuen Dateinamen mit ursprünglicher Endung
#                    new_file_path = os.path.join(subdir_path, new_file_name)
#                    os.rename(old_file_path, new_file_path)
#                    print(f"Renamed: {old_file_path} to {new_file_path}")
#
## Pfad zum Basisverzeichnis (angepasst an deinen spezifischen Fall)
#base_path = "/home/paul/TSR/OwnTestImages"
#
#rename_and_delete_identifier_files(base_path)


import os
from PIL import Image

def rename_and_delete_identifier_files(base_path):
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path):
            files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
            files.sort()
            for i, filename in enumerate(files):
                old_file_path = os.path.join(subdir_path, filename)

                if filename.endswith('.Identifier'):
                    os.remove(old_file_path)
                    print(f"Deleted: {old_file_path}")
                else:
                    file_extension = os.path.splitext(filename)[1]
                    new_file_name = f"{subdir}_{i}{file_extension}"
                    new_file_path = os.path.join(subdir_path, new_file_name)
                    
                    if file_extension.lower() == '.png':  # Check if the file is a PNG
                        img = Image.open(old_file_path)
                        rgb_img = img.convert('RGB')  # Convert PNG to RGB
                        new_file_path = new_file_path.replace('.png', '.jpg')  # Change the file extension to JPG
                        rgb_img.save(new_file_path, format='JPEG')
                        os.remove(old_file_path)  # Remove the original PNG file
                        print(f"Converted and renamed: {old_file_path} to {new_file_path}")
                    else:
                        os.rename(old_file_path, new_file_path)
                        print(f"Renamed: {old_file_path} to {new_file_path}")

# Set the base path to your directory
base_path = "/home/paul/TSR/OwnTestImages"

rename_and_delete_identifier_files(base_path)
