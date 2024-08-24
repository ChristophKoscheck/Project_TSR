import os
import csv

def save_filenames_to_csv(base_path):
    # Gehe durch jeden Unterordner im Basisverzeichnis
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path):  # Stelle sicher, dass es ein Verzeichnis ist
            # Liste alle Dateien im Unterordner
            files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
            # Sortiere die Dateien (optional, um sicherzustellen, dass die Reihenfolge konsistent ist)
            files.sort()
            # CSV-Datei für diesen Unterordner erstellen
            csv_file_path = os.path.join(subdir_path, f"{subdir}.csv")
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['filename', 'folder_name'])  # Schreibe die Kopfzeile
                # Schreibe die Dateinamen und Ordnernamen in die CSV
                for filename in files:
                    csvwriter.writerow([filename, subdir])
                print(f"CSV created in {subdir_path}: {csv_file_path}")

# Pfad zum Basisverzeichnis (angepasst an deinen spezifischen Fall)
base_path = "/home/paul/TSR/OwnTestImages"
save_filenames_to_csv(base_path)
