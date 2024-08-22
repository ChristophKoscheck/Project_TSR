import os
import shutil
import pandas as pd

def copy_images_and_create_csv(source_folder, target_folder):
  """
  Kopiert Bilder aus einem Quellordner mit Unterordnern in einen Zielordner und erstellt eine CSV-Datei mit Dateinamen und Unterordnernamen.
  Überspringt leere Unterordner.

  Args:
    source_folder: Der Quellordner, in dem die Bilder liegen.
    target_folder: Der Zielordner, in den die Bilder kopiert werden sollen.
  """

  data = []

  for root, dirs, files in os.walk(source_folder):
      if files:  # Nur weitermachen, wenn der Ordner Dateien enthält
          for file in files:
              if file.endswith('.jpg'):
                  source_file = os.path.join(root, file)
                  target_file = os.path.join(target_folder, file)
                  shutil.copy2(source_file, target_file)

                  subfolder = os.path.basename(root)
                  data.append([file, subfolder])

  df = pd.DataFrame(data, columns=['Dateiname', 'Unterordner'])
  csv_path = os.path.join(target_folder, 'Labels.csv')
  df.to_csv(csv_path, index=False)

# Beispielhafte Verwendung
source_folder = '/home/paul/TSR/OwnTestImages'
target_folder = '/home/paul/TSR/TryTest'

copy_images_and_create_csv(source_folder, target_folder)