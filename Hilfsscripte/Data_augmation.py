#import os
#import shutil
#import cv2
#import pandas as pd
#from keras.preprocessing.image import ImageDataGenerator
#
## Define the ImageDataGenerator with the desired augmentations
#datagen = ImageDataGenerator(
#        rotation_range=15,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=False,
#        vertical_flip=False,
#        fill_mode='nearest',
#        shear_range=0.2,
#        brightness_range=[0.5, 1.5])
#
## Source and target directories
##source_dir = '/home/paul/TSR/Training'
##target_dir = '/home/paul/TSR/TSR_Data_Train'
## Source and target directories
#source_dir = '/home/paul/TSR/Testing'
#target_dir = '/home/paul/TSR/TSR_Data_Test'
#
## Function to create subdirectories in the target directory
#def create_subdirectories(source_dir, target_dir):
#    for root, dirs, files in os.walk(source_dir):
#        # Determine the path relative to the source directory
#        relative_path = os.path.relpath(root, source_dir)
#        target_path = os.path.join(target_dir, relative_path)
#        
#        # Create the subdirectory in the target directory
#        if not os.path.exists(target_path):
#            os.makedirs(target_path)
#
## Call the function to create the subdirectories
#create_subdirectories(source_dir, target_dir)
#
## Function to load an image and convert it to a 4D tensor
#def load_image(img_path):
#    img = cv2.imread(img_path)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    img = img.reshape((1,) + img.shape)  # Reshape to (1, height, width, channels)
#    return img
#
## Loop through directories and perform augmentation
#for root, dirs, files in os.walk(source_dir):
#    if any(file.endswith('.ppm') for file in files):  # Check if there are any image files in the directory
#        # Load the corresponding CSV for the directory
#        csv_path = os.path.join(root, f"GT-{os.path.basename(root)}.csv")
#        if os.path.exists(csv_path):
#            df = pd.read_csv(csv_path, sep=';')
#
#            # List to collect all new metadata rows for the directory
#            new_metadata_list = []
#
#            # Process each image in the directory
#            for file in files:
#                if file.endswith('.ppm'):
#                    # Load image
#                    img_path = os.path.join(root, file)
#                    img = load_image(img_path)
#
#                    # Copy original image to the new directory
#                    relative_path = os.path.relpath(root, source_dir)
#                    new_img_path = os.path.join(target_dir, relative_path, file)
#                    shutil.copy(img_path, new_img_path)
#
#                    # Get image metadata
#                    metadata = df[df['Filename'] == file]
#
#                    # Append the original image's metadata to the new metadata list
#                    new_metadata_list.append(metadata.copy())
#
#                    # Create an iterator for augmentation
#                    aug_iter = datagen.flow(img, batch_size=1)
#
#                    # Create 10 augmented images
#                    for i in range(3):
#                        # Generate augmented image
#                        img_aug = next(aug_iter)[0].astype('uint8')
#
#                        # Save augmented image
#                        new_filename = f"{os.path.splitext(file)[0]}_aug_{i}.ppm"
#                        new_aug_img_path = os.path.join(target_dir, relative_path, new_filename)
#                        cv2.imwrite(new_aug_img_path, cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))
#
#                        # Adjust bounding box (adjustments may be needed based on augmentations)
#                        new_metadata = metadata.copy()
#                        new_metadata['Filename'] = new_filename
#                        new_metadata_list.append(new_metadata)
#
#            # After processing all images, append all new metadata to the DataFrame
#            if new_metadata_list:
#                new_metadata_df = pd.concat(new_metadata_list, ignore_index=True)
#                df = pd.concat([df, new_metadata_df], ignore_index=True)
#
#            # Save updated CSV
#            new_csv_path = os.path.join(target_dir, os.path.relpath(csv_path, source_dir))
#            df.to_csv(new_csv_path, index=False, sep=';')

import os
import shutil
import cv2
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

# Define the ImageDataGenerator with the desired augmentations
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

# Source and target directories
source_dir = '/home/paul/TSR/Training'
target_dir = '/home/paul/TSR/TSR_Data_Train'
# Source and target directories
#source_dir = '/home/paul/TSR/Testing'
#target_dir = '/home/paul/TSR/TSR_Data_Test'

# Function to create subdirectories in the target directory
def create_subdirectories(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir):
        # Determine the path relative to the source directory
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)
        
        # Create the subdirectory in the target directory
        if not os.path.exists(target_path):
            os.makedirs(target_path)

# Call the function to create the subdirectories
create_subdirectories(source_dir, target_dir)

# Function to load an image and convert it to a 4D tensor
def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((1,) + img.shape)  # Reshape to (1, height, width, channels)
    return img

# Function to determine the number of augmentations based on the number of existing images
def determine_augmentation_count(num_images):
    if num_images < 10:
        return 6
    elif num_images < 20:
        return 3
    elif num_images < 75:
        return 2
    else:
        return 1
    
# Loop through directories and perform augmentation
for root, dirs, files in os.walk(source_dir):
    if any(file.endswith('.ppm') for file in files):  # Check if there are any image files in the directory
        # Load the corresponding CSV for the directory
        csv_path = os.path.join(root, f"GT-{os.path.basename(root)}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, sep=';')

            # Target directory corresponding to the current source directory
            relative_path = os.path.relpath(root, source_dir)
            target_subdir = os.path.join(target_dir, relative_path)

            # Count the number of images in the target subdirectory
            existing_images = [f for f in os.listdir(target_subdir) if f.endswith('.ppm')]
            num_existing_images = len(existing_images)

            # Determine the number of augmentations based on the number of existing images
            num_augmentations = determine_augmentation_count(num_existing_images)

            # List to collect all new metadata rows for the directory
            new_metadata_list = []

            # Process each image in the directory
            for file in files:
                if file.endswith('.ppm'):
                    # Load image
                    img_path = os.path.join(root, file)
                    img = load_image(img_path)

                    # Copy original image to the new directory
                    new_img_path = os.path.join(target_dir, relative_path, file)
                    shutil.copy(img_path, new_img_path)

                    # Get image metadata
                    metadata = df[df['Filename'] == file]

                    # Append the original image's metadata to the new metadata list
                    new_metadata_list.append(metadata.copy())

                    # Create an iterator for augmentation
                    aug_iter = datagen.flow(img, batch_size=1)

                    # Generate the determined number of augmented images
                    for i in range(num_augmentations):
                        # Generate augmented image
                        img_aug = next(aug_iter)[0].astype('uint8')

                        # Save augmented image
                        new_filename = f"{os.path.splitext(file)[0]}_aug_{i}.ppm"
                        new_aug_img_path = os.path.join(target_dir, relative_path, new_filename)
                        cv2.imwrite(new_aug_img_path, cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))

                        # Adjust bounding box (adjustments may be needed based on augmentations)
                        new_metadata = metadata.copy()
                        new_metadata['Filename'] = new_filename
                        new_metadata_list.append(new_metadata)

            # After processing all images, append all new metadata to the DataFrame
            if new_metadata_list:
                new_metadata_df = pd.concat(new_metadata_list, ignore_index=True)
                df = pd.concat([df, new_metadata_df], ignore_index=True)

            # Save updated CSV
            new_csv_path = os.path.join(target_dir, os.path.relpath(csv_path, source_dir))
            df.to_csv(new_csv_path, index=False, sep=';')
