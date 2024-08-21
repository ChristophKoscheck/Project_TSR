import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage import io, transform, feature
from skimage.transform import resize
from sklearn.preprocessing import normalize
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from sklearn.cluster import KMeans

resolution = 128
sample_image_num = 3

# Function Calls
def main():
    # Load training and testing datasets
    ROOT_PATH = "./"
    train_data_dir = os.path.join(ROOT_PATH, "Training")
    test_data_dir = os.path.join(ROOT_PATH, "Testing")

    train_images, train_labels, train_paths = load_data(train_data_dir)
    test_images, test_labels, test_paths = load_data(test_data_dir)

    # Create a DataFrame for labels and paths
    labels_df = pd.DataFrame({
        'Path': np.concatenate([train_paths, test_paths]),
        'ClassId': np.concatenate([train_labels, test_labels])
    })

    # # Display the first few rows of the dataset
    # print("First few rows of the dataset:")
    # print(labels_df.head())

    # # Display the distribution of classes
    # print("\nDistribution of classes in the dataset:")
    # print(labels_df['ClassId'].value_counts())

    # Plot the distribution of classes
    # plot_class_distribution(labels_df)

    # Plot images for classes
    # plot_images_for_classes(labels_df, train_data_dir, test_data_dir, n_images=sample_image_num)

    # Get and summarize image dimensions
    dimensions_df = get_image_stats(train_data_dir, test_data_dir, labels_df)

    # Plot image dimension and aspect ratio distribution
    # plot_image_stats(dimensions_df)

    # Split the data into training and testing sets
    # train_df, test_df = split_data(labels_df)

    # Calculate and plot vibrant colors
    vibrant_colors = calculate_vibrant_colors(labels_df, train_data_dir, test_data_dir)
    labels_df['VibrantColor'] = vibrant_colors

    # Plot random images with identified vibrant color
    plot_vibrant_color_images(labels_df, train_data_dir, test_data_dir, vibrant_colors, dimensions_df)

    # Extract and plot color histograms
    # plot_color_histograms(labels_df, train_data_dir, test_data_dir)

    # Extract and plot edges
    # plot_edges(labels_df, train_data_dir, test_data_dir)

    # Augment and plot images
    # augment_and_plot_images(labels_df, train_data_dir, test_data_dir)

# Function Definitions
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
                    paths.append(os.path.join(class_dir, f))  # Save relative path for later
    return np.array(images), np.array(labels), np.array(paths)

def plot_class_distribution(labels_df):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='ClassId', data=labels_df, palette='viridis')
    plt.title('Distribution of Traffic Sign Classes')
    plt.xlabel('Class ID')
    plt.ylabel('Number of Images')
    plt.show()

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
                print(f"Image not found: {class_images[j]}")
                continue
            try:
                img = Image.open(img_path)
                plt.subplot(n_rows, n_cols, i * n_images + j + 1)
                plt.imshow(img)
                plt.axis('off')
                if j == 0:
                    plt.title(f'Class {class_id}')
            except FileNotFoundError as e:
                print(f"Error loading image: {e}")
                continue

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.0, left=0.2, right=0.8, hspace=0.87, wspace=0.2)
    plt.show()

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
            print(f"Image not found: {img_path}")
            continue
        try:
            img = Image.open(full_img_path)
            dimensions.append(img.size)
        except FileNotFoundError as e:
            print(f"Error loading image: {e}")
            continue
    
    dimensions_df = pd.DataFrame(dimensions, columns=['Width', 'Height'])
    dimensions_df['AspectRatio'] = dimensions_df['Width'] / dimensions_df['Height']
    
    print("\nSummary statistics for image dimensions:")
    print(dimensions_df.describe())
    
    return dimensions_df

def plot_image_stats(dimensions_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    sns.histplot(dimensions_df['Width'], kde=True, label='Width', ax=ax1)
    sns.histplot(dimensions_df['Height'], kde=True, label='Height', ax=ax1)
    ax1.set_xlabel('Dimension in pixels')
    ax1.set_ylabel('Image count')
    ax1.set_title('Distribution of Image Width and Height')
    ax1.legend()

    sns.histplot(dimensions_df['AspectRatio'], kde=True, label='Aspect Ratio', ax=ax2, color='r')
    ax2.set_xlabel('Aspect Ratio')
    ax2.set_ylabel('Image count')
    ax2.set_title('Distribution of Image Aspect Ratio')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def split_data(labels_df):
    train_df, test_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['ClassId'], random_state=42)

    print("\nNumber of training samples:", len(train_df))
    print("Number of testing samples:", len(test_df))

    # Save the training and test sets for future use
    train_df.to_csv('train_labels.csv', index=False)
    test_df.to_csv('test_labels.csv', index=False)

    # Visualize class imbalance
    plt.figure(figsize=(12, 6))
    sns.countplot(x='ClassId', data=labels_df, palette='viridis')
    plt.title('Class Distribution with Imbalance')
    plt.xlabel('Class ID')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=90)
    plt.show()

    # Check the ratio of the most frequent class to the least frequent class
    class_counts = labels_df['ClassId'].value_counts()
    print(f"Most frequent class has {class_counts.max()} samples.")
    print(f"Least frequent class has {class_counts.min()} samples.")
    print(f"Ratio (most/least): {class_counts.max() / class_counts.min():.2f}")

    return train_df, test_df

def calculate_vibrant_colors(labels_df, train_data_dir, test_data_dir):
    def calculate_vibrant_color(image):
        image = image.convert('RGB')
        np_image = np.array(image)
        np_image = np_image.reshape(-1, 3)
        avg_color = np.mean(np_image, axis=0)
        max_color = np.argmax(avg_color)
        color_names = ['Red', 'Green', 'Blue', 'Yellow']
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
            print(f"Image not found: {img_path}")
            continue
        try:
            img = Image.open(full_img_path)
            vibrant_color = labels_df.loc[labels_df['Path'] == img_path, 'VibrantColor'].values[0]
            plt.subplot(2, 5, idx+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Vibrant Color: {vibrant_color}')
        except FileNotFoundError as e:
            print(f"Error loading image: {e}")

    plt.tight_layout()
    plt.show()

    # Add vibrant color to labels_df
    color_mapping = {'Red': 0, 'Green': 1, 'Blue': 2, 'Yellow': 3}
    labels_df['VibrantColor'] = labels_df['VibrantColor'].map(color_mapping)

    # Add image dimensions to the dataframe
    labels_df['Width'] = dimensions_df['Width']
    labels_df['Height'] = dimensions_df['Height']

    # Calculate aspect ratio
    labels_df['AspectRatio'] = labels_df['Width'] / labels_df['Height']

    # Update the correlation matrix
    correlation = labels_df[['ClassId', 'VibrantColor', 'Width', 'Height', 'AspectRatio']].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Features Including Dominant Colors')
    plt.show()

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
            print(f"Image not found: {img_path}")
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
                plt.plot(hist[:, i].flatten(), color=color, label=f'{color.upper()} Channel')
            plt.xlabel('Bin Number')
            plt.ylabel('Frequency')
            plt.title(f'Color Histogram - Class {sample_images.iloc[idx]["ClassId"]}')
            plt.legend(loc='upper right')
        except FileNotFoundError as e:
            print(f"Error loading image: {e}")
            continue

    plt.tight_layout()
    plt.show()

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
            print(f"Image not found: {img_path}")
            continue
        try:
            img = Image.open(full_img_path)
            edges = extract_edges(img)
            plt.subplot(2, 5, idx+1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.xlabel('Width')
            plt.ylabel('Height')
            plt.subplot(2, 5, idx+6)
            plt.imshow(edges, cmap='binary')
            plt.title(f'Edges - Class {sample_images.iloc[idx]["ClassId"]}')
            plt.xlabel('Width')
            plt.ylabel('Height')
        except FileNotFoundError as e:
            print(f"Error loading image: {e}")
            continue

    plt.tight_layout()
    plt.show()

def augment_and_plot_images(labels_df, train_data_dir, test_data_dir):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    sample_image_train_path = os.path.join(train_data_dir, labels_df.iloc[0]['Path'])
    sample_image_test_path = os.path.join(test_data_dir, labels_df.iloc[0]['Path'])
    if os.path.exists(sample_image_train_path):
        sample_image_path = sample_image_train_path
    elif os.path.exists(sample_image_test_path):
        sample_image_path = sample_image_test_path
    else:
        raise FileNotFoundError(f"Sample image not found in either directory.")

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
        plt.title(f'Augmented {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run the main function to execute the desired parts of the script
if __name__ == "__main__":
    main()
