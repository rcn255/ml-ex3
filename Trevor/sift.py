import pickle
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import argparse
import os
from keras.datasets import fashion_mnist

# CIFAR-10 and Fashion MNIST labels
cifar10_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

fashion_mnist_labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Predefined paths for the datasets
cifar10_data_dir = "paul/data/cifar-10-batches-py"
fashion_mnist_data_dir = "paul/data/minst_clothing"

# Load CIFAR-10 or Fashion MNIST data
def load_dataset(dataset_name):
    if dataset_name == 'cifar10':
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        data_batches = [unpickle(os.path.join(cifar10_data_dir, f"data_batch_{i}")) for i in range(1, 6)]
        test_batch = unpickle(os.path.join(cifar10_data_dir, "test_batch"))

        train_data = np.concatenate([batch[b'data'] for batch in data_batches])
        train_labels = np.concatenate([batch[b'labels'] for batch in data_batches])
        test_data = test_batch[b'data']
        test_labels = test_batch[b'labels']

        train_data = train_data.reshape((len(train_data), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_data = test_data.reshape((len(test_data), 3, 32, 32)).transpose(0, 2, 3, 1)

        labels = cifar10_labels

    elif dataset_name == 'fashion_mnist':
        (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
        train_data = np.expand_dims(train_data, axis=-1).repeat(3, axis=-1)  # convert to 3 channels
        test_data = np.expand_dims(test_data, axis=-1).repeat(3, axis=-1)
        labels = fashion_mnist_labels
    else:
        raise ValueError("Unsupported dataset. Please choose 'cifar10' or 'fashion_mnist'.")

    return (train_data, train_labels), (test_data, test_labels), labels

# Convert images to uint8
def convert_to_uint8(image):
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

# Extract SIFT features
def extract_sift_features(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    valid_indices = []

    start_time = time.time()  # Start timing

    for idx, img in enumerate(images):
        img = convert_to_uint8(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img_gray, None)
        if descriptors is not None and len(descriptors) > 0:
            descriptors_list.append(descriptors)
            valid_indices.append(idx)

    end_time = time.time()  # End timing
    print(f"SIFT feature extraction time: {end_time - start_time} seconds")

    return descriptors_list, valid_indices

# Create BoW histograms
def create_bow_histograms(descriptors_list, kmeans):
    histograms = []
    for descriptors in descriptors_list:
        words = kmeans.predict(descriptors)
        histogram, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1), density=True)
        histograms.append(histogram)
    return np.array(histograms)

def main(dataset_name, num_clusters):
    (train_data, train_labels), (test_data, test_labels), labels = load_dataset(dataset_name)

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    train_descriptors_list, train_valid_indices = extract_sift_features(train_data)
    test_descriptors_list, test_valid_indices = extract_sift_features(test_data)

    # Flatten the list of descriptors for k-means clustering
    all_descriptors = np.vstack(train_descriptors_list + test_descriptors_list)

    # Perform k-means clustering to create the visual words
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(all_descriptors)

    # Save the visual word vocabulary
    vocabulary = kmeans.cluster_centers_

    train_bow_histograms = create_bow_histograms(train_descriptors_list, kmeans)
    test_bow_histograms = create_bow_histograms(test_descriptors_list, kmeans)

    # Normalize the histograms
    scaler = StandardScaler()
    train_bow_histograms = scaler.fit_transform(train_bow_histograms)
    test_bow_histograms = scaler.transform(test_bow_histograms)

    # Train an SVM classifier
    svm = SVC(kernel='linear', random_state=0)
    svm.fit(train_bow_histograms, train_labels[train_valid_indices])

    # Predict and evaluate on test data
    test_predictions = svm.predict(test_bow_histograms)
    print("BoW SVM Classification Report:")
    print(classification_report(test_labels[test_valid_indices], test_predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(test_labels[test_valid_indices], test_predictions))

    # Plot confusion matrix
    cm = confusion_matrix(test_labels[test_valid_indices], test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Count the most frequent class labels for each visual word
    visual_word_labels = {i: [] for i in range(num_clusters)}

    for i, descriptors in enumerate(train_descriptors_list):
        words = kmeans.predict(descriptors)
        most_frequent_word = np.argmax(np.bincount(words))
        visual_word_labels[most_frequent_word].append(train_labels[train_valid_indices[i]])

    # Find the most frequent class label for each visual word
    visual_word_class_labels = {}
    for word, labels in visual_word_labels.items():
        if labels:
            most_common_label = Counter(labels).most_common(1)[0][0]
            visual_word_class_labels[word] = cifar10_labels[most_common_label]
        else:
            visual_word_class_labels[word] = 'Unknown'

    # Visualize the frequency of visual words with their labels
    word_counts = np.bincount(kmeans.labels_)
    visual_word_labels = [f'Word {i}' for i in range(num_clusters)]

    # Adjust figure size for the plot
    fig, ax = plt.subplots(figsize=(20, 10))
    bars = ax.bar(range(num_clusters), word_counts)
    ax.set_xlabel('Visual Word Index', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Complete Vocabulary Generated', fontsize=16)

    # Keep x-axis labels simple
    ax.set_xticks(range(num_clusters))
    ax.set_xticklabels(visual_word_labels, rotation=45, ha='right', fontsize=12)

    # Prepare the index/legend in a separate table
    index_table = [[f'Word {i}', visual_word_class_labels[i]] for i in range(num_clusters)]

    # Create a new figure for the index table
    fig2, ax2 = plt.subplots(figsize=(20, 10))
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(cellText=index_table, colLabels=['Visual Word Index', 'Most Frequent Label'], cellLoc='center', loc='center')

    # Display both plots
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 and Fashion MNIST Feature Extraction and BoW Model using SIFT and SVM")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'fashion_mnist'], help='Dataset to use (cifar10 or fashion_mnist)')
    parser.add_argument('--num_clusters', type=int, default=100, help='Number of clusters for k-means')
    args = parser.parse_args()

    main(args.dataset, args.num_clusters)
