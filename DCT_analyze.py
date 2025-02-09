import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from random_image import get_random_image_from_folder

FOLDER_A = './generated/'
FOLDER_B = './original_compressed/'
IMAGE_PATHS = [get_random_image_from_folder(FOLDER_A), get_random_image_from_folder(FOLDER_B)]
BLOCK_SIZE = 8
COLORMAP = 'hot'

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def extract_dct_coefficients(image_path):
    logging.info(f"Loading image: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logging.error(f"Failed to load image: {image_path}")
        return None

    h, w = img.shape
    dct_coeffs = np.zeros((h, w))

    # Compute DCT for 8x8 blocks
    for i in range(0, h, BLOCK_SIZE):
        for j in range(0, w, BLOCK_SIZE):
            block = img[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            dct_block = cv2.dct(np.float32(block))
            dct_coeffs[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = dct_block

    return dct_coeffs


def extract_high_frequencies(dct_coeffs):
    """ Extracts high-frequency components (upper-right part of 8x8 DCT blocks) """
    h, w = dct_coeffs.shape
    high_freq = np.zeros_like(dct_coeffs)

    for i in range(0, h, BLOCK_SIZE):
        for j in range(0, w, BLOCK_SIZE):
            block = dct_coeffs[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            high_freq[i:i+BLOCK_SIZE//2, j+BLOCK_SIZE//2:j +
                      BLOCK_SIZE] = block[BLOCK_SIZE//2:, BLOCK_SIZE//2:]

    return high_freq


def process_images(image_paths):
    dct_data = []
    std_devs = []
    high_freq_data = []
    high_freq_energy = []

    for image_path in image_paths:
        dct_coeffs = extract_dct_coefficients(image_path)

        if dct_coeffs is None:
            dct_data.append(None)
            std_devs.append(None)
            high_freq_data.append(None)
            high_freq_energy.append(None)
            continue

        std_dev = np.std(dct_coeffs)
        high_freq = extract_high_frequencies(dct_coeffs)
        energy = np.sum(np.abs(high_freq)) / np.sum(np.abs(dct_coeffs))

        dct_data.append(dct_coeffs)
        std_devs.append(std_dev)
        high_freq_data.append(high_freq)
        high_freq_energy.append(energy)

    return dct_data, std_devs, high_freq_data, high_freq_energy


# Process images
dct_data, std_devs, high_freq_data, high_freq_energy = process_images(
    IMAGE_PATHS)

if all(data is not None for data in dct_data):
    fig, axes = plt.subplots(len(IMAGE_PATHS), 4, figsize=(18, 10))

    for i, (data, std_dev, high_freq, energy) in enumerate(zip(dct_data, std_devs, high_freq_data, high_freq_energy)):
        # 1. DCT visualization
        axes[i, 0].imshow(np.log(np.abs(data) + 1), cmap=COLORMAP)
        axes[i, 0].set_title(f"{IMAGE_PATHS[i]}\nStd Dev: {std_dev:.2f}")

        # 2. Histogram of DCT coefficients
        axes[i, 1].hist(data.ravel(), bins=100, log=True,
                        color='blue', alpha=0.7)
        axes[i, 1].set_title(f"{IMAGE_PATHS[i]}\nDCT Histogram")

        # 3. High-frequency components
        axes[i, 2].imshow(np.log(np.abs(high_freq) + 1), cmap=COLORMAP)
        axes[i, 2].set_title(f"{IMAGE_PATHS[i]}\nHigh-Frequency Components")

        # 4. High-frequency energy (text annotation)
        axes[i, 3].axis('off')
        axes[i, 3].text(0.5, 0.5, f"High-Freq Energy:\n{energy:.4f}", fontsize=14,
                        ha='center', va='center', bbox=dict(facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plt.show()
else:
    logging.error(
        "DCT data is None for one or more images. Visualization skipped.")
