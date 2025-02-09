import os
import random
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_random_image_from_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    random_image = random.choice(files)

    file_path = os.path.join(folder_path, random_image)

    if not random_image.lower().endswith(('jpeg', 'jpg', 'png', 'gif')):
        logging.error("%s is not an image", random_image)
        
    return file_path
