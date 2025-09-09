# from 4.0-data-cleaning-and-collection.ipynb
from zipfile import ZipFile
import os
import shutil
import matplotlib.pyplot as plt
import random
import cv2
import json

raw_folder = '../data/raw'

def extract_data(folder):
    zip_file_path = os.path.join(folder, 'dog-and-cat-classification-dataset.zip')
    extract_to_path = folder

    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)

    # Move files from PetImages to the raw_folder
    source_dir = os.path.join(folder, 'PetImages')
    for item in os.listdir(source_dir):
        source_item_path = os.path.join(source_dir, item)
        destination_item_path = os.path.join(folder, item)
        shutil.move(source_item_path, destination_item_path)

    os.rmdir(source_dir)
    
def load_raw_images():
    raw_cat_images = []
    raw_dog_images = []
    
    for filename in os.listdir(raw_folder + '/Cat'):
        img = cv2.imread(raw_folder + '/Cat/' + filename)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw_cat_images.append(img)
        
    for filename in os.listdir(raw_folder + '/Dog'):
        img = cv2.imread(raw_folder + '/Dog/' + filename)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        raw_dog_images.append(img)
        
    return (raw_cat_images, raw_dog_images)

def visualize_raw_pet_images(cat_images, dog_images):
    _, ax = plt.subplots(5, 5, figsize=(8, 8))
    for i in range(5):
        for j in range(5):
            is_cat = random.randint(0, 1)
            cat_images_len = len(cat_images)
            dog_images_len = len(dog_images)
            
            label = "Cat" if is_cat else "Dog"
            
            ax[i, j].imshow(cat_images[random.randint(0, cat_images_len - 1)] if is_cat
                            else dog_images[random.randint(0, dog_images_len - 1)])
            ax[i, j].axis("off")
            ax[i, j].set_title(label)

def visualize_pet_images(images, labels):
    _, ax = plt.subplots(2, 8, figsize=(16, 4))
    img_idx = 0
    for i in range(2):
        for j in range(8):    
            ax[i, j].imshow(images[img_idx])
            ax[i, j].axis("off")
            ax[i, j].set_title(labels[img_idx])
            img_idx += 1
            
def process_image (img, filename, dimension=(224, 224)):
    resized_image = cv2.resize(img, dimension, cv2.INTER_AREA)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename, resized_image)
    
def process_raw_images(images, label, folderpath, counter=0, dimension=(224, 224)):
    for i, img in enumerate(images):
        process_image(img, folderpath + f"/{counter}-{label}.jpg", dimension)
        counter += 1
    return counter
def save_metadata(metadata, outpath="../data/processed/metadata.json"):
    with open(outpath, "w") as f:
        json.dump(metadata, f, indent=4)
        
class CatAndDogDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        with open("../data/processed/metadata.json", "r") as f:
            metadata = json.load(f)
        self.metadata = metadata
        self.img_dir = img_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")],
                                key=lambda x: int(x.split('-')[0])) # Sort files by their numerical ID
        
        self.transform = transform
        self.target_transform = target_transform
        self.class_map = {"cat": 0, "dog": 1}
    
    def __len__(self):
        return self.metadata['num_images']
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # load image with cv2
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR
        
        # parse label from filename ("1-cat.jpg" -> "cat")
        label_str = img_name.split("-")[1].split(".")[0]  # "cat" or "dog"
        label = self.class_map[label_str]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label