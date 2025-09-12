import cv2
import json 
from data import extract_data, load_raw_images

raw_folder = '../data/raw'

extract_data(raw_folder)

raw_cat_images, raw_dog_images = load_raw_images()

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
        
total_cats = process_raw_images(raw_cat_images, "cat", "../data/processed")
total_imgs = process_raw_images(raw_dog_images, "dog", "../data/processed", total_cats)

metadata = {
    "num_images": total_imgs,
    "dimension": [224, 224],
    "format": "jpg"
}

save_metadata(metadata) 