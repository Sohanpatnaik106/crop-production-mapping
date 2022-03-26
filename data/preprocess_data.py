import os
import json
import pickle
import tarfile
import numpy as np
from tqdm import tqdm
from PIL import Image
from radiant_mlhub import Collection

def fetch_and_extract_data(dataset_name = "umd_mali_crop_type", api_key = None, save_dir = "./dataset"):

    print(f"\nFetching dataset {dataset_name}\n")
    dataset_source = Collection.fetch(dataset_name + "_source", api_key = api_key)
    dataset_labels = Collection.fetch(dataset_name + "_labels", api_key = api_key)
    source_archive = dataset_source.download(save_dir, api_key = api_key)
    labels_archive = dataset_labels.download(save_dir, api_key = api_key)
    
    print(f"\nExtracting dataset {source_archive}\n")
    source_file = tarfile.open(source_archive)
    source_file.extractall(save_dir)
    source_file.close()
    
    print(f"\nExtracting dataset {labels_archive}\n")
    labels_file = tarfile.open(labels_archive)
    labels_file.extractall(save_dir)
    labels_file.close()

    print("Dataset fetched successfully\n")

def create_image_label_mapping(save_dir, labels_dir):

    labels = os.listdir(labels_dir)
    labels = sorted(labels)
    labels = labels[2:]

    labels_path_list = []
    images_path_list = []

    for label in labels:
        label_map_path = os.path.join(labels_dir, label, "labels.tif")
        stac_path = os.path.join(labels_dir, label, "stac.json")
        stac = json.load(open(stac_path, "r"))
        links = stac["links"]
        links = links[4:]
        for link in links:
            image_dir = link["href"]
            image_dir = image_dir[6:-10]
            image_path = os.path.join(save_dir, image_dir, "B01.tif")
            image = Image.open(image_path)


            images_path_list.append(image_path)
            labels_path_list.append(label_map_path)


    pickle.dump(images_path_list, open(os.path.join(save_dir, "images.pkl"), "wb"))
    pickle.dump(labels_path_list, open(os.path.join(save_dir, "labels.pkl"), "wb"))

    print("\nCreated Images and Labels mapping")