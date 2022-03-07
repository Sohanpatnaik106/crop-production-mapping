import os
import json
import tarfile
from tqdm import tqdm
from radiant_mlhub import Collection

def preprocess(dataset_name = "umd_mali_crop_type", api_key = None, save_dir = "./dataset"):

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