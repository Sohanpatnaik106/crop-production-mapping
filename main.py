import os
import pickle
import argparse
from utils.env import set_seed
from torch.utils.data import DataLoader
from data import dataloader, preprocess_data

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--api_key', default = "1fd701d8a3e21bbdd410eee7423717d41e08db4eae4eb0c57c8c9081fd3861da", type = str)
	parser.add_argument('--batch_size', default = 128, type = int)
	parser.add_argument('--create_image_label_mapping', default = False, type = bool)
	parser.add_argument('--dataset_name', default = "umd_mali_crop_type", type = str)
	parser.add_argument('--fetch_and_extract_data', default = False, type = bool)
	parser.add_argument('--images_path', default = "./dataset/images.pkl", type = str)
	parser.add_argument('--labels_path', default = "./dataset/labels.pkl", type = str)
	parser.add_argument('--save_dir', default = "./dataset", type = str)
	parser.add_argument('--labels_dir', default = "./dataset/umd_mali_crop_type_labels", type = str)
	parser.add_argument('--seed', default = 42, type = int)
	parser.add_argument('--train_shuffle', default = True, type = bool)
	parser.add_argument('--test_shuffle', default = False, type = bool)
	parser.add_argument('--train_test_split', default = 0.8, type = float)
	args = parser.parse_args()
	
	set_seed(args.seed)
	
	if args.fetch_and_extract_data:
		preprocess_data.fetch_and_extract_data(dataset_name = args.dataset_name, api_key = args.api_key, save_dir = args.save_dir)
	
	if args.create_image_label_mapping:
		preprocess_data.create_image_label_mapping(args.save_dir, args.labels_dir)
	


	assert (os.path.exists(args.images_path) and os.path.exists(args.labels_path)), "Please create image label mapping"
		 
	images_path = pickle.load(open(args.images_path, "rb"))
	labels_path = pickle.load(open(args.labels_path, "rb"))

	train_images_path = images_path[: int(args.train_test_split * len(images_path))]
	train_labels_path = labels_path[: int(args.train_test_split * len(labels_path))]

	test_images_path = images_path[int(args.train_test_split * len(images_path)): ]
	test_labels_path = labels_path[int(args.train_test_split * len(labels_path)): ]

	train_data = dataloader.CropMappingDataset(train_images_path, train_labels_path)
	train_dataloder = DataLoader(train_data, batch_size = args.batch_size, shuffle = args.train_shuffle)

	test_data = dataloader.CropMappingDataset(test_images_path, train_labels_path)
	test_dataloder = DataLoader(test_data, batch_size = args.batch_size, shuffle = args.test_shuffle)


