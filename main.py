import os
import torch
import pickle
import argparse
from src.model import *
from utils.env import set_seed
from utils.trainer import train
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from data import dataloader, preprocess_data

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--api_key', default = "1fd701d8a3e21bbdd410eee7423717d41e08db4eae4eb0c57c8c9081fd3861da", type = str)
	parser.add_argument('--batch_size', default = 16, type = int)
	parser.add_argument('--create_image_label_mapping', default = False, type = bool)
	parser.add_argument('--dataset_name', default = "umd_mali_crop_type", type = str)
	parser.add_argument('--fetch_and_extract_data', default = False, type = bool)
	parser.add_argument('--images_path', default = "./dataset/images.pkl", type = str)
	parser.add_argument('--labels_path', default = "./dataset/labels.pkl", type = str)
	parser.add_argument('--model_name', default = "FCN8", type = str)
	parser.add_argument('--num_classes', default = 5, type = int)
	parser.add_argument('--num_epochs', default = 10, type = int)
	parser.add_argument('--labels_dir', default = "./dataset/umd_mali_crop_type_labels", type = str)
	parser.add_argument('--save_dir', default = "./dataset", type = str)
	parser.add_argument('--seed', default = 42, type = int)
	parser.add_argument('--train_shuffle', default = True, type = bool)
	parser.add_argument('--test_shuffle', default = False, type = bool)
	parser.add_argument('--train_test_split', default = 0.8, type = float)
	args = parser.parse_args()
	
	set_seed(args.seed)
	
	device = "cpu"
	if torch.cuda.is_available(): 
		device = "cuda"
	
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
	train_dataloader = DataLoader(train_data, batch_size = args.batch_size, shuffle = args.train_shuffle)

	test_data = dataloader.CropMappingDataset(test_images_path, train_labels_path)
	test_dataloader = DataLoader(test_data, batch_size = args.batch_size, shuffle = args.test_shuffle)

	model = None

	if args.model_name == "FCN8":
		model = FCN8(num_classes = args.num_classes)
	elif args.model_name == "FCN16":
		model = FCN16(num_classes = args.num_classes)

	if torch.cuda.device_count() > 1:
		print("More than one GPUs available")
		model = nn.DataParallel(model, device_ids = None)

	model = model.to(device)

	optimizer = Adam(model.module.parameters())

	if args.model_name.startswith('FCN'):
		optimizer = SGD(model.module.parameters(), 1e-4, .9, 2e-5)
	if args.model_name.startswith('PSP'):
		optimizer = SGD(model.module.parameters(), 1e-2, .9, 1e-4)
	if args.model_name.startswith('Seg'):
		optimizer = SGD(model.module.parameters(), 1e-3, .9)

	train(args, model, train_dataloader = train_dataloader, test_dataloader = test_dataloader, optimizer = optimizer, device = device)