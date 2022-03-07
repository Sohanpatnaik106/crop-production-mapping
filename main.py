import argparse
from data import dataloader, preprocess_data
# from utils.env import set_seed





if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--api_key', default = "1fd701d8a3e21bbdd410eee7423717d41e08db4eae4eb0c57c8c9081fd3861da", type = str)
	parser.add_argument('--batch_size', default = 128, type = int)
	parser.add_argument('--dataset_name', default = "umd_mali_crop_type", type = str)
	parser.add_argument('--preprocess_data', default = False, type = bool)
	parser.add_argument('--save_dir', default = "./dataset", type = str)
	parser.add_argument('--seed', default = 42, type = int)
	args = parser.parse_args()
	
	if args.preprocess_data:
		preprocess_data.preprocess(dataset_name = args.dataset_name, api_key = args.api_key, save_dir = args.save_dir)

	

	set_seed(args.seed)
	