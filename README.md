# Crop Production Mapping

The following codebase is an attempt to implement the task of semantic segmentation for mapping crop production. 

In the context of climate change, land cover is a critical variable. Crop type information is especially important for understanding the spatial distribution of water use and anticipating the likelihood of water scarcity and, as a result, food insecurity. This is true in dry locations like Mali, where agriculture is primarily reliant on irrigation. Remote sensing is useful for mapping crop kinds in this case, but the quality of the data relies on reliable ground-truth data.

Furthermore, crop mapping is critical for agricultural management, economic development, and environmental conservation. We need to develop a viable technique for large-scale crop mapping using remote sensing with a huge field of view. However, most past research has concentrated on multi-temporal crop mapping, which necessitates multiple imaging over time, which is impractical to conduct during the gloomy season due to the lack of clear atmospheric windows, and thus reducing the generalisation ability of the models due to scarcity of data.

In this term project, we approach the problem of mapping crop production using Deep Learning based approaches. The task of mapping crop production falls under the
broad category of semantic segmentation when deep learning approaches come into the picture. Our focus mainly lies on capturing the climate change using the crop type information. Owing to this reason, we choose to build a crop mapping (semantic segmentation) framework on the data named [2019 Mali CropType Training Data](https://mlhub.earth/data/umd_mali_crop_type), which is made publicly available by  [Radiant MLHub](https://mlhub.earth/). This dataset produced by the NASA Harvest team includes crop types labels from ground referencing matched with time-series of Sentinel-2 imagery during the growing season. Ground reference data are collected using an ODK app. Crop types include Maize, Millet, Rice and Sorghum. Labels are vectorized over the Sentinel-2 grid, and provided as raster files. Funding for this dataset is provided by Lutheran World Relief, Bill Melinda Gates Foundation, and University of Maryland NASA Harvest program.

### File Description

This repository contains three directories.

The following files are present in the directory ```data```.
- [dataloader.py](./data/dataloader.py): Script to create a custom Dataloader for the Crop Production Mapping dataset. The class ```CropMappingDataset``` uses the list of images and labels to return the images and labels as pytorch tensors.
- [preprocess_data.py](./data/preprocess_data.py): Script having the following functionalities.
  - Fetch and Extract the required dataset and store in the local directory.
  - Create the images and labels path lists to be used in future efficiently.

The following file is present in the directory ```src```.
- [model.py](./src/model.py): Script to create different model architectures for the task of semantic segmentation. The following architectures are implemented.
  - A fully convolutional architecture with 8 layers over VGG 16 baseline architecture.
  - A fully convolutional architecture with 16 layers over VGG 16 baseline architecture.
  - A fully convolutional architecture with 32 layers over VGG 16 baseline architecture.
  - UNet architecture class used for semantic segmentation.
  - SegNet architecture class used for semantic segmentation task.

The following files are present in the directory ```utils```.
- [criterion.py](./utils/criterion.py): Script to define the loss function as Cross Entropy over 2D tensors (Cross entropy between actual segmentation mask and predicted segmentation mask).
- [env.py](./utils/env.py): Script to define the global random seed environment for the sake of reproducibility.
- [trainer.py](./utils/trainer.py): Script to train the model and evaluate the model by calculating the IoU score. 

### Experiments

To run the code with default arguments: ```python ./main.py``` 

To run the code with user-defined arguments: ```python ./main.py --api_key <API key> --batch_size 16 --create_image_label_mapping False --dataset_name umd_mali_crop_type --fetch_and_extract_data False --images_path ./dataset/images.pkl --labels_dir ./dataset/umd_mali_crop_type_labels --labels_path ./dataset/label.pkl --model_name FCN8 --num_classes 5 --num_epochs 10 --save_dir ./dataset --seed 0 --test_shuffle False --train_shuffle True --train_test_split 0.8```
