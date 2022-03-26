"""
    Following codebase is the implementation of semantic segmentation task for 
    Crop Production Mapping on the dataset made available by "Radiant MLHub".
	The link to the dataset is https://mlhub.earth/data/umd_mali_crop_type
    Five different models, namely, Fully Convolutional Network with 8, 16 and 32 layers
	and UNet and SegNet architectures are used to carry out the experiments. To evaluate the models, 
	IoU or Jaccard Similarity is measured over the actual and predicted segmented maps.
"""

# Import the required libraries
import torch
from utils.criterion import CrossEntropyLoss2d

# Function to take channels, used for calculating IoU score
def _take_channels(*xs, ignore_channels = None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim = 1, index = torch.tensor(channels).to(x.device)) for x in xs]
        return 

# Function to modify x based on the threshold
def _threshold(x, threshold = None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

# Function to calculate the IoU score
def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union

# Function to train the model
def train(args, model, train_dataloader, test_dataloader = None, optimizer = None, device = "cuda:0"):

	# Set the model in training model
	model.train()

	# Initialise the criterion
	criterion = CrossEntropyLoss2d()

	# Iterate over the number of epochs
	for epoch in range(1, args.num_epochs + 1):

		# Declare lists to store the train and test losses over the epochs
		train_epoch_loss = []
		test_epoch_loss = []

		print("Epoch: ", epoch)

		# Iterate over the batches in each epoch
		for step, (images, labels) in enumerate(train_dataloader):
			
			# Transfer the data to GPU is available
			if device == "cuda":
				images = images.cuda()
				labels = labels.cuda()

			# Obtain the outputs
			outputs = model(images)
			
			# Backpropagate and update the parameters of the model
			optimizer.zero_grad()
			loss = criterion(outputs, labels[:, 0].long())
			loss.backward()
			optimizer.step()

			# Append the train loss
			train_epoch_loss.append(loss.item())

			# Calculate the IoU score
			outputs = torch.argmax(outputs, dim = 1)
			iou_score = iou(outputs, labels)

			print(f"Epoch: {epoch} Batch idx: {step} Train Loss: {loss.item()} IOU Score: {iou_score}")

		print("Evaluating on Test set")
		
		# Evaluate on the test set
		for step, (images, labels) in enumerate(test_dataloader):

			if device == "cuda":
				images = images.cuda()
				labels = labels.cuda()

			outputs = model(images)
			loss = criterion(outputs, labels[:, 0].long())

			test_epoch_loss.append(loss.item())
		
			outputs = torch.argmax(outputs, dim = 1)
			iou_score = iou(outputs, labels)

			print(f"Epoch: {epoch} Batch idx: {step} Train Loss: {loss.item()} IOU Score: {iou_score}")

		print("\n\n")