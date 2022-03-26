import numpy as np
import torch

from PIL import Image

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score

from utils.criterion import CrossEntropyLoss2d

from tqdm import tqdm

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return 

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

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


def train(args, model, train_dataloader, test_dataloader = None, optimizer = None, device = "cuda:0"):

	model.train()
	criterion = CrossEntropyLoss2d()

	for epoch in range(1, args.num_epochs + 1):
		train_epoch_loss = []
		test_epoch_loss = []

		print("Epoch: ", epoch)

		for step, (images, labels) in enumerate(train_dataloader):
			if device == "cuda":
				images = images.cuda()
				labels = labels.cuda()

			outputs = model(images)
			
			optimizer.zero_grad()
			loss = criterion(outputs, labels[:, 0].long())
			loss.backward()
			optimizer.step()

			train_epoch_loss.append(loss.item())

			outputs = torch.argmax(outputs, dim = 1)
			iou_score = iou(outputs, labels)

			print(f"Epoch: {epoch} Batch idx: {step} Train Loss: {loss.item()} IOU Score: {iou_score}")

		print("Evaluating on Test set")

		for step, (images, labels) in enumerate(test_dataloader):
			if device == "cuda":
				images = images.cuda()
				labels = labels.cuda()

			outputs = model(images)
			loss = criterion(outputs, labels[:, 0].long())

			test_epoch_loss.append(loss.item())
			
			labels = labels.detach().cpu().numpy().reshape(-1)
			outputs = outputs.detach().cpu().numpy().reshape(-1)

			iou_score = iou(outputs, labels)

			print(f"Epoch: {epoch} Batch idx: {step} Train Loss: {loss.item()} IOU Score: {iou_score}")

		print("\n\n")