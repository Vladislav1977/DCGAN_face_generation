import os
import time
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from PIL import Image
from pathlib import Path




stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

class MyDataset(Dataset):
	def __init__(self, dataset, transform):
		super().__init__()
		self.dataset = dataset
		self.transform = transform

	def __getitem__(self, index):
		x = Image.open(self.dataset[index])
		x = self.transform(x)
		return x, 0

	def __len__(self):
		return len(self.dataset)


def get_dataloader(image_size, path_dataset, batch_size, folder=False):
	"""
  Builds dataloader for training data.
  Use tt.Compose and tt.Resize for transformations
  :param image_size: height and wdith of the image
  :param batch_size: batch_size of the dataloader
  :returns: DataLoader object
  """

	transform = tt.Compose([
		tt.Resize(image_size),
		tt.CenterCrop(image_size),
		tt.ToTensor(),
		tt.Normalize(*stats)])

	if folder:
		dataset = ImageFolder(path_dataset, transform=transform)
	else:
		path_omg = list(Path(path_dataset).rglob("*.png"))
	# custom dataset только для KNN, при обчении модели использовался ImageFolder
		dataset = MyDataset(path_omg, transform)

	train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return train
