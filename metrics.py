import torch
import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from model import Generator

from train import device, batch_size, latent_size
from data import get_dataloader



def collect_img(model, dl):
	m_l = 30
	lis = []

	for i, data in tqdm(enumerate(dl), total=m_l):
		data = data[0].to(device)
		if i == 0:
			real_images = data
		elif i == m_l:
			break
		else:
			real_images = torch.cat((real_images, data))

	laten_vector = torch.randn(real_images.shape[0], latent_size, 1, 1, device=device)
	with torch.no_grad():
		fake_images = model(laten_vector)

	images = torch.cat((real_images, fake_images)).flatten(start_dim=1)

	real_label = torch.ones(real_images.shape[0], device=device)
	fake_label = torch.zeros(fake_images.shape[0], device=device)
	labels = torch.cat((real_label, fake_label))
	return images, labels


def accuracy_KNN(model, path_gen, path_data):

	checkpoint_gen = torch.load(path_gen, map_location=device)
	model.load_state_dict(checkpoint_gen["model_dict"])

	train_dl = get_dataloader((128, 128), path_data, batch_size)

	print("Collecting images")
	images, labels = collect_img(model, train_dl)

	loo = LeaveOneOut()
	accuracy = []

	print("Metric evaluation")
	for train_idx, test_idx in tqdm(loo.split(images), total=images.shape[0]):
		x_train, y_train = images[train_idx], labels[train_idx]
		x_test, y_test = images[test_idx], labels[test_idx]

		X_i = x_test[:, None, :]  # (1, 1, 49152) test set
		X_j = x_train[None, :, :]  # (1, 7679, 49152) train set

		D_ij = ((X_i - X_j) ** 2).sum(-1)  # (1, 7679)
		ind_knn = D_ij.argmin(axis=1)
		lab_knn = y_train[ind_knn]
		pred = (lab_knn == y_test).item()
		accuracy.append(pred)
	total_acc = np.array(accuracy).mean()
	print("Accuracy:", round(total_acc, 3))



def tsne(model, path_gen, path_data, plot=False):
	checkpoint_gen = torch.load(path_gen, map_location=device)
	model.load_state_dict(checkpoint_gen["model_dict"])

	train_dl = get_dataloader((128, 128), path_data, batch_size)
	print("Collecting images")
	images, labels = collect_img(model, train_dl)

	images = images.detach().cpu()
	labels = labels.detach().cpu()

	print("Evaluating distribution")
	images_tse = TSNE(n_components=2).fit_transform(images)
	# print(images_tse)
	# print(labels)

	if plot:
		plt.figure(figsize=(8, 8))
		scatter = plt.scatter(images_tse[:, 0], images_tse[:, 1], c=labels, cmap='gist_rainbow')
		plt.legend(handles=scatter.legend_elements()[0],
				   title="imgs",
				   labels=["real", "fake"])
		plt.savefig("tsne.png")

	return images_tse, labels