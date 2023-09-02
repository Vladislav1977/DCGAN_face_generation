import os
import time
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torchvision.utils import save_image
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import os
from model import *
from data import *
from utils import *

import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
batch_size = 128
latent_size = 128

def fit(model, optimizer, epochs, lr, Buffer, save=False, name = None, mode="kaggle"):

	model["discriminator"].train()
	model["generator"].train()
	torch.cuda.empty_cache()

	train_dl = get_dataloader((128, 128), batch_size)

	# Losses & scores
	losses_g = []
	losses_d = []
	real_scores = []
	fake_scores = []

	criterion = nn.BCEWithLogitsLoss()

	for epoch in range(epochs):

		tic = time.time()
		loss_d_per_epoch = []
		loss_g_per_epoch = []
		real_score_per_epoch = []
		fake_score_per_epoch = []
		for real_images, _ in tqdm(train_dl):
			real_images = real_images.to(device)
			# Train discriminator

			# Pass real images through discriminator
			real_preds = model["discriminator"](real_images)

			real_targets = torch.ones(real_images.size(0), 1, device=device)
			smooth = torch.empty_like(real_targets).uniform_(0, 0.1)
			real_targets_smooth = real_targets - smooth

			real_loss = criterion(real_preds, real_targets_smooth)
			cur_real_score = torch.mean(real_preds).item()

			# Generate fake images
			latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
			fake_images = model["generator"](latent)
			fake_images_buff = Buffer.extract(fake_images.detach())

			# Pass fake images through discriminator
			fake_targets = torch.zeros(fake_images.size(0), 1, device=device)

			fake_preds = model["discriminator"](fake_images_buff)  # detach added
			fake_loss = criterion(fake_preds, fake_targets)
			cur_fake_score = torch.mean(fake_preds).item()

			real_score_per_epoch.append(cur_real_score)
			fake_score_per_epoch.append(cur_fake_score)

			# Update discriminator weights
			optimizer["discriminator"].zero_grad()

			loss_d = real_loss + fake_loss
			loss_d.backward()
			optimizer["discriminator"].step()
			loss_d_per_epoch.append(loss_d.item())

			# Train generator

			# Try to fool the discriminator
			preds = model["discriminator"](fake_images)
			targets = torch.ones(batch_size, 1, device=device)
			loss_g = criterion(preds, targets)

			# Update generator weights
			optimizer["generator"].zero_grad()
			loss_g.backward()
			optimizer["generator"].step()

			loss_g_per_epoch.append(loss_g.item())

		#       scheduller["generator"].step()
		#       scheduller["discriminator"].step()

		toc = time.time()
		min = (toc - tic) // 60
		sec = round((toc - tic) % 60, 2)
		print("Train time:", min, "min", sec, "sec")
		# Record losses & scores
		losses_g.append(np.mean(loss_g_per_epoch))
		losses_d.append(np.mean(loss_d_per_epoch))
		real_scores.append(np.mean(real_score_per_epoch))
		fake_scores.append(np.mean(fake_score_per_epoch))

		# Log losses & scores (last batch)
		print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
			epoch +1, epochs,
			losses_g[-1], losses_d[-1], real_scores[-1], fake_scores[-1]))

		# Save generated images
		if save == True:
			if mode == "collab":
				path_g = f"/content/gdrive/MyDrive/models/GAN/{name[0]}"
				path_d = f"/content/gdrive/MyDrive/models/GAN/{name[1]}"
			else:
				path_g = f"./{name[0]}"
				path_d = f"./{name[1]}"
			stat = losses_g, losses_d, real_scores, fake_scores
			with open('stat.pkl', 'wb') as file:
				pickle.dump(stat, file)

			check_gen = {
				"model_dict": model["generator"].state_dict(),
				"optimizer": optimizer["generator"].state_dict(),
			}
			check_disc = {
				"model_dict": model["discriminator"].state_dict(),
				"optimizer": optimizer["discriminator"].state_dict(),
			}
			torch.save(check_gen, path_g)
			torch.save(check_disc, path_d)
		# Save generated images

		print(show_images(fake_images))

	return losses_g, losses_d, real_scores, fake_scores

def train():

	netG = Generator().to(device)
	netD = Discriminator().to(device)

	model = {
		"discriminator": netD,
		"generator": netG
	}
	lr = 0.0001

	optimizer = {
		"discriminator": torch.optim.Adam(model["discriminator"].parameters(),
										  lr=lr, betas=(0.5, 0.999)),
		"generator": torch.optim.Adam(model["generator"].parameters(),
									  lr=lr, betas=(0.5, 0.999))
	}

	epochs = 50

	checkpoint_gen = torch.load("/kaggle/input/model-stat-new/generator_50_55.pth", map_location=device)
	checkpoint_disc = torch.load("/kaggle/input/model-stat-new/discriminator_50_55.pth", map_location=device)

	model["generator"].load_state_dict(checkpoint_gen["model_dict"])
	optimizer["generator"].load_state_dict(checkpoint_gen["optimizer"])

	model["discriminator"].load_state_dict(checkpoint_disc["model_dict"])
	optimizer["discriminator"].load_state_dict(checkpoint_disc["optimizer"])