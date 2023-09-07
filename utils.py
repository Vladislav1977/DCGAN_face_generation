import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torch


def denorm(img_tensors, stats):
    return img_tensors * stats[1][0] + stats[0][0]


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach().cpu()[:nmax]), nrow=8).permute(1, 2, 0))
    plt.show()


def eval(model, load=False, path_gen=None, device=torch.device('cuda')):
    if load:
        checkpoint_gen = torch.load(path_gen, map_location=device)
        model.load_state_dict(checkpoint_gen["model_dict"])

    model.eval()
    rand = torch.randn(64, 128, 1, 1, device=device)
    out = model(rand)

    print(show_images(out))