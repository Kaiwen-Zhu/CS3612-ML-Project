import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt
from config import make_visualization_argparser
from model import MyVAE
from data import get_image_path, load_image


def plot_loss(model_path: str, img_path: str):
    with open(os.path.join(model_path, 'loss.pkl'), 'rb') as f:
        loss_acc = pickle.load(f)
    
    plt.figure()
    plt.plot(loss_acc['train_losses'], label='Training Loss')
    plt.plot(loss_acc['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(img_path, 'loss.png'), dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_fig(img, path):
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()


def reconstruct(model: MyVAE, img_path: str, num_images: int = 1):
    recons_path = os.path.join(img_path, "reconstruction")
    if not os.path.exists(recons_path):
        os.mkdir(recons_path)

    image_paths = get_image_path(args.data_root)
    X_train = load_image(image_paths[:num_images])
    mean, _ = model.encode(torch.from_numpy(X_train).to(device))
    X_prime = model.decode(mean)
    X_prime = X_prime.detach().cpu().numpy()
    for i in range(num_images):
        X_i = X_train[i].swapaxes(0, 2).swapaxes(0, 1)
        X_prime_i = X_prime[i].swapaxes(0, 2).swapaxes(0, 1)
        X_prime_i = (X_prime_i - X_prime_i.min()) / (X_prime_i.max() - X_prime_i.min())
        save_fig(X_i, os.path.join(recons_path, f"original_{i}.png"))
        save_fig(X_prime_i, os.path.join(recons_path, f"reconstructed_{i}.png"))


def interpolate(model: MyVAE, img_path: str, comp_path: str):
    inter_path = os.path.join(img_path, "interpolation")
    if not os.path.exists(inter_path):
        os.mkdir(inter_path)

    imgs = get_image_path(comp_path)
    imgs = load_image(imgs)
    imgs = torch.from_numpy(np.array(imgs)).to(device)
    means, _ = model.encode(imgs)
    z1, z2 = means[0], means[1]
    for alpha in np.linspace(0, 1, 11):
        z = alpha * z1 + (1 - alpha) * z2
        z = z.unsqueeze(0)
        X_prime = model.decode(z)
        X_prime = X_prime.detach().cpu().numpy()
        X_prime = X_prime[0].swapaxes(0, 2).swapaxes(0, 1)
        X_prime = (X_prime - X_prime.min()) / (X_prime.max() - X_prime.min())
        save_fig(X_prime, os.path.join(inter_path, f"interpolate_{int(10*alpha)}.png"))


args = make_visualization_argparser()
model_path = os.path.join(args.save_root, args.model_dir)
img_path = os.path.join(model_path, 'images')
if not os.path.exists(img_path):
    os.mkdir(img_path)
with open(os.path.join(model_path, 'log.txt'), 'r') as f:
    log = f.read().split('\n')[1:11]
train_args = {}
for arg in log:
    idx = arg.find(':')
    train_args[arg[:idx].strip()] = arg[idx+1:].strip()

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyVAE(init_size=256, code_dim=int(train_args['code_dim']), 
              hidden_dims=[int(d) for d in train_args['hidden_dims'][1:-1].split(',')],
            ).to(device)
state_dict = torch.load(os.path.join(model_path, 'model.pth'))
model.load_state_dict(state_dict)
model.eval()

# Visualize
plot_loss(model_path, img_path)
reconstruct(model, img_path, num_images=5)
interpolate(model, img_path, args.interpolate_component)
