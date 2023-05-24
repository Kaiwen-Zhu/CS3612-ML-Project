import numpy as np
import torch
import torch.nn.functional as F
import gc
import random
import os
import pickle
from time import time
import logging
import matplotlib.pyplot as plt
from config import make_visualization_argparser
from data_loader import get_dataloader
from model import MyVAE
from data import get_data


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
model = MyVAE(init_size=32, code_dim=int(train_args['code_dim']), 
              hidden_dims=[int(d) for d in train_args['hidden_dims'][1:-1].split(',')]).to(device)
state_dict = torch.load(os.path.join(model_path, 'model.pth'))
model.load_state_dict(state_dict)
model.eval()

X_train = get_data(args.data_root)
X = X_train[0]
X_prime = model(torch.from_numpy(np.expand_dims(X,axis=0)).to(device))[0]
X_prime = X_prime.detach().cpu().numpy()[0]
X = X.swapaxes(0, 2).swapaxes(0, 1)
X_prime = X_prime.swapaxes(0, 2).swapaxes(0, 1)
# normalize X_prime
X_prime = (X_prime - X_prime.min()) / (X_prime.max() - X_prime.min())
# plot X and X_prime
fig, ax = plt.subplots(1, 2)
ax[0].imshow(X)
ax[1].imshow(X_prime)
plt.show()