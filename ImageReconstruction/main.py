import numpy as np
import torch
import gc
import random
import os
import pickle
from time import time
import logging
from config import make_train_argparser
from data_loader import get_dataloader
from model import MyVAE


seed = 230423
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

args = make_train_argparser()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyVAE(init_size=256, code_dim=args.code_dim, 
                hidden_dims=args.hidden_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

train_loader, val_loader = get_dataloader(args.data_root, args.batch_size)

save_path = os.path.join(args.save_root, str(round(time()))[-6:])
os.mkdir(save_path)

# Create logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("Arguments:")
for k, v in vars(args).items():
    logger.info(f'  {k}: {v}')

train_losses, val_losses = [], []
best_stat = {'epoch': -1, 'val_loss': float('inf')}

# Train the model
for epoch in range(args.epochs):
    # train
    model.train()
    train_loss, train_acc = 0, 0
    for X in train_loader:
        X_prime, mean, log_var = model(X)
        loss = model.loss_fn(X, X_prime, mean, log_var)
        train_loss += loss.cpu().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_losses.append(train_loss/len(train_loader))

    torch.cuda.empty_cache()
    gc.collect()

    # validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X in val_loader:
            X_prime, mean, log_var = model(X)
            loss = model.loss_fn(X, X_prime, mean, log_var)
            val_loss += loss.cpu().item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    if val_loss < best_stat['val_loss']:
        best_stat['epoch'] = epoch
        best_stat['val_loss'] = val_loss
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    best_msg = f"\nNew best model in epoch {epoch}!" if epoch == best_stat['epoch'] else ''

    logger.info(f'Epoch {epoch},\ttraining loss: {train_losses[-1]:.4f},\t\
validation loss: {val_losses[-1]:.4f},'+best_msg) 

logger.info(f"Final best model in epoch {best_stat['epoch']}, validation loss: {best_stat['val_loss']:.4f}")


# Save the loss and accuracy
with open(os.path.join(save_path, 'loss.pkl'), 'wb') as f:
    pickle.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

logger.info(f"Saved to {save_path}!")
