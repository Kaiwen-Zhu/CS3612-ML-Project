import numpy as np
import torch
import torch.nn.functional as F
import gc
import random
import os
import pickle
from time import time
import logging
from config import make_train_argparser
from data_loader import get_dataloader
from model import MyNet


seed = 230423
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

args = make_train_argparser()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyNet(num_res_blocks=args.num_res_blocks, 
              num_channel_1=args.num_channel_1,
              num_channel_2=args.num_channel_2,
              hidden_dim_fc=args.hidden_dim_fc).to(device)
loss_fn = F.nll_loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

train_loader, val_loader, test_loader = get_dataloader(args.data_root, args.batch_size)

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

train_losses, train_accs = [], []
test_losses, test_accs = [], []
val_losses = []
best_stat = {'epoch': -1, 'val_loss': float('inf'), 'test_loss': float('inf'), 'test_acc': 0}

# Train the model
for epoch in range(args.epochs):
    # train
    model.train()
    train_loss, train_acc = 0, 0
    for X, Y in train_loader:
        pred = model(X)
        loss = loss_fn(pred, Y)
        train_loss += loss.cpu().item()
        train_acc += (pred.argmax(dim=1) == Y).cpu().float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_losses.append(train_loss/len(train_loader))
    train_accs.append(train_acc/len(train_loader))

    torch.cuda.empty_cache()
    gc.collect()

    # test
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, Y in test_loader:
            pred = model(X)
            loss = loss_fn(pred, Y)
            test_loss += loss.cpu().item()
            test_acc += (pred.argmax(dim=1) == Y).cpu().float().mean()
    test_losses.append(test_loss/len(test_loader))
    test_accs.append(test_acc/len(test_loader))

    # validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, Y in val_loader:
            pred = model(X)
            loss = loss_fn(pred, Y)
            val_loss += loss.cpu().item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    if val_loss < best_stat['val_loss']:
        best_stat['epoch'] = epoch
        best_stat['val_loss'] = val_loss
        best_stat['test_loss'] = test_losses[-1]
        best_stat['test_acc'] = test_accs[-1]
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    best_msg = f"\nNew best model in epoch {epoch}!" if epoch == best_stat['epoch'] else ''

    logger.info(f'Epoch {epoch},\ttraining loss: {train_losses[-1]:.4f},\ttrain acc: {100*train_accs[-1]:.2f}%,\t\
validation loss: {val_losses[-1]:.4f},\ttesting loss: {test_losses[-1]:.4f},\ttest acc: {100*test_accs[-1]:.2f}%'+best_msg) 

logger.info(f"Final best model in epoch {best_stat['epoch']}, validation loss: {best_stat['val_loss']:.4f}, \
testing loss: {best_stat['test_loss']:.4f}, test acc: {100*best_stat['test_acc']:.2f}%")


# Save the loss and accuracy
with open(os.path.join(save_path, 'loss_acc.pkl'), 'wb') as f:
    pickle.dump({'train_losses': train_losses, 'train_accs': train_accs,
                    'test_losses': test_losses, 'test_accs': test_accs}, f)

logger.info(f"Saved to {save_path}!")
