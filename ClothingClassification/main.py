import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gc
import random
from config import make_argparser
from data_loader import get_dataloader
from model import MyNet


seed = 230423
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

args = make_argparser()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, test_loader = get_dataloader(args.data_root, args.batch_size)
model = MyNet().to(device)
loss_fn = F.nll_loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Train the model
train_losses, train_accs = [], []
test_losses, test_accs = [], []
for epoch in range(args.epochs):
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
    print(f'Epoch {epoch}:\tTraining Loss: {train_losses[-1]:.4f}\tTrain Acc: {100*train_accs[-1]:.2f}%', end='\t')    

    torch.cuda.empty_cache()
    gc.collect()
    
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, Y in test_loader:
            pred = model(X)
            loss = loss_fn(pred, Y)
            test_loss += loss.cpu().item()
            test_acc += (pred.argmax(dim=1) == Y).cpu().float().mean()
    test_losses.append(test_loss/len(test_loader))
    test_accs.append(test_acc/len(test_loader))
    print(f'Testing Loss: {test_losses[-1]:.4f}\tTest Acc: {100*test_accs[-1]:.2f}%')


# Plot the training and testing losses
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.legend()
plt.savefig('loss.png', bbox_inches='tight', pad_inches=0)
plt.show()

# Plot the training and testing accuracies
plt.figure()
plt.plot(train_accs, label='Training Accuracy')
plt.plot(test_accs, label='Testing Accuracy')
plt.legend()
plt.savefig('accuracy.png', bbox_inches='tight', pad_inches=0)
plt.show()
