import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import pickle
from MyPCA import my_PCA
# from MyTSNE import my_tSNE
from sklearn.manifold import TSNE
from typing import Optional, List
from torchsummary import summary
from model import MyNet
from config import make_visualization_argparser
from data import get_data



def plot_loss_accuracy(model_path: str, img_path: str):
    """Plots the loss and accuracy curves for the training and testing sets."""    

    with open(os.path.join(model_path, 'loss_acc.pkl'), 'rb') as f:
        loss_acc = pickle.load(f)
    
    plt.figure()
    plt.plot(loss_acc['train_losses'], label='Training Loss')
    plt.plot(loss_acc['test_losses'], label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(img_path, 'loss.png'), bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.plot(loss_acc['train_accs'], label='Training Accuracy')
    plt.plot(loss_acc['test_accs'], label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(img_path, 'accuracy.png'), bbox_inches='tight', pad_inches=0)

    plt.close('all')


def visualize_2d(samples: np.ndarray, method: str, img_path: str, name: str) -> Optional[np.float32]:
    """Visualizes the 2D embeddings of the given samples using the given method.

    Args:
        samples (np.ndarray): The samples. The label of `samples[15*y:15*(y+1)]` is `y`.
        method (str): 'pca', 'tsne' or 'both'.
        img_path (str): The path to save the images.
        name (str): The name of the plot.

    Returns:
        Optional[np.float32]: The proportions of preserved variance in PCA.
    """    

    assert method in ['pca', 'tsne', 'both']
    if method == 'pca':
        embeddings = [('pca', *my_PCA(samples.reshape(samples.shape[0], -1)))]
    elif method == 'tsne':
        # embeddings = ['tsne', my_tSNE(samples.reshape(samples.shape[0], -1))]
        embeddings = [('tsne', TSNE(n_components=2).fit_transform(samples.reshape(samples.shape[0], -1)))]
    else:
        embeddings = [('pca', *my_PCA(samples.reshape(samples.shape[0], -1))),
                      ('tsne', TSNE(n_components=2).fit_transform(samples.reshape(samples.shape[0], -1)))]
    prop = None
    for emb in embeddings:
        mtd = emb[0]
        if mtd == 'pca':
            prop = emb[2]  # Proportions of preserved variance
        emb = emb[1]
        for y in range(10):
            plt.scatter(emb[15*y:15*(y+1),0], emb[15*y:15*(y+1),1], label=y, s=5)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5,1), borderaxespad=1, ncol=5)
        plt.savefig(os.path.join(img_path, '{}_{}.png'.format(name, mtd)), bbox_inches='tight', pad_inches=0)
        plt.close('all')
    
    if prop is not None:
        return prop
    

def plot_preserved_var(props: List[float], names: List[str], img_path: str):
    """Plots the preserved variance of PCA."""    

    plt.bar(names[::-1], props[::-1])
    plt.xlabel('Layer')
    plt.ylabel('Preserved Variance')
    plt.savefig(os.path.join(img_path, 'preserved_var.png'), bbox_inches='tight', pad_inches=0)
    plt.close('all')


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
model = MyNet(num_res_blocks=int(train_args['num_res_blocks']),
              num_channel_1=int(train_args['num_channel_1']),
              num_channel_2=int(train_args['num_channel_2']),
              hidden_dim_fc=int(train_args['hidden_dim_fc'])).to(device)
state_dict = torch.load(os.path.join(model_path, 'model.pth'))
model.load_state_dict(state_dict)
model.eval()
# print(model)
# summary(model, (1,32,32))


# Plot the loss and accuracy curves
plot_loss_accuracy(model_path, img_path)


# Visualize features
X_train, X_test, Y_train, Y_test = get_data(args.data_root)
X = np.concatenate((X_train, X_test), axis=0)
Y = np.concatenate((Y_train, Y_test), axis=0)
# Sample 15 images from each class
samples = np.concatenate([X[np.random.choice(np.where(Y == y)[0], 15)] 
                          for y in range(10)], axis=0)
samples = torch.from_numpy(samples).to(device)
# Compute features
with torch.no_grad():
    embeds = [embed.cpu().detach().numpy().reshape(150,-1) for embed in model(samples, return_emd=True)]
# Compute accuracy
pred = np.argmax(embeds[0], axis=1)
truth = np.repeat(np.arange(10), 15)
print(f"Accuracy: {100*round((pred == truth).mean(), 4)}%")

props = []
names = ['out', 'fc2_out', 'fc1_out', 'layer2_out', 'layer1_out']
for embed, name in zip(embeds, names):
    props.append(visualize_2d(embed, 'both', img_path, name))

plot_preserved_var(props, names, img_path)
