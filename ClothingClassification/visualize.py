import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pickle
from MyPCA import my_PCA
from MyTSNE import my_tSNE
from typing import List, Optional
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
    plt.savefig(os.path.join(img_path, 'loss.png'), dpi=500, bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.plot(loss_acc['train_accs'], label='Training Accuracy')
    plt.plot(loss_acc['test_accs'], label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(img_path, 'accuracy.png'), dpi=500, bbox_inches='tight', pad_inches=0)

    plt.close('all')


def visualize_2d(samples: np.ndarray, method: str, img_path: str, name: str) -> Optional[float]:
    """Visualizes the 2D embeddings of the given samples using the given method.

    Args:
        samples (np.ndarray): The samples. The label of `samples[15*y:15*(y+1)]` is `y`.
        method (str): 'pca', 'tsne' or 'both'.
        img_path (str): The path to save the images.
        name (str): The name of the plot.

    Returns:
        Optional[float]: Proportion of preserved variance of PCA.
    """    

    assert method in ['pca', 'tsne', 'both']
    if method == 'pca':
        embeddings = [('pca', *my_PCA(samples))]
    elif method == 'tsne':
        embeddings = [('tsne', my_tSNE(samples))]
    else:
        embeddings = [('pca', *my_PCA(samples)),
                      ('tsne', my_tSNE(samples))]
    prop = None
    for emb in embeddings:
        mtd = emb[0]
        if mtd == 'pca':
            prop = emb[2]  # Proportions of preserved variance
        emb = emb[1]
        for y in range(10):
            plt.scatter(emb[15*y:15*(y+1),0], emb[15*y:15*(y+1),1], label=y, s=5)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5,1), borderaxespad=1, ncol=5)
        plt.savefig(os.path.join(img_path, '{}_{}.png'.format(name, mtd)), dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close('all')
    
    if prop is not None:
        return prop
    

def plot_preserved_var(props: List[float], names: List[str], img_path: str):
    """Plots the preserved variance of PCA."""    

    plt.bar(names[::-1], props[::-1])
    plt.xlabel('Layer')
    plt.ylabel('Proportion of Preserved Variance')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig(os.path.join(img_path, 'preserved_var.png'), dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close('all')


def get_CH_index(samples: np.ndarray):
    """Computes the Calinski-Harabasz index of the given samples."""    
    
    K = 10
    # Compute the centers of classes
    c = np.empty((K, samples.shape[1]))
    for i in range(K):
        c[i] = np.mean(samples[15*i:15*(i+1)], axis=0)
    # Compute the sum of within-cluster dispersion
    W = 0
    for i in range(K):
        W += np.sum((samples[15*i:15*(i+1)] - c[i])**2)
    # Compute the centers of all samples
    c_all = np.mean(samples, axis=0)
    # Compute the sum of between-cluster dispersion 
    B = 15 * np.sum((c - c_all)**2)

    return (B / W) * (samples.shape[0] - K) / (K - 1)


def plot_CHI(CHIs: List[float], names: List[str], img_path: str):
    """Plots the Calinski-Harabasz index."""    

    plt.bar(names[::-1], CHIs[::-1])
    plt.xlabel('Layer')
    plt.ylabel('Calinski-Harabasz Index')
    plt.savefig(os.path.join(img_path, 'CHI.png'), dpi=500, bbox_inches='tight', pad_inches=0)
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
# # Compute accuracy
# pred = np.argmax(embeds[0], axis=1)
# truth = np.repeat(np.arange(10), 15)
# print(f"Accuracy: {round(100*(pred == truth).mean(), 2)}%")

props = []
CHI = []
names = ['out', 'fc_out', 'conv_out']
for embed, name in zip(embeds, names):
    props.append(visualize_2d(embed, 'both', img_path, name))
    CHI.append(get_CH_index(embed))

plot_preserved_var(props, names, img_path)
plot_CHI(CHI, names, img_path)
