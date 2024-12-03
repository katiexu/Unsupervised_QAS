import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import copy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.utils.random import sample_without_replacement

# visulize the full embedding of 4-qubits fidelity in 2D space
def visualize2D(method, features, fidelity, palette, bound, norm_smooth, norm_scatter, method_name=None):
    emb_features = method.fit_transform(features)
    emb_x = emb_features[:, 0] / np.amax(np.abs(emb_features[:, 0]))
    emb_y = emb_features[:, 1] / np.amax(np.abs(emb_features[:, 1]))

    plt.rcParams.update({'font.size': 14})

    ## architecture density
    fig, ax = plt.subplots(figsize=(5, 5))
    xedges = np.linspace(-1.02, 1.02, 52)
    yedges = np.linspace(-1.02, 1.02, 52)
    H, xedges, yedges, img = ax.hist2d(emb_x, emb_y, bins=(xedges, yedges))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax, ax=ax)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.set_label('Density')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(args.save_path, method_name, 'density-{}-{} points.png'.format(args.emb_path[:-3], 'full_embedding_density')),
                dpi=500, bbox_inches='tight')
    plt.close()

    # fidelity distribution
    xw = xedges[1] - xedges[0]
    yw = yedges[1] - yedges[0]
    x_cor = np.floor((emb_x - xedges[0]) / xw).astype(int)
    y_cor = np.floor((emb_y - yedges[0]) / yw).astype(int)
    mean_fidelity = np.zeros((51, 51))
    for xx in range(51):
        for yy in range(51):
            idx = np.logical_and((x_cor == xx), (y_cor == yy))
            if idx.any():
                mean_fidelity[xx, yy] = np.mean(fidelity[idx])
    xx = (np.linspace(0, 50, 51) + 0.5) * xw + xedges[0]
    yy = (np.linspace(0, 50, 51) + 0.5) * yw + yedges[0]

    ma_fidelity = np.ma.masked_where(mean_fidelity == 0, mean_fidelity)

    ## raw version
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    im = ax.imshow(ma_fidelity.T,
                   cmap=palette,
                   norm=norm_smooth,
                   origin='lower',
                   extent=[xx[0], xx[-1], yy[0], yy[-1]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both')
    cbar.set_label('fidelity distribution')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(args.save_path, method_name, '{}_{}.png'.format(args.emb_path[:-3], 'full_embedding_raw')),
                dpi=500, bbox_inches='tight')
    plt.close()

    ## smooth version
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    im = ax.imshow(ma_fidelity.T,
                   cmap=palette,
                   interpolation='bilinear',
                   norm=norm_smooth,
                   origin='lower',
                   extent=[xx[0], xx[-1], yy[0], yy[-1]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both')
    cbar.set_label('fidelity distribution')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(args.save_path, method_name, '{}_{}.png'.format(args.emb_path[:-3], 'full_embedding_smooth')),
                dpi=500, bbox_inches='tight')
    plt.close()

    # scatter version
    x1 = emb_x[fidelity >= 0.95]
    y1 = emb_y[fidelity >= 0.95]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.scatter(emb_x, emb_y, c=fidelity, s=1, cmap=palette, norm=norm_scatter, edgecolors='none')
    ax.scatter(x1, y1, c='r', s=1, edgecolors='none')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(
        im,
        ax=ax,
        extend='both',
        ticks=bound,
        spacing='uniform',
        orientation='vertical',
        label='fidelity distribution',
    )
    plt.title("4-qubit fidelity feature embedding")
    plt.xlabel("emb_X")
    plt.ylabel("emb_y")
    plt.savefig(os.path.join(args.save_path, method_name, '{}_{}.png'.format(args.emb_path[:-3], 'full_embedding_scatter')),
                dpi=500, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="circuit_full_embedding_extraction")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument('--dir_name', type=str, default='pretrained\\dim-16')
    parser.add_argument('--emb_path', type=str, default='fidelity-model-circuits_4_qubits.pt')
    parser.add_argument('--save_path', type=str, default='saved_figs\\fidelity')
    parser.add_argument("--sample_num", type=int, default=100000, help="total number of samples (default 100000)")
    parser.add_argument("--threshold", type=float, default=0.95, help="vqe threshold (default 0.95)")

    features = []
    fidelity = []
    args = parser.parse_args()

    f_path = os.path.join(args.dir_name, '{}_full_embedding.pt'.format(args.emb_path[:-3]))
    if not os.path.exists(f_path):
        print('{} is not saved, please save it first!'.format(f_path))
        exit()
    print("load full feature embedding from: {}".format(f_path))
    embedding = torch.load(f_path)
    print("load finished")

    if args.sample_num <= 0 or args.sample_num >= len(embedding):
        sample_idx = range(len(embedding))
        args.sample_num = len(embedding)
    else:
        sample_idx = sample_without_replacement(len(embedding), args.sample_num, random_state=0)

    for i in tqdm(range(len(sample_idx)), desc=f'get {args.sample_num} samples from full feature embedding'):
        ind = sample_idx[i]
        features.append(embedding[ind]['feature'].detach().numpy())
        fidelity.append(embedding[ind]['fidelity'])

    features = np.stack(features, axis=0)
    fidelity = np.stack(fidelity, axis=0)
    filtered_fidelity = list(filter(lambda fidelity: fidelity >= args.threshold, fidelity))
    print("The number of accepted candidates in the sampled dataset: {}".format(len(filtered_fidelity)))

    palette = copy(plt.cm.viridis)
    palette.set_over('r', 1.0)
    palette.set_under('k', 1.0)
    palette.set_bad('w', 1.0)

    bound_scatter = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
    norm_scatter = mpl.colors.Normalize(vmin=0.2, vmax=0.95)
    norm_smooth =  mpl.colors.Normalize(vmin=0.2, vmax=0.5)

    ########### TSNE ############
    ## tsne reduces dim
    
    print('TSNE...')
    tsne = TSNE(perplexity=50, learning_rate=1000, early_exaggeration=8, n_iter=3000, random_state=args.seed)
    visualize2D(tsne, features, fidelity, palette, bound_scatter, norm_smooth, norm_scatter, method_name="tsne")
    print('TSNE done.')
    
    ########### PCA ############
    ## PCA reduces dim
    print('PCA...')
    pca = PCA(n_components=2, random_state=args.seed)
    visualize2D(pca, features, fidelity, palette, bound_scatter, norm_smooth, norm_scatter, method_name="pca")
    print('PCA done.')