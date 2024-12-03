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

# visulize the full embedding of 4-qubits maxcut in 2D space
def visualize2D(method, features, energy, palette, bound, norm_smooth, norm_scatter, method_name=None):
    emb_features = method.fit_transform(features)
    emb_x = emb_features[:, 0] / np.amax(np.abs(emb_features[:, 0]))
    emb_y = emb_features[:, 1] / np.amax(np.abs(emb_features[:, 1]))

    plt.rcParams.update({'font.size': 14})

    ## architecture density
    fig, ax = plt.subplots(figsize=(5, 5))
    xedges = np.linspace(-1.02, 1.02, 103)
    yedges = np.linspace(-1.02, 1.02, 103)
    H, xedges, yedges, img = ax.hist2d(emb_x, emb_y, bins=(xedges, yedges))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax, ax=ax)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.set_label('Density')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(args.save_path, method_name, 'density-{}-{}-points.png'.format(args.emb_path[:-3], 'full_embedding_density')),
                dpi=500, bbox_inches='tight')
    plt.close()

    # fidelity distribution
    xw = xedges[1] - xedges[0]
    yw = yedges[1] - yedges[0]
    x_cor = np.floor((emb_x - xedges[0]) / xw).astype(int)
    y_cor = np.floor((emb_y - yedges[0]) / yw).astype(int)
    mean_energy = np.zeros((101, 101))
    for xx in range(101):
        for yy in range(101):
            idx = np.logical_and((x_cor == xx), (y_cor == yy))
            if idx.any():
                mean_energy[xx, yy] = np.mean(energy[idx])
    xx = (np.linspace(0, 100, 101) + 0.2) * xw + xedges[0]
    yy = (np.linspace(0, 100, 101) + 0.2) * yw + yedges[0]

    ma_energy = np.ma.masked_where(mean_energy == 0, mean_energy)

    ## raw version
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    im = ax.imshow(ma_energy.T,
                   cmap=palette,
                   norm=norm_smooth,
                   origin='lower',
                   extent=[xx[0], xx[-1], yy[0], yy[-1]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both')
    cbar.set_label('energy distribution')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(args.save_path, method_name, '{}_{}.png'.format(args.emb_path[:-3], 'full_embedding_raw')),
                dpi=500, bbox_inches='tight')
    plt.close()

    ## smooth version
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    im = ax.imshow(ma_energy.T,
                   cmap=palette,
                   interpolation='bilinear',
                   norm=norm_smooth,
                   origin='lower',
                   extent=[xx[0], xx[-1], yy[0], yy[-1]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both')
    cbar.set_label('energy distribution')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(args.save_path, method_name, '{}_{}.png'.format(args.emb_path[:-3], 'full_embedding_smooth')),
                dpi=500, bbox_inches='tight')
    plt.close()

    # scatter version
    #x1 = emb_x[energy/args.ground_state_energy >= 0.95]
    #y1 = emb_y[energy/args.ground_state_energy >= 0.95]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.scatter(emb_x, emb_y, c=energy, s=1, cmap=palette, norm=norm_scatter, edgecolors='none')
    #ax.scatter(x1, y1, c='r', s=1, edgecolors='none')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(
        im,
        ax=ax,
        extend='both',
        ticks=bound,
        spacing='uniform',
        orientation='vertical',
        label='energy distribution',
    )
    plt.title("4-qubit maxcut feature embedding")
    plt.xlabel("emb_X")
    plt.ylabel("emb_y")
    plt.savefig(os.path.join(args.save_path, method_name, '{}_{}.png'.format(args.emb_path[:-3], 'full_embedding_scatter')),
                dpi=500, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="circuit_full_embedding_extraction")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--dir_name', type=str, default='pretrained\\dim-16')
    parser.add_argument('--emb_path', type=str, default='maxcut-model-circuits_4_qubits_wokl.pt')
    parser.add_argument('--save_path', type=str, default='saved_figs\\maxcut')
    parser.add_argument("--sample_num", type=int, default=100000, help="total number of samples (default 100000)")
    parser.add_argument("--threshold", type=float, default=0.95, help="energy threshold (default 0.95)")
    parser.add_argument('--ground_state_energy', type=float, default=-10, 
                        help="The ground state energy of the hamiltonian")

    features = []
    energy = []
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
        #features.append(embedding[ind]['feature'].detach().numpy())
        features.append(embedding[ind]['feature'].detach().numpy())
        energy.append(embedding[ind]['energy'])

    features = np.stack(features, axis=0)
    energy = np.stack(energy, axis=0)
    filtered_energy = list(filter(lambda energy: energy / args.ground_state_energy >= args.threshold, energy))
    print("The number of accepted candidates in the sampled dataset: {}".format(len(filtered_energy)))

    palette = copy(plt.cm.viridis).reversed()
    palette.set_under('r', 1.0)
    palette.set_over('k', 1.0)
    palette.set_bad('w', 1.0)

    bound_scatter = np.array([0.95, 0.9, 0.85, 0.8, 0.75, 0.7]) * args.ground_state_energy
    norm_scatter = mpl.colors.Normalize(vmin=args.ground_state_energy*0.95, vmax=args.ground_state_energy*0.7)
    norm_smooth = mpl.colors.Normalize(vmin=args.ground_state_energy*0.9, vmax=args.ground_state_energy*0.74)

    ########### TSNE ############
    ## tsne reduces dim
    print('TSNE...')
    tsne = TSNE(perplexity=50, learning_rate=1000, early_exaggeration=8, n_iter=3000, random_state=args.seed)
    visualize2D(tsne, features, energy, palette, bound_scatter, norm_smooth, norm_scatter, method_name="tsne")
    print('TSNE done.')

    ########### PCA ############
    ## PCA reduces dim
    print('PCA...')
    pca = PCA(n_components=2, random_state=args.seed)
    visualize2D(pca, features, energy, palette, bound_scatter, norm_smooth, norm_scatter, method_name="pca")
    print('PCA done.')