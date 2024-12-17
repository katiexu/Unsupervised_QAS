import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import argparse
import numpy as np
import torch.nn as nn
import var_config as vc

from torch import optim
from configs import configs
from nasbench.lib import graph_util
from model import Model, VAEReconstructed_Loss, GVAE, Reconstructed_Loss
from utils.utils import load_json, save_checkpoint_vae, preprocessing
from utils.utils import get_val_acc_vae, is_valid_circuit


# def transform_operations(max_idx):
#     transform_dict =  {0:'START', 1:'PauliX', 2:'PauliY', 3:'PauliZ', 4:'Hadamard', 5:'RX',
#                        6:'RY', 7:'RZ', 8:'CNOT', 9:'CZ', 10:'U3', 11:'SWAP', 12:'END'}
#     ops = []
#     for idx in max_idx:
#         ops.append(transform_dict[idx.item()])
#     return ops

def transform_operations(max_idx):
    transform_dict =  {0:'START', 1:'U3', 2:'C(U3)', 3:'RX', 4:'RY', 5:'RZ', 6:'Identity', 7:'END'}
    ops = []
    for idx in max_idx:
        ops.append(transform_dict[idx.item()])
    return ops

def _build_dataset(dataset, list):
    indices = np.random.permutation(list)
    X_adj = []
    X_ops = []
    for ind in indices:
        X_adj.append(torch.Tensor(dataset[ind]['adj_matrix']))
        X_ops.append(torch.Tensor(dataset[ind]['gate_matrix']))
    X_adj = torch.stack(X_adj)
    X_ops = torch.stack(X_ops)
    return X_adj, X_ops, torch.Tensor(indices)

def pretraining_model(dataset, cfg, args):
    train_ind_list, val_ind_list = range(int(len(dataset)*0.9)), range(int(len(dataset)*0.9), len(dataset))
    X_adj_train, X_ops_train, indices_train = _build_dataset(dataset, train_ind_list)
    X_adj_val, X_ops_val, indices_val = _build_dataset(dataset, val_ind_list)
    # model = Model(input_dim=args.input_dim, hidden_dim=args.hidden_dim, latent_dim=args.dim,
    #                num_hops=args.hops, num_mlp_layers=args.mlps, dropout=args.dropout, **cfg['GAE']).cuda()
    model = GVAE((args.input_dim, 32, 64, 128, 64, 32, args.dim), normalize=True, dropout=args.dropout, **cfg['GAE']).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    epochs = args.epochs
    bs = args.bs
    loss_total = []
    for epoch in range(0, epochs):
        chunks = len(train_ind_list) // bs
        if len(train_ind_list) % bs > 0:
            chunks += 1
        X_adj_split = torch.split(X_adj_train, bs, dim=0)
        X_ops_split = torch.split(X_ops_train, bs, dim=0)
        indices_split = torch.split(indices_train, bs, dim=0)
        loss_epoch = []
        Z = []
        for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
            optimizer.zero_grad()
            adj, ops = adj.cuda(), ops.cuda()
            # preprocessing
            adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
            # forward
            ops_recon, adj_recon, mu, logvar = model(ops, adj)
            Z.append(mu)
            adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
            adj, ops = prep_reverse(adj, ops)
            loss = VAEReconstructed_Loss(**cfg['loss'])((ops_recon, adj_recon), (ops, adj), mu, logvar) # With KL
            # loss = Reconstructed_Loss(**cfg['loss'])((ops_recon, adj_recon), (ops, adj)) # Without KL
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_epoch.append(loss.item())
            if i%100 == 0:
                print('epoch {}: batch {} / {}: loss: {:.5f}'.format(epoch, i, chunks, loss.item()))
        Z = torch.cat(Z, dim=0)
        z_mean, z_std = Z.mean(0), Z.std(0)
        validity_counter = 0
        buckets = {}
        model.eval()
        for _ in range(args.latent_points):
            z = torch.randn(X_adj_train[0].shape[0], args.dim).cuda()
            z = z * z_std + z_mean
            if epoch == args.epochs - 1:
                torch.save(z, 'z.pt')
            full_op, full_ad = model.decoder(z.unsqueeze(0))
            full_op = full_op.squeeze(0).cpu()
            ad = full_ad.squeeze(0).cpu()
            op = full_op[:, 0:-(vc.num_qubits)]
            max_idx = torch.argmax(op, dim=-1)
            one_hot = torch.zeros_like(op)
            for i in range(one_hot.shape[0]):
                one_hot[i][max_idx[i]] = 1
            op_decode = transform_operations(max_idx)
            ad_decode = (ad>0.5).int().triu(1).numpy()
            ad_decode = np.ndarray.tolist(ad_decode)
            if is_valid_circuit(ad_decode, op_decode):
                validity_counter += 1
                fingerprint = graph_util.hash_module(np.array(ad_decode), one_hot.numpy().tolist())
                if fingerprint not in buckets:
                    buckets[fingerprint] = (ad_decode, one_hot.numpy().astype('int8').tolist())

        validity = validity_counter / args.latent_points
        print('Ratio of valid decodings from the prior: {:.4f}'.format(validity))
        print('Ratio of unique decodings from the prior: {:.4f}'.format(len(buckets) / (validity_counter+1e-8)))
        acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val = get_val_acc_vae(model, cfg, X_adj_val, X_ops_val, indices_val)
        print('validation set: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(
                acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val))
        print('epoch {}: average loss {:.5f}'.format(epoch, sum(loss_epoch)/len(loss_epoch)))
        loss_total.append(sum(loss_epoch) / len(loss_epoch))
        save_checkpoint_vae(model, optimizer, epoch, sum(loss_epoch) / len(loss_epoch), args.dim, args.name, args.dropout, args.seed)

    print('loss for epochs: \n', loss_total)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretraining')
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    # parser.add_argument('--data', type=str, default=f'circuit\\data\\data_{vc.num_qubits}_qubits.json',
    #                     help='Data file (default: data.json')
    parser.add_argument('--data', type=str, default=f'data/data_4_qubits.json',
                        help='Data file (default: data.json')
    parser.add_argument('--name', type=str, default=f'circuits_{vc.num_qubits}_qubits.json',
                        help='circuits with correspoding number of qubits')
    parser.add_argument('--cfg', type=int, default=4,
                        help='configuration (default: 4)')
    parser.add_argument('--bs', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=16,
                        help='training epochs (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='decoder implicit regularization (default: 0.3)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='use input normalization')
    parser.add_argument('--input_dim', type=int, default=2+len(vc.allowed_gates)+vc.num_qubits)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=16,
                        help='feature (latent) dimension (default: 16)')
    parser.add_argument('--hops', type=int, default=5)
    parser.add_argument('--mlps', type=int, default=2)
    parser.add_argument('--latent_points', type=int, default=10000,
                        help='latent points for validaty check (default: 10000)')

    args = parser.parse_args()
    # args.epochs = 100
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = configs[args.cfg]
    dataset = load_json(args.data)
    print('using {}'.format(args.data))
    print('feat dim {}'.format(args.dim))
    train_ind_list, val_ind_list = range(int(len(dataset)*0.9)), range(int(len(dataset)*0.9), len(dataset))
    X_adj_train, X_ops_train, indices_train = _build_dataset(dataset, train_ind_list)
    print(X_adj_train[0])
    print(X_ops_train[0])
    print(indices_train[0])
    pretraining_model(dataset, cfg, args)