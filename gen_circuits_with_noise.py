import json
import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import argparse
import numpy as np
import var_config as vc

from configs import configs
from model import GVAE
from utils.utils import preprocessing
from utils.utils import is_valid_circuit


np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run script with alpha parameter.')
    parser.add_argument('--alpha', type=float, required=True, help='Alpha parameter')
    args = parser.parse_args()

    # Get the alpha parameter
    alpha = args.alpha

    print(f"Running script with alpha = {alpha}")

    # alpha = 0.5

    # checkpoint = torch.load('../models/pretrained/dim-16/model-circuits_4_qubits.json.pt')
    checkpoint = torch.load('pretrained/dim-16/model-circuits_4_qubits.json.pt')
    input_dim = 2 + len(vc.allowed_gates) + vc.num_qubits
    model = GVAE((input_dim, 32, 64, 128, 64, 32, 16), normalize=True, dropout=0.3, **configs[4]['GAE']).cuda()
    model.load_state_dict(checkpoint['model_state'])
    # print(model)

    with open('data_selected_circuits.json', 'r') as file:
        dataset = json.load(file)

    adj, ops, indices = _build_dataset(dataset, range(int(len(dataset))))
    adj, ops = adj.cuda(), ops.cuda()

    bs = 32
    chunks = len(dataset)// bs
    if len(dataset) % bs > 0:
        chunks += 1
    adj_split = torch.split(adj, bs, dim=0)
    ops_split = torch.split(ops, bs, dim=0)
    indices_split = torch.split(indices, bs, dim=0)

    # preprocessing
    adj, ops, prep_reverse = preprocessing(adj, ops, **configs[4]['prep'])
    # forward
    Z = []
    model.eval()
    mu, logvar = model.encoder(ops, adj)
    Z.append(mu)
    z = torch.cat(Z, dim=0)

    circuits_with_noise = []
    # Generate 10 new circuits with noise
    while len(circuits_with_noise) < 10:
        noise = torch.randn_like(z) * z.std() + z.mean()
        z_with_noise = z + alpha * noise
        model.eval()
        full_op, full_ad = model.decoder(z_with_noise)
        full_op = full_op.squeeze(0).cpu()
        ad = full_ad.squeeze(0).cpu()
        # Restore ops
        op = full_op[:, 0:-(vc.num_qubits)]
        max_idx = torch.argmax(op, dim=-1)
        one_hot = torch.zeros_like(op)
        for i in range(one_hot.shape[0]):
            one_hot[i][max_idx[i]] = 1
        op_decode = transform_operations(max_idx)
        # Restore adj matrix
        ad_decode = (ad > 0.5).int().triu(1).numpy()
        ad_decode = np.ndarray.tolist(ad_decode)
        if is_valid_circuit(ad_decode, op_decode):
            op_results = []
            # Restore the qubit choices of ops
            qubit_choices = full_op[:, -(vc.num_qubits):]
            for i in range(qubit_choices.size(0)):
                if op_decode[i] == 'C(U3)':
                    # Select the two largest values and sort indices by value
                    values, indices = torch.topk(qubit_choices[i], 2)
                    indices = indices[values.argsort(descending=True)]
                    op_results.append((op_decode[i], indices.tolist()))
                elif op_decode[i] in ['U3', 'RX', 'RY', 'RZ']:
                    values, indices = torch.topk(qubit_choices[i], 1)
                    op_results.append((op_decode[i], indices.tolist()))
                else:
                    pass  # Skip 'START', 'END', and 'Identity' gates as they don't change the state

            circuits_with_noise.append(op_results)

        with open('circuits_with_noise.json', 'w') as file:
            json.dump(circuits_with_noise, file)