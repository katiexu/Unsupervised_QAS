import os
import sys
sys.path.insert(0, os.getcwd())
import json
import torch
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.utils import load_json
from sklearn.utils.random import sample_without_replacement

# calculate the distance between qubit sequences
def qubit_distance(q1, q2):
    q_xor = [(ord(a)^ord(b)) for a, b in zip(q1, q2)]
    q_distance = q_xor.count(1)
    return q_distance

# if only consider ops equal matching without qubits, then using node_match 
def node_match(n1, n2):
    if n1['op'] == n2['op'] and n1['qubits'] == n2['qubits']:
        return True
    else:
        return False

# if consider ops equal matching with qubits, then using node subsitution cost
def node_subst_cost(n1, n2):
    if n1['op'] == n2['op']:
        if n1['qubits'] == n2['qubits']:
            return 0
        else:
            return qubit_distance(n1['qubits'], n2['qubits'])
    else:
        if n1['qubits'] == n2['qubits']:
            return 1
        else:
            return qubit_distance(n1['qubits'], n2['qubits']) + 1
    

def edge_match(e1, e2):
    return True  

# calculate the edit distance between two graphs of circuits
def edit_distance(G1, G2, with_qubits=False):
    if with_qubits:
        d = nx.graph_edit_distance(G1,G2, node_match=node_match, edge_match=edge_match, node_subst_cost=node_subst_cost)
    else:
        d = nx.graph_edit_distance(G1,G2, node_match=node_match, edge_match=edge_match)
    return int(d)

# calculate the l2 distance between two latent feature vectors of circuits
def l2_distance(feat_1, feat_2):
    return np.linalg.norm(feat_1-feat_2, ord=2)

# preprocess gate and adjacent matrices
def preprocess_adj_op(full_ops, adj):
    full_ops = np.array(full_ops)
    qubits = full_ops[:, -args.num_qubits:]
    ops = full_ops[:, :-args.num_qubits]

    def transform_ops(max_idx):
        transform_dict =  {0:'START', 1:'PauliX', 2:'PauliY', 3:'PauliZ', 4:'Hadamard', 5:'RX', 
                        6:'RY', 7:'RZ', 8:'CNOT', 9:'CZ', 10:'U3', 11:'SWAP', 12:'END'}
        temp_ops = []
        i = 0
        for idx in max_idx:
            converted_qubits = map(str, qubits[i])
            converted_qubits = ''.join(converted_qubits)
            temp_ops.append(transform_dict[idx.item()] + converted_qubits)
            i += 1
        return temp_ops

    max_idx = np.argmax(ops, axis=-1)
    converted_ops = transform_ops(max_idx)

    return converted_ops, np.array(adj)

# generate the networkx graph of circuits
def gen_graph(full_ops, adj):
    G = nx.DiGraph()
    for k, full_op in enumerate(full_ops):
        qubits = full_op[-args.num_qubits:]
        op = full_op[:-args.num_qubits]
        G.add_node(k, op=op, qubits=qubits)
    assert adj.shape[0] == adj.shape[1] == len(full_ops)
    for row in range(len(full_ops)):
        for col in range(row + 1, len(full_ops)):
            if adj[row, col] > 0:
                G.add_edge(row, col)
    return G  

# randomly choose some circuits to compare
def random_walk(dataset, feature_embedding):
    dist_pair = []
    sample_idx = sample_without_replacement(len(dataset), args.sample_num, random_state=args.seed)
    for i in tqdm(range(len(sample_idx)), desc=f'calculate the edit and l2 distance between the sample with others'):
        ind_curr = sample_idx[i]
        gate_mat = dataset[ind_curr]['gate_matrix']
        adj_mat = dataset[ind_curr]['adj_matrix']
        ops, adj = preprocess_adj_op(gate_mat, adj_mat)
        G1 = gen_graph(ops, adj)
        f1 = feature_embedding[ind_curr]['feature']
        for j in range(i, len(sample_idx)):
            ind_oth = sample_idx[j]
            gate_mat = dataset[ind_oth]['gate_matrix']
            adj_mat = dataset[ind_oth]['adj_matrix']
            ops, adj = preprocess_adj_op(gate_mat, adj_mat)
            G2 = gen_graph(ops, adj)
            f2 = feature_embedding[ind_oth]['feature']
            edit_dist = edit_distance(G1, G2, args.with_qubits)
            l2_dist = l2_distance(f1, f2)
            dist_pair.append((edit_dist, l2_dist))
    dtype = [('edit_distance', int), ('l2_distance', float)]
    dist_pair = np.array(dist_pair, dtype=dtype)
    dist_pair = np.sort(dist_pair, order='edit_distance') 
    return dist_pair


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="circuit_full_embedding_extraction")
    parser.add_argument("--seed", type=int, default=4, help="random seed")
    parser.add_argument("--num_qubits", type=int, default=4, help="the number of used qubits")
    parser.add_argument('--dir_name', type=str, default='pretrained\\dim-16')
    parser.add_argument('--emb_path', type=str, default='maxcut-model-circuits_4_qubits.pt')
    parser.add_argument('--save_path', type=str, default='saved_figs\\maxcut')
    parser.add_argument("--sample_num", type=int, default=20)
    parser.add_argument('--with_qubits', type=bool, default=True, help='whether considering qubits when calculating edit distance')

    args = parser.parse_args()

    f_path = os.path.join(args.dir_name, '{}_full_embedding.pt'.format(args.emb_path[:-3]))
    if not os.path.exists(f_path):
        print('{} is not saved, please save it first!'.format(f_path))
        exit()
    
    print("load full feature embedding from: {}".format(f_path))
    feature_embedding = torch.load(f_path)

    print("load circuit dataset from circuit\\data\\data_4_qubits.json")
    dataset = load_json('circuit\\data\\data_4_qubits.json')

    dist_pair = random_walk(dataset, feature_embedding)
    print(dist_pair)

    ###### The code used to test ######
    '''
    gate_mat = dataset[0]['gate_matrix']
    adj_mat = dataset[1]['adj_matrix']
    ops, adj = preprocess_adj_op(gate_mat, adj_mat)
    G1 = gen_graph(ops, adj)
    G2 = gen_graph(ops, adj)
    print(G1)
    print(G2)
    edit_dist1 = edit_distance(G1, G1)
    edit_dist2 = edit_distance(G1, G2)
    print(edit_dist1)
    print(edit_dist2)
    '''

    