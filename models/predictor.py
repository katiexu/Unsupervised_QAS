'''
Reference:
Zhimin He, Maijie Deng, Shenggen Zheng, Lvzhou Li, Haozhen Situ,
GSQAS: Graph Self-supervised Quantum Architecture Search,
Physica A: Statistical Mechanics and its Applications,
Volume 630,
2023,
129286,
ISSN 0378-4371,
https://doi.org/10.1016/j.physa.2023.129286.
'''

import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
import torch.nn.functional as F

class GNNPredictor(nn.Module):

    def __init__(self, args):
        super(GNNPredictor, self).__init__()

        self.linear1 = nn.Linear(args.dim, 64)
        self.linear5 = nn.Linear(64, 32)
        self.linear6 = nn.Linear(32, 1)

    def forward(self, x):
        #batch_size, node_num, embedding_dim = x.shape
        #x = torch.sum(x, dim=1)
        feature = x

        out = self.linear1(x)
        out = torch.relu(out)
        out = F.dropout(out, p=0.2, training=self.training)

        out = self.linear5(out)
        out = torch.relu(out)
        out = F.dropout(out, p=0.2, training=self.training)

        out = self.linear6(out)
        out = torch.sigmoid(out)

        out = out.mean(-1)
        return out, feature
    
class GSQASPredictor(nn.Module):

    def __init__(self, args):
        super(GSQASPredictor, self).__init__()
        self.linear1 = nn.Linear(args.dim, 20)
        self.linear2 = nn.Linear(20, 1)
        self.BN = nn.BatchNorm1d(20)
        self.args = args

    def forward(self, x):
        #x = torch.sum(x, dim=1)
        feature = x
        x = self.linear1(x)
        x = torch.relu(self.BN(x))
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = x.mean(-1)
        return x, feature

def GNNPredictor_prediction(embedding, args):
    idx = np.random.choice(range(len(embedding)), size=args.sample_num, replace=False)
    app = args.emb_path[:args.emb_path.find('-')]
    train_index = idx[0:1000]
    val_index = idx[99000:]
    test_index = idx[1000:]
    x_train = torch.Tensor(np.array([embedding[i]['feature'].detach().numpy() for i in train_index]))
    #x_train = embedding[train_index]['feature']
    x_val = torch.Tensor(np.array([embedding[i]['feature'].detach().numpy() for i in val_index]))
    #x_val = embedding[val_index]['feature']
    x_test = torch.Tensor(np.array([embedding[i]['feature'].detach().numpy() for i in test_index]))
    #x_test = embedding[test_index]['feature']
    if app == "fidelity":
        y_train = torch.Tensor(np.array([embedding[i]['fidelity'] for i in train_index]))
        #y_train = embedding[train_index]['fidelity']
        y_val = torch.Tensor(np.array([embedding[i]['fidelity'] for i in val_index]))
        #y_val = embedding[val_index]['fidelity']
        y_test = torch.Tensor(np.array([embedding[i]['fidelity'] for i in test_index]))
        #y_test = embedding[test_index]['fidelity']
    else:
        y_train = torch.Tensor(np.array([embedding[i]['energy'] for i in train_index]))
        #y_train = embedding[train_index]['fidelity']
        y_val = torch.Tensor(np.array([embedding[i]['energy'] for i in val_index]))
        #y_val = embedding[val_index]['fidelity']
        y_test = torch.Tensor(np.array([embedding[i]['energy'] for i in test_index]))
        #y_test = embedding[test_index]['fidelity']
        y_train = torch.div(y_train, args.ground_state_energy)
        y_val = torch.div(y_val, args.ground_state_energy)
        y_test = torch.div(y_test, args.ground_state_energy)
        y_train = torch.where(y_train < 0, torch.zeros_like(y_train), y_train)
        y_train = torch.where(y_train > 1, torch.ones_like(y_train), y_train)
        y_val = torch.where(y_val < 0, torch.zeros_like(y_val), y_val)
        y_val = torch.where(y_val > 1, torch.ones_like(y_val), y_val)
        y_test = torch.where(y_test < 0, torch.zeros_like(y_test), y_test)
        y_test = torch.where(y_test > 1, torch.ones_like(y_test), y_test)

    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    test_data = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=100, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False, drop_last=False)
    gnn_model = GNNPredictor(args).cuda()
    gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.0001)
    loss_func = nn.MSELoss()

    print("GNNPredictor begins training...")
    for i in range(300):

        gnn_model.train()
        train_loss = []
        for step, (b_x, b_y) in enumerate(train_loader):
            gnn_optimizer.zero_grad()
            b_x, b_y = b_x.cuda(), b_y.cuda()
            forward = gnn_model(b_x)[0]
            loss = loss_func(forward, b_y)
            loss.backward()
            gnn_optimizer.step()
            train_loss.append(loss.item())
        if (i+1) % 10 == 0:
            print(f"epoch:{i+1}")
            print('train_loss:{:.5f}'.format(sum(train_loss)/len(train_loss)))

        valid_loss = []
        gnn_model.eval()
        for step, (b_x, b_y) in enumerate(val_loader):
            b_x, b_y = b_x.cuda(), b_y.cuda()
            with torch.no_grad():
                forward = gnn_model(b_x)[0]
            loss = loss_func(forward, b_y)
            valid_loss.append(loss.item())
        if (i+1) % 10 == 0:
            print('valid_loss:{:.5f}'.format(sum(valid_loss) / len(valid_loss)))

    prediction = gnn_model(x_val.cuda())[0].detach().cpu().numpy()
    y_val = y_val.detach().cpu().numpy()

    #plt.scatter(prediction, y_val, s=1)
    #plt.show()


    corr_search = pearsonr(prediction, y_val)
    print(f"the pearson correlation of the prediction: {corr_search}")

    selected_index = []
    for i in range(len(test_index)):
        test_point_x = x_test[i].cuda()
        with torch.no_grad():
            p = gnn_model(test_point_x)[0].item()
        if p >= args.filter_threshold:
           selected_index.append(test_index[i]) 
    print(len(selected_index))

    count = 0
    for i in selected_index:
        if app == 'fidelity':
            reward = embedding[i]['fidelity']
        else:
            reward = embedding[i]['energy'] / args.ground_state_energy
        if reward >= args.threshold:
            count += 1
    print(f"count: {count}")

    return selected_index, count, train_index, corr_search

def GSQASPredictor_prediction(embedding, args):
    idx = np.random.choice(range(len(embedding)), size=args.sample_num, replace=False)
    app = args.emb_path[:args.emb_path.find('-')]
    train_index = idx[0:1000]
    val_index = idx[99000:]
    test_index = idx[1000:]
    x_train = torch.Tensor(np.array([embedding[i]['feature'].detach().numpy() for i in train_index]))
    #x_train = embedding[train_index]['feature']
    x_val = torch.Tensor(np.array([embedding[i]['feature'].detach().numpy() for i in val_index]))
    #x_val = embedding[val_index]['feature']
    x_test = torch.Tensor(np.array([embedding[i]['feature'].detach().numpy() for i in test_index]))
    #x_test = embedding[test_index]['feature']
    if app == "fidelity":
        y_train = torch.Tensor(np.array([embedding[i]['fidelity'] for i in train_index]))
        #y_train = embedding[train_index]['fidelity']
        y_val = torch.Tensor(np.array([embedding[i]['fidelity'] for i in val_index]))
        #y_val = embedding[val_index]['fidelity']
        y_test = torch.Tensor(np.array([embedding[i]['fidelity'] for i in test_index]))
        #y_test = embedding[test_index]['fidelity']
    else:
        y_train = torch.Tensor(np.array([embedding[i]['energy'] for i in train_index]))
        #y_train = embedding[train_index]['fidelity']
        y_val = torch.Tensor(np.array([embedding[i]['energy'] for i in val_index]))
        #y_val = embedding[val_index]['fidelity']
        y_test = torch.Tensor(np.array([embedding[i]['energy'] for i in test_index]))
        #y_test = embedding[test_index]['fidelity']
        y_train = torch.div(y_train, args.ground_state_energy)
        y_val = torch.div(y_val, args.ground_state_energy)
        y_test = torch.div(y_test, args.ground_state_energy)
        y_train = torch.where(y_train < 0, torch.zeros_like(y_train), y_train)
        y_train = torch.where(y_train > 1, torch.ones_like(y_train), y_train)
        y_val = torch.where(y_val < 0, torch.zeros_like(y_val), y_val)
        y_val = torch.where(y_val > 1, torch.ones_like(y_val), y_val)
        y_test = torch.where(y_test < 0, torch.zeros_like(y_test), y_test)
        y_test = torch.where(y_test > 1, torch.ones_like(y_test), y_test)

    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    test_data = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=100, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False, drop_last=False)
    gsqas_model = GSQASPredictor(args).cuda()
    gsqas_optimizer = torch.optim.Adam(gsqas_model.parameters(), lr=0.0001)
    loss_func = nn.MSELoss()

    print("GSQASPredictor begins training...")
    for i in range(200):

        gsqas_model.train()
        train_loss = []
        for step, (b_x, b_y) in enumerate(train_loader):
            gsqas_optimizer.zero_grad()
            b_x, b_y = b_x.cuda(), b_y.cuda()
            forward = gsqas_model(b_x)[0]
            loss = loss_func(forward, b_y)
            loss.backward()
            gsqas_optimizer.step()
            train_loss.append(loss.item())
        if (i+1) % 10 == 0:
            print(f"epoch:{i+1}")
            print('train_loss:{:.5f}'.format(sum(train_loss)/len(train_loss)))

        valid_loss = []
        gsqas_model.eval()
        for step, (b_x, b_y) in enumerate(val_loader):
            b_x, b_y = b_x.cuda(), b_y.cuda()
            with torch.no_grad():
                forward = gsqas_model(b_x)[0]
            loss = loss_func(forward, b_y)
            valid_loss.append(loss.item())
        if (i+1) % 10 == 0:
            print('valid_loss:{:.5f}'.format(sum(valid_loss) / len(valid_loss)))

    prediction = gsqas_model(x_val.cuda())[0].detach().cpu().numpy()

    #plt.scatter(prediction, y_val, s=1)
    #plt.show()


    corr_search = pearsonr(prediction, y_val)
    print(f"the pearson correlation of the prediction: {corr_search}")

    selected_index = []
    for i in range(len(test_index)):
        test_point_x = x_test[i].unsqueeze(0).cuda()
        with torch.no_grad():
            p = gsqas_model(test_point_x)[0].item()
        if p >= args.filter_threshold:
           selected_index.append(test_index[i]) 
    print(len(selected_index))

    count = 0
    for i in selected_index:
        if app == 'fidelity':
            reward = embedding[i]['fidelity']
        else:
            reward = embedding[i]['energy'] / args.ground_state_energy
        if reward >= args.threshold:
            count += 1
    print(f"count: {count}")

    return selected_index, count, train_index, corr_search


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="circuit_full_embedding_extraction")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--dir_name', type=str, default='pretrained\\dim-16')
    parser.add_argument('--emb_path', type=str, default='vqe-model-circuits_4_qubits.pt')
    parser.add_argument('--save_path', type=str, default='saved_figs\\fidelity')
    parser.add_argument("--sample_num", type=int, default=100000, help="total number of samples (default 100000)")
    parser.add_argument('--dim', type=int, default=16, help='feature dimension (default: 16)')
    parser.add_argument("--threshold", type=float, default=0.95, help="fidelity threshold (default 0.95)")
    parser.add_argument("--filter_threshold", type=float, default=0.675, help="fidelity predictor-filtered threshold (default 0.4)")
    parser.add_argument('--ground_state_energy', type=float, default=-1.136, help="The ground state energy of the graph hamiltonian")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    num_dataset = args.sample_num
    f_path = os.path.join(args.dir_name, '{}_full_embedding.pt'.format(args.emb_path[:-3]))
    if not os.path.exists(f_path):
        print('{} is not saved, please save it first!'.format(f_path))
        exit()
    print("load full feature embedding from: {}".format(f_path))
    embedding = torch.load(f_path)
    print("load finished")

    _, _, _, _ = GNNPredictor_prediction(embedding, args)
    #_, _, _, _ = GSQASPredictor_prediction(embedding, args)