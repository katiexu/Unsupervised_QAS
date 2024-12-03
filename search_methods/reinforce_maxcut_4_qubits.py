import os
import sys
sys.path.insert(0, os.getcwd())
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.model import Model
from models.configs import configs
from circuit.maxcut import vqe_maxcut
from utils.utils import load_json, preprocessing
from torch.distributions import MultivariateNormal


class Env(object):
    def __init__(self, name, seed, emb_path, model_path, graph, cfg, data_path=None, save=False, full_embedding=False):
        self.name = name
        self.seed = seed
        self.emb_path = emb_path
        self.model_path = model_path
        self.graph = graph
        self.cfg = cfg
        self.dir_name = 'pretrained\\dim-{}'.format(args.dim)
        self.visited = {}
        self.features = []
        self.embedding = {}
        self.dataset = load_json(data_path)
        if full_embedding:
            self.get_full_embedding()
        else:
            self._reset(save)


    def _reset(self, save):
        if not save:
            print("extract features from {}".format(os.path.join(self.dir_name, self.model_path)))
            if not os.path.exists(os.path.join(self.dir_name, self.model_path)):
                exit()
            self.model = Model(input_dim=args.input_dim, hidden_dim=args.hidden_dim, latent_dim=args.dim,
                               num_hops=args.hops, num_mlp_layers=args.mlps, dropout=args.dropout, **cfg['GAE']).cuda()
            self.model.load_state_dict(torch.load(os.path.join(self.dir_name, self.model_path).format(args.dim))['model_state'])
            self.model.eval()
            with torch.no_grad():
                print("length of the dataset: {}".format(len(self.dataset)))
                self.f_path = os.path.join(self.dir_name, 'maxcut-{}'.format(self.model_path))
                if os.path.exists(self.f_path):
                    print('{} is already saved'.format(self.f_path))
                    exit()
                print('save to {}'.format(self.f_path))
                for ind in range(len(self.dataset)):
                    adj = torch.Tensor(self.dataset[ind]['adj_matrix']).unsqueeze(0).cuda()
                    ops = torch.Tensor(self.dataset[ind]['gate_matrix']).unsqueeze(0).cuda()
                    adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
                    x, _ = self.model._encoder(ops, adj)
                    self.embedding[ind] = {'feature': x.squeeze(0).mean(dim=0).cpu(), 'energy': float(0), 'time': float(0)}
                torch.save(self.embedding, self.f_path)
                print("finish features extraction")
                exit()
        else:
            self.f_path = os.path.join(self.dir_name, self.emb_path)
            if not os.path.exists(self.f_path):
                print('{} is not saved, please save it first!'.format(self.f_path))
                exit()
            print("load feature embedding from: {}".format(self.f_path))
            self.embedding = torch.load(self.f_path)
            for ind in range(len(self.embedding)):
                self.features.append(self.embedding[ind]['feature'])
            self.features = torch.stack(self.features, dim=0)
            print('loading finished. pretrained embeddings shape: {}'.format(self.features.shape))


    def get_init_state(self):
        """
        :return: 1 x dim
        """
        random.seed(args.seed)
        rand_indices = random.randint(0, self.features.shape[0]-1)
        self.visited[rand_indices] = True
        return self.get_feature(rand_indices)
        

    def step(self, action):
        """
        action: 1 x dim
        self.features. N x dim
        """
        dist = torch.norm(self.features - action.cpu(), dim=1)
        knn = (-1 * dist).topk(dist.shape[0])
        min_dist, min_idx = knn.values, knn.indices
        count = 0
        while True:
            if len(self.visited) == dist.shape[0]:
                print("cannot find in the dataset")
                exit()
            if min_idx[count].item() not in self.visited:
                self.visited[min_idx[count].item()] = True
                break
            count += 1
        return self.get_feature(min_idx[count].item())


    def get_feature(self, index):
        circuit_list = self.dataset[index]['op_list']
        obj, time_cost, _= vqe_maxcut(self.graph, circuit_list, sample=False)
        self.embedding[index]['energy'] = obj[-1].item()
        self.embedding[index]['time'] = time_cost
        return index, self.features[index], self.embedding[index]['energy'], self.embedding[index]['time']
    
    
    def get_full_embedding(self):
        self.f_path = os.path.join(self.dir_name, self.emb_path)
        self.save_path = os.path.join(self.dir_name, '{}_full_embedding.pt'.format(self.emb_path[:-3]))
        if not os.path.exists(self.f_path):
            print('{} is not saved, please save it first!'.format(self.f_path))
            exit()
        if os.path.exists(self.save_path):
            print('{} is already saved, please check if it is necessary to reun it!'.format(self.save_path))
            exit()
        print("load feature embedding from: {}".format(self.f_path))
        self.embedding = torch.load(self.f_path)
        print('loading finished, begin getting full embedding.')
        for index in range(len(self.embedding)):
            print("index: {}".format(index))
            circuit_list = self.dataset[index]['op_list']
            obj, time_cost, _ = vqe_maxcut(self.graph, circuit_list, sample=False)
            self.embedding[index]['energy'] = obj[-1].item()
            self.embedding[index]['time'] = time_cost
        
        print('embedding is completed, begin saving full embedding, which takes a few minutes.')
        torch.save(self.embedding, self.save_path)
        print("finish full_embedding extraction")
        exit()


class Policy(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.saved_log_probs = []
        self.rewards = []


    def forward(self, input):
        x = F.relu(self.fc1(input))
        out = self.fc2(x)
        return out


class Policy_LSTM(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2):
        super(Policy_LSTM, self).__init__()
        self.lstm = torch.nn.LSTMCell(input_size=hidden_dim1, hidden_size=hidden_dim2)
        self.fc = nn.Linear(hidden_dim2, hidden_dim1)
        self.saved_log_probs = []
        self.rewards = []
        self.hx = None
        self.cx = None


    def forward(self, input):
        if self.hx is None and self.cx is None:
            self.hx, self.cx = self.lstm(input)
        else:
            self.hx, self.cx = self.lstm(input, (self.hx, self.cx))
        mean = self.fc(self.hx)
        return mean


def select_action(state, policy):
    """
     MVN based action selection.
    :param state: 1 x dim
    :param policy: policy network
    :return: action: 1 x dim
    """
    mean = policy(state.view(1, state.shape[0]))
    mvn = MultivariateNormal(mean, torch.eye(state.shape[0]).cuda())
    action = mvn.sample()
    policy.saved_log_probs.append(torch.mean(mvn.log_prob(action)))
    return action


def finish_episode(policy, optimizer, baseline):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards:
        R = r + args.gamma * R
        returns.append(R)
    returns = torch.Tensor(policy.rewards)
    returns = returns - baseline
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.mean(torch.stack(policy_loss, dim=0))
    avg_reward = sum(policy.rewards)/len(policy.rewards)

    print("basline: {}, average reward: {}, policy loss: {}".format(baseline, avg_reward, policy_loss.item()))
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    policy.hx = None
    policy.cx = None


def reinforce_search(env, args):
    """ implementation of circuit_feature-REINFORCE """
    policy = Policy_LSTM(args.dim, args.hidden_dim).cuda()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    counter = 0
    rt = 0
    acc_reward = 0
    GROUND_STATE_ENERGY = args.ground_state_energy
    MAX_SAMPLE = args.num_sample
    index, state, _, _ = env.get_init_state()
    CURR_BEST_ENERGY = 0
    energy_trace = []
    time_trace = []
    candidates = []
    avg_reward_trace = []
    avg_rewar_per_100 = []
    res = dict()

    res['gamma'] = args.gamma
    res['alpha'] = args.alpha
    res['initial_baseline'] = args.baseline
    res['penalty'] = args.penalty
    res['seed'] = args.seed
    res['latent_dim'] = args.dim
    res['threshold'] = args.threshold
    res['num_sample'] = args.num_sample
    res['ground_state_energy'] = args.ground_state_energy

    while counter < MAX_SAMPLE:
        for c in range(args.bs):
            state = state.cuda()
            action = select_action(state, policy)
            index, state, energy, time = env.step(action)
            reward = energy/GROUND_STATE_ENERGY
            penalty_factor = reward

            if reward < 0:
                reward = 0
            
            # reward penalty
            if args.penalty:
                if reward < args.threshold:
                    reward *= penalty_factor
            
            policy.rewards.append(reward)
            counter += 1
            rt += time
            acc_reward += reward
            print('counter: {}, energy: {}, energy reward: {}, time: {}'.format(counter, energy, reward, time))

            if energy < CURR_BEST_ENERGY:
                CURR_BEST_ENERGY = energy

            energy_trace.append(float(np.absolute(GROUND_STATE_ENERGY - CURR_BEST_ENERGY)))
            time_trace.append(float(rt))

            if reward >= args.threshold:
                candidates.append({"index": index, "energy": energy, "time": time})

            if counter % 100 == 0:
                print('current number of candidates {}'.format(len(candidates)))
                avg_rewar_per_100.append(acc_reward / 100)
                acc_reward = 0

            if counter >= MAX_SAMPLE:
                break

        avg_reward = sum(policy.rewards)/len(policy.rewards)
        avg_reward_trace.append(avg_reward)

        # adaptive baseline
        args.baseline = args.alpha * args.baseline + (1 - args.alpha) * avg_reward
        #args.baseline = args.alpha * args.baseline + (1 - args.alpha) * max(avg_reward, args.baseline)
        
        # adaptive batch size
        if avg_reward < 0.8:
            args.bs = 8
        elif avg_reward < 0.9:
            args.bs = 16
        elif avg_reward < args.threshold:
            args.bs = 24
        else:
            args.bs = 32

        finish_episode(policy, optimizer, args.baseline)

    res['regret_energy'] = energy_trace
    res['runtime'] = time_trace
    res['avg_reward_per_bs'] = avg_reward_trace
    res['avg_reward_per_100'] = avg_rewar_per_100
    res['candidates'] = candidates
    res['num_candidates'] = len(candidates)
    save_path = os.path.join(args.output_path, 'dim{}'.format(args.dim))
    if not os.path.exists(save_path):
        os.mkdir(save_path, True)
    print('save to {}'.format(save_path))
    if args.emb_path.endswith('.pt'):
        s = args.emb_path[:-3]
    fh = open(os.path.join(save_path, 'run_{}_{}.json'.format(args.seed, s)),'w')
    json.dump(res, fh)
    fh.close()


if __name__ == '__main__':
    # TODO: change parameters
    parser = argparse.ArgumentParser(description="circuit_feature-REINFORCE")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor of returns (default 0.99)")
    parser.add_argument("--alpha", type=float, default=0.7, help="balance factor of baseline (default 0.8)")
    parser.add_argument("--baseline", type=float, default=0.9, help="rl adaptive baseline (default intialized value 0.75)")
    parser.add_argument("--penalty", type=bool, default=False, help="reward penalty when state is relative stable, otherwise not")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--cfg', type=int, default=4, help='configuration (default: 4)')
    parser.add_argument('--bs', type=int, default=16, help='initial batch size')
    parser.add_argument('--output_path', type=str, default='saved_logs\\rl\\maxcut', help='rl, rs or bo; fidelity, maxcut or vqe')
    parser.add_argument('--emb_path', type=str, default='maxcut-model-circuits_4_qubits.pt')
    parser.add_argument('--model_path', type=str, default='model-circuits_4_qubits.pt')
    parser.add_argument('--saved_maxcut', action="store_true", default=True)
    parser.add_argument("--threshold", type=float, default=0.95, help="maxcut threshold (default 0.95)")
    parser.add_argument("--num_sample", type=int, default=1000, help="The number of sample circuits")
    parser.add_argument('--input_dim', type=int, default=17)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=16, help='feature dimension (default: 16)')
    parser.add_argument('--hops', type=int, default=5)
    parser.add_argument('--mlps', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--ground_state_energy', type=float, default=-10, help="The ground state energy of the graph hamiltonian")

    graph = [(0, 1), (0, 3), (1, 2), (2, 3)]
    get_full_embedding = False
    args = parser.parse_args()
    cfg = configs[args.cfg]
    if args.num_sample == 100000:
        get_full_embedding = True
    env = Env('REINFORCE', args.seed, args.emb_path, args.model_path, graph, cfg, 
              data_path='circuit\\data\\data_4_qubits.json', save=args.saved_maxcut, full_embedding=get_full_embedding)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    reinforce_search(env, args)