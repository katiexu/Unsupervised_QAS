import os
import sys
sys.path.insert(0, os.getcwd())
import json
import torch
import argparse
import numpy as np

from pybnn.dngo import DNGO
from utils.utils import load_json
from collections import defaultdict
from torch.distributions import Normal
from circuit.fidelity import opt_fidelity
from circuit.circuit_manager import circuit_qnode


def load_features(embedding_path):
    if not os.path.exists(embedding_path):
        print('{} is not saved, please save it first with the REINFORCE version!'.format(embedding_path))
        exit()
    embedding = torch.load(embedding_path)
    print('load circuit features from {}'.format(embedding_path))
    ind_list = range(len(embedding))
    features = [embedding[ind]['feature'] for ind in ind_list]
    fidelity = [embedding[ind]['fidelity'] for ind in ind_list]
    training_time = [embedding[ind]['time'] for ind in ind_list]
    features = torch.stack(features, dim=0)
    fidelity = torch.Tensor(fidelity)
    training_time = torch.Tensor(training_time)
    print('loading finished. pretrained embeddings shape {}'.format(features.shape))
    return features, fidelity, training_time


def get_features(target, dataset, index, features, fidelity, training_time):
    circuit_list = dataset[index]['op_list']
    obj, time_cost, _ = opt_fidelity(target, circuit_list)
    fidelity[index] = -obj[-1].item()
    training_time[index] = time_cost
    print("fidelity: {}, training_time: {}".format(fidelity[index], training_time[index]))
    return index, features[index], fidelity[index], training_time[index]


def get_init_samples(target, dataset, features, fidelity, training_time, visited):
    np.random.seed(args.seed)
    init_inds = np.random.permutation(list(range(features.shape[0])))[:args.init_size]
    init_inds = torch.Tensor(init_inds).long()
    init_feat_samples = torch.zeros((init_inds.shape[0], features.shape[1]), dtype=torch.float64)
    init_fidelity_samples = torch.zeros_like(init_inds, dtype=torch.float64)
    init_time_samples = torch.zeros_like(init_inds, dtype=torch.float64)
    for i in range(len(init_inds)):
        idx = init_inds[i].item()
        visited[idx] = True
        _, init_feat_samples[i], init_fidelity_samples[i], init_time_samples[i] = \
            get_features(target, dataset, idx, features, fidelity, training_time)
    return init_inds, init_feat_samples, init_fidelity_samples, init_time_samples, visited


def propose_location(ei, target, dataset, features, fidelity, training_time, visited):
    k = args.topk
    print('remaining length of indices set:', len(features) - len(visited))
    indices = torch.argsort(ei, descending=True)[0:2*k]
    ind_dedup = []
    for idx in indices:
        idx = idx.item()
        if idx not in visited:
            visited[idx] = True
            ind_dedup.append(idx)
        if len(ind_dedup) >= k:
            break
    ind_dedup = torch.Tensor(ind_dedup).long()
    proposed_feature = torch.zeros((ind_dedup.shape[0], features.shape[1]), dtype=torch.float64)
    proposed_fidelity = torch.zeros_like(ind_dedup, dtype=torch.float64)
    proposed_time = torch.zeros_like(ind_dedup, dtype=torch.float64)
    for i in range(len(ind_dedup)):
        idx = ind_dedup[i]
        _, proposed_feature[i], proposed_fidelity[i], proposed_time[i] = \
            get_features(target, dataset, idx, features, fidelity, training_time)
    return ind_dedup, proposed_feature, proposed_fidelity, proposed_time, visited


def expected_improvement_search(args, target, dataset):
    """ implementation of circuit-DNGO """
    BEST_FIDELITY = 1
    CURR_BEST_FIDELITY = 0
    MAX_SAMPLE = args.num_sample
    window_size = args.window_size
    counter = 0
    rt = 0
    acc_reward = 0
    avg_reward_trace = []
    candidates = []
    visited = {}
    best_trace = defaultdict(list)
    res = dict()

    res['seed'] = args.seed
    res['latent_dim'] = args.dim
    res['init_size'] = args.init_size
    res['window_size'] = args.window_size
    res['topk'] = args.topk
    res['penalty'] = args.penalty
    res['threshold'] = args.threshold
    res['objective'] = args.objective
    res['num_sample'] = args.num_sample

    features, fidelity, training_time = load_features(os.path.join('pretrained\dim-{}'.format(args.dim), args.emb_path))
    features, fidelity, training_time = features.cpu().detach().to(torch.float64), \
        fidelity.cpu().detach().to(torch.float64), training_time.cpu().detach().to(torch.float64)
    index_samples, feat_samples, fidelity_samples, time_samples, visited = \
        get_init_samples(target, dataset, features, fidelity, training_time, visited)
    reward_samples = fidelity_samples
    reward_samples = torch.where(reward_samples < 0, torch.zeros_like(reward_samples), reward_samples)
    if args.penalty:
        reward_samples = torch.where(args.penalty and reward_samples < args.threshold, reward_samples ** 2, reward_samples)

    for index, feat, curr_fidelity, curr_reward, t in zip(index_samples, feat_samples, fidelity_samples, reward_samples, time_samples):
        counter += 1
        rt += t.item()
        acc_reward += curr_reward.item()

        if curr_fidelity > CURR_BEST_FIDELITY:
            CURR_BEST_FIDELITY = curr_fidelity

        if curr_reward >= args.threshold:
            candidates.append({"index": index.item(), "fidelity": curr_fidelity.item(), "time": t.item()})

        best_trace['regret_fidelity'].append(float(BEST_FIDELITY - CURR_BEST_FIDELITY))
        best_trace['time'].append(rt)

    while counter < MAX_SAMPLE:
        print("counter:", counter)
        print("feat_samples:", feat_samples.shape)
        print("fidelity_samples:", fidelity_samples.shape)
        print("current best fidelity: {}".format(CURR_BEST_FIDELITY))
        print('current number of candidates {}'.format(len(candidates)))
        print("rt: {}".format(rt))

        model = DNGO(num_epochs=100, n_units=128, do_mcmc=False, normalize_output=False, rng=args.seed)
        model.train(X=feat_samples.numpy(), y=reward_samples.view(-1).numpy(), do_optimize=True)
        m = []
        v = []
        chunks = int(features.shape[0] / window_size)

        if features.shape[0] % window_size > 0:
            chunks += 1
        features_split = torch.split(features, window_size, dim=0)
        for i in range(chunks):
            m_split, v_split = model.predict(features_split[i].numpy())
            m.extend(list(m_split))
            v.extend(list(v_split))

        mean = torch.Tensor(m)
        sigma = torch.Tensor(v)
        u = (mean - torch.Tensor([args.objective]).expand_as(mean)) / sigma
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        index_next, feat_next, fidelity_next, time_next, visited = \
            propose_location(ei, target, dataset, features, fidelity, training_time, visited)
        reward_next = fidelity_next
        reward_next = torch.where(reward_next < 0, torch.zeros_like(reward_next), reward_next)
        if args.penalty:
            reward_next = torch.where(args.penalty and reward_next < args.threshold, reward_next ** 2, reward_next)

        # add proposed networks to the pool
        for index, feat, curr_fidelity, curr_reward, t in zip(index_next, feat_next, fidelity_next, reward_next, time_next):
            counter += 1
            rt += t.item()
            acc_reward += curr_reward.item()

            if curr_fidelity > CURR_BEST_FIDELITY:
                CURR_BEST_FIDELITY = curr_fidelity
            
            if curr_reward >= args.threshold:
                candidates.append({"index": index.item(), "fidelity": curr_fidelity.item(), "time": t.item()})
            
            feat_samples = torch.cat((feat_samples, feat.view(1, -1)), dim=0)
            fidelity_samples = torch.cat((fidelity_samples.view(-1, 1), curr_fidelity.view(1, 1)), dim=0)
            reward_samples = torch.cat((reward_samples.view(-1, 1), curr_reward.view(1, 1)), dim=0)
            
            best_trace['regret_fidelity'].append(float(BEST_FIDELITY - CURR_BEST_FIDELITY))
            best_trace['time'].append(rt)

            if counter % 100 == 0:
                avg_reward_trace.append(acc_reward / 100)
                acc_reward = 0

            if counter >= MAX_SAMPLE:
                break
    
    res['regret_fidelity'] = best_trace['regret_fidelity']
    res['runtime'] = best_trace['time']
    res['avg_reward_per_100'] = avg_reward_trace
    res['candidates'] = candidates
    res['num_candidates'] = len(candidates)
    save_path = os.path.join(args.output_path, 'dim{}'.format(args.dim))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print('save to {}'.format(save_path))
    if args.emb_path.endswith('.pt'):
        s = args.emb_path[:-3]
    fh = open(os.path.join(save_path, 'run_{}_{}.json'.format(args.seed, s)),'w')
    json.dump(res, fh)
    fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="circuit-DNGO")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--cfg', type=int, default=4, help='configuration (default: 4)')
    parser.add_argument('--dim', type=int, default=32, help='feature dimension')
    parser.add_argument('--init_size', type=int, default=16, help='init samples')
    parser.add_argument('--window_size', type=int, default=200, help='window size')
    parser.add_argument('--topk', type=int, default=8, help='acquisition samples')
    parser.add_argument("--penalty", type=bool, default=False, help="reward penalty when state is relative stable, otherwise not")
    parser.add_argument('--output_path', type=str, default='saved_logs\\bo\\fidelity', help='rl, rs or bo; fidelity, maxcut or vqe')
    parser.add_argument('--emb_path', type=str, default='fidelity-model-circuits_8_qubits_20_gates.pt')
    parser.add_argument("--threshold", type=float, default=0.75, help="fidelity threshold (default 0.75)")
    parser.add_argument("--objective", type=float, default=0.7, help="expected objective optimization (default 0.7)")
    parser.add_argument("--num_sample", type=int, default=1000, help="The number of sample circuits")

    args = parser.parse_args()
    dataset = load_json('circuit\\data\\data_8_qubits_20_gates.json')
    dataset_test = load_json("circuit\\data\\data_8_qubits_test.json")
    target = circuit_qnode(dataset_test[0]["op_list"])
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    expected_improvement_search(args, target, dataset)