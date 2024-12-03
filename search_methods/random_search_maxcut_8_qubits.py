import os
import sys
sys.path.insert(0, os.getcwd())
import json
import random
import argparse
import numpy as np

from utils.utils import load_json
from circuit.maxcut import vqe_maxcut


def random_search(graph, dataset):
    """ implementation of circuit_random_search"""
    counter = 0
    rt = 0
    acc_reward = 0
    visited = []
    GROUND_STATE_ENERGY = args.ground_state_energy
    MAX_SAMPLE = args.num_sample
    CURR_BEST_ENERGY = 0
    energy_trace = []
    time_trace = []
    avg_reward_trace = []
    candidates = []
    res = dict()

    res['seed'] = args.seed
    res['penalty'] = args.penalty
    res['threshold'] = args.threshold
    res['num_sample'] = args.num_sample
    res['ground_state_energy'] = args.ground_state_energy

    while counter < MAX_SAMPLE:
        rand_indices = random.randint(0, len(dataset)-1)
        if rand_indices not in visited:
            visited.append(rand_indices)
        else:
            continue
        circuit_list = dataset[rand_indices]["op_list"]
        obj, time, _ = vqe_maxcut(graph, circuit_list, sample=False)
        energy = obj[-1].item()
        reward = energy/GROUND_STATE_ENERGY
        penalty_factor = reward

        if reward < 0:
            reward = 0
        
        if args.penalty and reward < args.threshold:
            reward *= penalty_factor
        
        counter += 1
        rt += time
        acc_reward += reward
        print('counter: {}, energy: {}, reward: {}, time: {}'.format(counter, energy, reward, time))

        if energy < CURR_BEST_ENERGY:
            CURR_BEST_ENERGY = energy

        energy_trace.append(float(np.absolute(GROUND_STATE_ENERGY - CURR_BEST_ENERGY)))
        time_trace.append(float(rt))

        if reward >= args.threshold:
            candidates.append({"index": rand_indices, "energy": energy, "time": time})

        if counter % 100 == 0:
            print('current number of candidates {}'.format(len(candidates)))
            avg_reward_trace.append(acc_reward / 100)
            acc_reward = 0

        if counter >= MAX_SAMPLE:
            break

    res['regret_energy'] = energy_trace
    res['runtime'] = time_trace
    res['avg_reward_per_100'] = avg_reward_trace
    res['candidates'] = candidates
    res['num_candidates'] = len(candidates)
    save_path = args.output_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print('save to {}'.format(save_path))
    if args.emb_path.endswith('.pt'):
        s = args.emb_path[:-3]
    fh = open(os.path.join(save_path, 'run_{}_{}.json'.format(args.seed, s)),'w')
    json.dump(res, fh)
    fh.close()


if __name__ == '__main__':
    # TODO: change parameters
    parser = argparse.ArgumentParser(description="circuit_random_search")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--penalty", type=bool, default=True, help="reward penalty when state is relative stable, otherwise not")
    parser.add_argument("--data_path", type=str, default="circuit\\data\\data_8_qubits_20_gates.json", help="the dataset path of circuits")
    parser.add_argument('--output_path', type=str, default='saved_logs\\rs\\maxcut', help='rl, rs or bo; fidelity, maxcut or vqe')
    parser.add_argument('--emb_path', type=str, default='maxcut-rs-circuits_8_qubits_20_gates.pt')
    parser.add_argument("--threshold", type=float, default=0.925, help="maxcut threshold (default 0.95)")
    parser.add_argument("--num_sample", type=int, default=1000, help="The number of sample circuits")
    parser.add_argument('--ground_state_energy', type=float, default=-52, help="The ground state energy of the graph hamiltonian")

    args = parser.parse_args()
    print("load data set from: {}".format(args.data_path))
    dataset = load_json(args.data_path) 
    graph = [(0,1), (0,6), (0,7), (1,2), (1,6), (2,4), (2,7), (3,7), (3,6), (4,5), (4,6), (5,1), (5,6), (5,7), (6,7)]
    np.random.seed(args.seed)
    random_search(graph, dataset)