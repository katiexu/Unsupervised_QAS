import os
import sys
sys.path.insert(0, os.getcwd())
import json
import random
import argparse
import numpy as np

from utils.utils import load_json
from circuit.fidelity import opt_fidelity
from circuit.circuit_manager import circuit_qnode


def random_search(target, dataset):
    """ implementation of circuit_random_search"""
    counter = 0
    rt = 0
    acc_reward = 0
    visited = []
    BEST_FIDELITY = 1
    MAX_SAMPLE = args.num_sample
    CURR_BEST_FIDELITY = 0
    fidelity_trace = []
    time_trace = []
    avg_reward_trace = []
    candidates = []
    res = dict()

    res['seed'] = args.seed
    res['penalty'] = args.penalty
    res['threshold'] = args.threshold
    res['num_sample'] = args.num_sample

    while counter < MAX_SAMPLE:
        rand_indices = random.randint(0, len(dataset)-1)
        if rand_indices not in visited:
            visited.append(rand_indices)
        else:
            continue
        circuit_list = dataset[rand_indices]["op_list"]
        fidelities, time, _ = opt_fidelity(target, circuit_list)
        fidelity = -fidelities[-1].item()
        reward = fidelity
        penalty_factor = reward

        if reward < 0:
            reward = 0
        
        if args.penalty and reward < args.threshold:
            reward *= penalty_factor
        
        counter += 1
        rt += time
        acc_reward += reward
        print('counter: {}, fidelity: {}, time: {}'.format(counter, fidelity, time))

        if fidelity > CURR_BEST_FIDELITY:
            CURR_BEST_FIDELITY = fidelity

        fidelity_trace.append(float(BEST_FIDELITY - CURR_BEST_FIDELITY))
        time_trace.append(float(rt))

        if fidelity >= args.threshold:
            candidates.append({"index": rand_indices, "fidelity": fidelity, "time": time})
        
        if counter % 100 == 0:
            print('current number of candidates {}'.format(len(candidates)))
            avg_reward_trace.append(acc_reward / 100)
            acc_reward = 0

        if counter >= MAX_SAMPLE:
            break

    res['regret_fidelity'] = fidelity_trace
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
    parser.add_argument("--penalty", type=bool, default=False, help="reward penalty when state is relative stable, otherwise not")
    parser.add_argument("--data_path", type=str, default="circuit\\data\\data_4_qubits.json", help="the dataset path of circuits")
    parser.add_argument('--output_path', type=str, default='saved_logs\\rs\\fidelity', help='rl, rs or bo; fidelity, maxcut or vqe')
    parser.add_argument('--emb_path', type=str, default='fidelity-rs-circuits_4_qubits.pt')
    parser.add_argument("--threshold", type=float, default=0.95, help="fidelity threshold (default 0.95)")
    parser.add_argument("--num_sample", type=int, default=1000, help="The number of sample circuits")

    args = parser.parse_args()
    print("load data set from: {}".format(args.data_path))
    dataset = load_json(args.data_path) 
    target = circuit_qnode(load_json("circuit\\data\\data_test.json")[0]["op_list"])
    np.random.seed(args.seed)
    random_search(target, dataset)