import os
import sys
sys.path.insert(0, os.getcwd())
import json
import time as tim
import random
import argparse
import torch
import numpy as np
import pennylane as qml


from utils.utils import load_json
from circuit.fidelity import opt_fidelity
from circuit.maxcut import vqe_maxcut
from circuit.vqe import vqe
from circuit.circuit_manager import circuit_qnode
from models.predictor import GSQASPredictor, GSQASPredictor_prediction


def random_search(embedding, dataset, args, target=None, graph=None, hamiltonian=None):
    """ implementation of circuit_random_search"""
    filtered_embedding_num = 0
    filtered_candidate_num = 0
    labelled_circuit_pretraining_time = 0
    predictor_training_time = 0
    counter = 0
    rt = 0
    acc_reward = 0
    visited = []
    fidelity_trace = []
    energy_trace = []
    time_trace = []
    avg_reward_trace = []
    candidates = []
    res = dict()
    app = args.emb_path[:args.emb_path.find('-')]

    if app == "fidelity":
        BEST_FIDELITY = 1
        MAX_SAMPLE = args.num_sample
        CURR_BEST_FIDELITY = 0
        T1 = tim.time()
        selected_index, count, train_index, corr_search = GSQASPredictor_prediction(embedding, args)
        T2 = tim.time()
        for id in train_index:
            labelled_circuit_pretraining_time += embedding[id]['time']
        predictor_training_time = T2-T1
        filtered_embedding_num = len(selected_index)
        filtered_candidate_num = count

        res['seed'] = args.seed
        res['penalty'] = args.penalty
        res['threshold'] = args.threshold
        res['num_sample'] = args.num_sample
        res["total_num"] = args.sample_num
        res["filtered_embedding_num"] = filtered_embedding_num
        res["filtered_candidate_num"] = filtered_candidate_num
        res["predictor_training_time"] = predictor_training_time
        res["labelled_circuit_pretraining_time"] = labelled_circuit_pretraining_time
        res["pearson_corr"] = corr_search.statistic
        res["pearson_pvalue"] = corr_search.pvalue

        while counter < MAX_SAMPLE:
            rand_indices = selected_index[np.random.randint(0, filtered_embedding_num)].item()
            if rand_indices not in visited:
                visited.append(rand_indices)
            else:
                continue
            if embedding[rand_indices]['time'] == 0:
                circuit_list = dataset[rand_indices]["op_list"]
                fidelities, time, _ = opt_fidelity(target, circuit_list)
                fidelity = -fidelities[-1].item()
                reward = fidelity
                penalty_factor = reward
            else:
                fidelity, time = embedding[rand_indices]['fidelity'], embedding[rand_indices]['time']
                reward = fidelity
                penalty_factor = reward
            
            if reward < 0:
                reward = 0
            elif reward > 1:
                reward = 1
            
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
        save_path = os.path.join(args.output_path, 'dim{}'.format(args.dim))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        print('save to {}'.format(save_path))
        if args.file_path.endswith('.pt'):
            s = args.file_path[:-3]
        fh = open(os.path.join(save_path, 'run_{}_{}_{}.json'.format(args.seed, args.filter_threshold, s)),'w')
        json.dump(res, fh)
        fh.close()
    else:
        GROUND_STATE_ENERGY = args.ground_state_energy
        MAX_SAMPLE = args.num_sample
        CURR_BEST_ENERGY = 0
        T1 = tim.time()
        selected_index, count, train_index, corr_search = GSQASPredictor_prediction(embedding, args)
        T2 = tim.time()
        for id in train_index:
            labelled_circuit_pretraining_time += embedding[id]['time']
        predictor_training_time = T2-T1
        filtered_embedding_num = len(selected_index)
        filtered_candidate_num = count

        res['seed'] = args.seed
        res['penalty'] = args.penalty
        res['threshold'] = args.threshold
        res['num_sample'] = args.num_sample
        res["total_num"] = args.sample_num
        res["filtered_embedding_num"] = filtered_embedding_num
        res["filtered_candidate_num"] = filtered_candidate_num
        res["predictor_training_time"] = predictor_training_time
        res["labelled_circuit_pretraining_time"] = labelled_circuit_pretraining_time
        res["pearson_corr"] = corr_search.statistic
        res["pearson_pvalue"] = corr_search.pvalue

        while counter < MAX_SAMPLE:
            rand_indices = selected_index[np.random.randint(0, filtered_embedding_num)].item()
            if rand_indices not in visited:
                visited.append(rand_indices)
            else:
                continue
            if embedding[rand_indices]['time'] == 0:
                circuit_list = dataset[rand_indices]["op_list"]
                if app == "maxcut":
                    obj, time, _ = vqe_maxcut(graph, circuit_list, sample=False)
                elif app == "vqe":
                    obj, time, _ = vqe(hamiltonian, circuit_list, max_iterations=100)
                energy = obj[-1].item()
                reward = energy/GROUND_STATE_ENERGY
                penalty_factor = reward
            else:
                energy, time = embedding[rand_indices]['energy'], embedding[rand_indices]['time']
                reward = energy/GROUND_STATE_ENERGY
                penalty_factor = reward
            
            if reward < 0:
                reward = 0
            elif reward > 1:
                reward = 1
            
            if args.penalty and reward < args.threshold:
                reward *= penalty_factor
            
            counter += 1
            rt += time
            acc_reward += reward
            print('counter: {}, energy: {}, reward: {}, time: {}'.format(counter, energy, reward, time))

            if energy < CURR_BEST_ENERGY:
                if energy < GROUND_STATE_ENERGY:
                    CURR_BEST_ENERGY = GROUND_STATE_ENERGY
                else:
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
        save_path = os.path.join(args.output_path, 'dim{}'.format(args.dim))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        print('save to {}'.format(save_path))
        if args.file_path.endswith('.pt'):
            s = args.file_path[:-3]
        fh = open(os.path.join(save_path, 'run_{}_{}_{}.json'.format(args.seed, args.filter_threshold, s)),'w')
        json.dump(res, fh)
        fh.close()


if __name__ == '__main__':
    # TODO: change parameters
    parser = argparse.ArgumentParser(description="circuit_random_search")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--penalty", action="store_true", default=False, help="reward penalty when state is relative stable, otherwise not")
    parser.add_argument('--dim', type=int, default=16, help='feature dimension (default: 16)')
    parser.add_argument('--dir_name', type=str, default='pretrained\\dim-16')
    parser.add_argument("--data_path", type=str, default="circuit\\data\\data_4_qubits.json", help="the dataset path of circuits")
    parser.add_argument('--output_path', type=str, default='saved_logs\\gsqas_predictor\\fidelity', help='predictor_based rs')
    parser.add_argument('--file_path', type=str, default='fidelity-gsqas_predictor-circuits_4_qubits.pt')
    parser.add_argument('--emb_path', type=str, default='fidelity-model-circuits_4_qubits.pt')
    parser.add_argument("--sample_num", type=int, default=100000, help="total number of samples (default 100000)")
    parser.add_argument("--threshold", type=float, default=0.95, help="fidelity threshold (default 0.95)")
    parser.add_argument("--filter_threshold", type=float, default=0.4, help="fidelity predictor-filtered threshold (default 0.4)")
    parser.add_argument("--num_sample", type=int, default=1000, help="The number of sample circuits")
    parser.add_argument('--ground_state_energy', type=float, default=-10, help="The ground state energy of the graph hamiltonian")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    num_dataset = args.sample_num
    f_path = os.path.join(args.dir_name, '{}_full_embedding.pt'.format(args.emb_path[:-3]))
    if not os.path.exists(f_path):
        print('{} is not saved, please save it first!'.format(f_path))
        exit()
    print("load full feature embedding from: {}".format(f_path))
    embedding = torch.load(f_path)
    print("load finished")

    app = args.emb_path[:args.emb_path.find('-')]
    #print("load data set from: {}".format(args.data_path))
    #dataset = load_json(args.data_path)
    dataset=None
    if app == 'fidelity':
        target = circuit_qnode(load_json("circuit\\data\\data_test.json")[0]["op_list"])
        random_search(embedding, dataset, args, target=target)
    elif app == 'maxcut':
        graph = [(0, 1), (0, 3), (1, 2), (2, 3)]
        random_search(embedding, dataset, args, graph=graph)
    else:
        symbols = ["H", "H"]
        coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
        H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
        random_search(embedding, dataset, args, hamiltonian=H)