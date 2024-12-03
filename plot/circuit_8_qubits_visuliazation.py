import os
import sys
sys.path.insert(0, os.getcwd())
import json
import argparse
import pennylane as qml
import matplotlib.pyplot as plt

from utils.utils import load_json
from circuit.circuit_manager import circuit_qnode

    
# get relevant information about candidate circuits of 8-qubit application experiments with corresponding search methods
def get_8_qubit_candidates(app: str, search_method: str, dim: int = None):
    '''
        app: So far, it can be "fidelity", "maxcut", "vqe"
        num_sample: the number of sample circuits
        dim: latent dimension, which should be claimed for rl and bo
        search_method: So far, it can be "rs", "rl", "bo"
    '''
    if search_method in ["rl", "bo"] and dim == None:
        raise ValueError("when using rl or bo, the dim must be claimed!")
    
    candidates = []
    index_list = []
    for seed in range(1, 51):
        if search_method == "rs":
            f_name = 'saved_logs\\{}\\{}\\run_{}_{}-rs-circuits_8_qubits_20_gates.json'.format(search_method, app, seed, app)
        else:
            f_name = 'saved_logs\\{}\\{}\\dim{}\\run_{}_{}-model-circuits_8_qubits_20_gates.json'.format(search_method, app, dim, seed, app)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        data = json.load(f)
        for candidate in data['candidates']:
            if candidate['index'] not in index_list:
                index_list.append(candidate['index'])
                candidates.append(candidate)
        f.close()

    if app == "fidelity":
        candidates.sort(key = lambda x: (x['fidelity'], x['time']), reverse=True)
    else:
        candidates.sort(key = lambda x: (x['energy'], x['time']))
    return candidates
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="8-qubits_top5_candidate_visulization")
    parser.add_argument('--save_path', type=str, default='saved_figs')

    args = parser.parse_args()

    fidelity_8_rs_candidates = get_8_qubit_candidates("fidelity", "rs")
    fidelity_8_rl_candidates = get_8_qubit_candidates("fidelity", "rl", dim=32)
    fidelity_8_bo_candidates = get_8_qubit_candidates("fidelity", "bo", dim=32)
    fidelity_8_all_candidates = fidelity_8_rs_candidates + fidelity_8_rl_candidates + fidelity_8_bo_candidates
    fidelity_8_all_candidates.sort(key = lambda x: (x['fidelity'], x['time']), reverse=True)
    fidelity_8_best_candidate = fidelity_8_all_candidates[0]

    maxcut_8_rs_candidates = get_8_qubit_candidates("maxcut", "rs")
    maxcut_8_rl_candidates = get_8_qubit_candidates("maxcut", "rl", dim=32)
    maxcut_8_bo_candidates = get_8_qubit_candidates("maxcut", "bo", dim=32)
    maxcut_8_all_candidates = maxcut_8_rs_candidates + maxcut_8_rl_candidates + maxcut_8_bo_candidates
    maxcut_8_all_candidates.sort(key = lambda x: (x['energy'], x['time']))
    maxcut_8_best_candidate = maxcut_8_all_candidates[0]

    vqe_8_rs_candidates = get_8_qubit_candidates("vqe", "rs")
    vqe_8_rl_candidates = get_8_qubit_candidates("vqe", "rl", dim=32)
    vqe_8_bo_candidates = get_8_qubit_candidates("vqe", "bo", dim=32)
    vqe_8_all_candidates = vqe_8_rs_candidates + vqe_8_rl_candidates + vqe_8_bo_candidates
    vqe_8_all_candidates.sort(key = lambda x: (x['energy'], x['time']))
    vqe_8_best_candidate = vqe_8_all_candidates[0]

    test_dataset = load_json("circuit\\data\\data_8_qubits_test.json")
    dataset_8 = load_json("circuit\\data\\data_8_qubits_20_gates.json")

    fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(8), show_all_wires=True, decimals=3)(test_dataset[0]['op_list'])
    plt.savefig(os.path.join(args.save_path, "fidelity\\circuits",
                f"8-qubits-fidelity_target.png"), dpi=400)
    plt.close()

    for i in range(5):
        fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(8), show_all_wires=True)(dataset_8[fidelity_8_rs_candidates[i]['index']]['op_list'])
        plt.savefig(os.path.join(args.save_path, "fidelity\\circuits", "rs", 
                    f"8-qubits-fidelity_candidate_{i+1}_{fidelity_8_rs_candidates[i]['index']}.png"), dpi=400)
        plt.close()

        fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(8), show_all_wires=True)(dataset_8[fidelity_8_rl_candidates[i]['index']]['op_list'])
        plt.savefig(os.path.join(args.save_path, "fidelity\\circuits", "rl",
                    f"8-qubits-fidelity_candidate_{i+1}_{fidelity_8_rl_candidates[i]['index']}.png"), dpi=400)
        plt.close()

        fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(8), show_all_wires=True)(dataset_8[fidelity_8_bo_candidates[i]['index']]['op_list'])
        plt.savefig(os.path.join(args.save_path, "fidelity\\circuits", "bo",
                    f"8-qubits-fidelity_candidate_{i+1}_{fidelity_8_bo_candidates[i]['index']}.png"), dpi=400)
        plt.close()

        fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(8), show_all_wires=True)(dataset_8[maxcut_8_rs_candidates[i]['index']]['op_list'])
        plt.savefig(os.path.join(args.save_path, "maxcut\\circuits", "rs",
                    f"8-qubits-maxcut_candidate_{i+1}_{maxcut_8_rs_candidates[i]['index']}.png"), dpi=400)
        plt.close()

        fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(8), show_all_wires=True)(dataset_8[maxcut_8_rl_candidates[i]['index']]['op_list'])
        plt.savefig(os.path.join(args.save_path, "maxcut\\circuits", "rl",
                    f"8-qubits-maxcut_candidate_{i+1}_{maxcut_8_rl_candidates[i]['index']}.png"), dpi=400)
        plt.close()

        fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(8), show_all_wires=True)(dataset_8[maxcut_8_bo_candidates[i]['index']]['op_list'])
        plt.savefig(os.path.join(args.save_path, "maxcut\\circuits", "bo",
                    f"8-qubits-maxcut_candidate_{i+1}_{maxcut_8_bo_candidates[i]['index']}.png"), dpi=400)
        plt.close()

        fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(8), show_all_wires=True)(dataset_8[vqe_8_rs_candidates[i]['index']]['op_list'])
        plt.savefig(os.path.join(args.save_path, "vqe\\circuits", "rs",
                    f"8-qubits-vqe_candidate_{i+1}_{vqe_8_rs_candidates[i]['index']}.png"), dpi=400)
        plt.close()

        fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(8), show_all_wires=True)(dataset_8[vqe_8_rl_candidates[i]['index']]['op_list'])
        plt.savefig(os.path.join(args.save_path, "vqe\\circuits", "rl",
                    f"8-qubits-vqe_candidate_{i+1}_{vqe_8_rl_candidates[i]['index']}.png"), dpi=400)
        plt.close()

        fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(8), show_all_wires=True)(dataset_8[vqe_8_bo_candidates[i]['index']]['op_list'])
        plt.savefig(os.path.join(args.save_path, "vqe\\circuits", "bo",
                    f"8-qubits-vqe_candidate_{i+1}_{vqe_8_bo_candidates[i]['index']}.png"), dpi=400)
        plt.close()
    
    fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(8), show_all_wires=True)(dataset_8[fidelity_8_best_candidate['index']]['op_list'])
    plt.savefig(os.path.join(args.save_path, "fidelity\\circuits", 
                    f"8-qubits-fidelity_best_candidate_{fidelity_8_best_candidate['index']}.png"), dpi=400)
    plt.close()

    fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(8), show_all_wires=True)(dataset_8[maxcut_8_best_candidate['index']]['op_list'])
    plt.savefig(os.path.join(args.save_path, "maxcut\\circuits", 
                    f"8-qubits-maxcut_best_candidate_{maxcut_8_best_candidate['index']}.png"), dpi=400)
    plt.close()

    fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(8), show_all_wires=True)(dataset_8[vqe_8_best_candidate['index']]['op_list'])
    plt.savefig(os.path.join(args.save_path, "vqe\\circuits", 
                    f"8-qubits-vqe_best_candidate_{vqe_8_best_candidate['index']}.png"), dpi=400)
    plt.close()