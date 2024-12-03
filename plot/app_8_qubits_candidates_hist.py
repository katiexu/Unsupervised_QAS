import os
import sys
sys.path.insert(0, os.getcwd())
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# get the number of candidate circuits of 8-qubits application experiments with corresponding search methods
def get_8_qubits_app_num_candidates(app: str, num_sample: int, search_method: str, dim: int = None):
    '''
        app: So far, it can be "fidelity", "maxcut", "vqe"
        num_sample: the number of sample circuits
        dim: latent dimension, which should be claimed for rl and bo
        search_method: So far, it can be "rs", "rl", "bo"
    '''
    if search_method in ["rl", "bo"] and dim == None:
        raise ValueError("when using rl or bo, the dim must be claimed!")
    
    num_candidates = []
    for seed in range(1, 51):
        if search_method == "rs":
            f_name = 'saved_logs\\{}\\{}\\run_{}_{}-rs-circuits_8_qubits_20_gates.json'.format(search_method, app, seed, app)
        else:
            f_name = 'saved_logs\\{}\\{}\\dim{}\\run_{}_{}-model-circuits_8_qubits_20_gates.json'.format(search_method, app, dim, seed, app)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        data = json.load(f)
        if num_sample != data["num_sample"]:
            raise ValueError("Please check if using correct 'num_sample' matching the experiments!")
        num_candidates.append(data['num_candidates'])
        f.close()

    avg_num_candidates = sum(num_candidates) // 50
    max_num_candidates = max(num_candidates)
    min_num_candidates = min(num_candidates)
    return avg_num_candidates, max_num_candidates, min_num_candidates

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="8-qubits_app_experiments_num_candidates")
    parser.add_argument('--dim', type=int, default=32, help='feature dimension')
    parser.add_argument("--num_sample", type=int, default=1000, help="The number of sample circuits")
    parser.add_argument('--save_path', type=str, default='saved_figs')

    args = parser.parse_args()

    fidelity_8_rs_avg, fidelity_8_rs_max, fidelity_8_rs_min = get_8_qubits_app_num_candidates("fidelity", args.num_sample, "rs")
    maxcut_8_rs_avg, maxcut_8_rs_max, maxcut_8_rs_min = get_8_qubits_app_num_candidates("maxcut", args.num_sample, "rs")
    vqe_8_rs_avg, vqe_8_rs_max, vqe_8_rs_min = get_8_qubits_app_num_candidates("vqe", args.num_sample, "rs")

    fidelity_8_rs_lower_err = fidelity_8_rs_avg - fidelity_8_rs_min
    fidelity_8_rs_upper_err = fidelity_8_rs_max - fidelity_8_rs_avg
    maxcut_8_rs_lower_err = maxcut_8_rs_avg - maxcut_8_rs_min
    maxcut_8_rs_upper_err = maxcut_8_rs_max - maxcut_8_rs_avg
    vqe_8_rs_lower_err = vqe_8_rs_avg - vqe_8_rs_min
    vqe_8_rs_upper_err = vqe_8_rs_max - vqe_8_rs_avg

    fidelity_8_rl_avg, fidelity_8_rl_max, fidelity_8_rl_min = get_8_qubits_app_num_candidates("fidelity", args.num_sample, "rl", args.dim)
    maxcut_8_rl_avg, maxcut_8_rl_max, maxcut_8_rl_min = get_8_qubits_app_num_candidates("maxcut", args.num_sample, "rl", args.dim)
    vqe_8_rl_avg, vqe_8_rl_max, vqe_8_rl_min = get_8_qubits_app_num_candidates("vqe", args.num_sample, "rl", args.dim)

    fidelity_8_rl_lower_err = fidelity_8_rl_avg - fidelity_8_rl_min
    fidelity_8_rl_upper_err = fidelity_8_rl_max - fidelity_8_rl_avg
    maxcut_8_rl_lower_err = maxcut_8_rl_avg - maxcut_8_rl_min
    maxcut_8_rl_upper_err = maxcut_8_rl_max - maxcut_8_rl_avg
    vqe_8_rl_lower_err = vqe_8_rl_avg - vqe_8_rl_min
    vqe_8_rl_upper_err = vqe_8_rl_max - vqe_8_rl_avg

    fidelity_8_bo_avg, fidelity_8_bo_max, fidelity_8_bo_min = get_8_qubits_app_num_candidates("fidelity", args.num_sample, "bo", args.dim)
    maxcut_8_bo_avg, maxcut_8_bo_max, maxcut_8_bo_min = get_8_qubits_app_num_candidates("maxcut", args.num_sample, "bo", args.dim)
    vqe_8_bo_avg, vqe_8_bo_max, vqe_8_bo_min = get_8_qubits_app_num_candidates("vqe", args.num_sample, "bo", args.dim)

    fidelity_8_bo_lower_err = fidelity_8_bo_avg - fidelity_8_bo_min
    fidelity_8_bo_upper_err = fidelity_8_bo_max - fidelity_8_bo_avg
    maxcut_8_bo_lower_err = maxcut_8_bo_avg - maxcut_8_bo_min
    maxcut_8_bo_upper_err = maxcut_8_bo_max - maxcut_8_bo_avg
    vqe_8_bo_lower_err = vqe_8_bo_avg - vqe_8_bo_min
    vqe_8_bo_upper_err = vqe_8_bo_max - vqe_8_bo_avg

    labels = ["State Preparation", "Maxcut", "Quantum Chemistry"]
    rs_8_avg = [fidelity_8_rs_avg, maxcut_8_rs_avg, vqe_8_rs_avg]
    rl_8_avg = [fidelity_8_rl_avg, maxcut_8_rl_avg, vqe_8_rl_avg]
    bo_8_avg = [fidelity_8_bo_avg, maxcut_8_bo_avg, vqe_8_bo_avg]

    rs_8_min = [fidelity_8_rs_min, maxcut_8_rs_min, vqe_8_rs_min]
    rl_8_min = [fidelity_8_rl_min, maxcut_8_rl_min, vqe_8_rl_min]
    bo_8_min = [fidelity_8_bo_min, maxcut_8_bo_min, vqe_8_bo_min]

    rs_8_max = [fidelity_8_rs_max, maxcut_8_rs_max, vqe_8_rs_max]
    rl_8_max = [fidelity_8_rl_max, maxcut_8_rl_max, vqe_8_rl_max]
    bo_8_max = [fidelity_8_bo_max, maxcut_8_bo_max, vqe_8_bo_max]

    rs_8_err = [[fidelity_8_rs_lower_err, maxcut_8_rs_lower_err, vqe_8_rs_lower_err], 
                [fidelity_8_rs_upper_err, maxcut_8_rs_upper_err, vqe_8_rs_upper_err]]
    rl_8_err = [[fidelity_8_rl_lower_err, maxcut_8_rl_lower_err, vqe_8_rl_lower_err], 
                [fidelity_8_rl_upper_err, maxcut_8_rl_upper_err, vqe_8_rl_upper_err]]
    bo_8_err = [[fidelity_8_bo_lower_err, maxcut_8_bo_lower_err, vqe_8_bo_lower_err], 
                [fidelity_8_bo_upper_err, maxcut_8_bo_upper_err, vqe_8_bo_upper_err]]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    plt.rcParams.update({'font.size': 15})

    # Add some text for labels, title and custom x-axis tick labels, etc.
    def autolabel(rects_list, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rects in rects_list:
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_axes([0.15, 0.05, 0.8, 0.9])
    rects1 = ax.bar(x - width, rs_8_avg, width, label='Random Search')
    rects2 = ax.bar(x, rl_8_avg, width, label='REINFORCE')
    rects3 = ax.bar(x + width, bo_8_avg, width, label='Bayersian Optimization')

    rects_8_qubits = [rects1, rects2, rects3]

    ax.set_ylabel('Candidate Quantity')
    ax.set_title('8-qubit experiments with 1000 searches')
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 1001, 100))
    ax.set_ylim(0, 1010)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=14)

    autolabel(rects_8_qubits, ax)
    plt.savefig(os.path.join(args.save_path, "8-qubits_experiments_candidates.png"), dpi=400)
    plt.close()


    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_axes([0.15, 0.05, 0.8, 0.9])

    rects1_min = ax.bar(x - width, rs_8_min, width, label='Random Search')
    rects2_min = ax.bar(x, rl_8_min, width, label='REINFORCE')
    rects3_min = ax.bar(x + width, bo_8_min, width, label='Bayersian Optimization')

    rects_8_qubits_min = [rects1_min, rects2_min, rects3_min]

    ax.set_ylabel('Minimal Candidate Quantity')
    ax.set_title('8-qubit experiments with 1000 searches')
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 1001, 100))
    ax.set_ylim(0, 1010)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=14)

    autolabel(rects_8_qubits_min, ax)

    plt.savefig(os.path.join(args.save_path, "8-qubits_experiments_candidates_min.png"), dpi=400)
    plt.close()


    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_axes([0.15, 0.05, 0.8, 0.9])

    rects1_max = ax.bar(x - width, rs_8_max, width, label='Random Search')
    rects2_max = ax.bar(x, rl_8_max, width, label='REINFORCE')
    rects3_max = ax.bar(x + width, bo_8_max, width, label='Bayersian Optimization')

    rects_8_qubits_max = [rects1_max, rects2_max, rects3_max]

    ax.set_ylabel('Maximal Candidate Quantity')
    ax.set_title('8-qubit experiments with 1000 searches')
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 1001, 100))
    ax.set_ylim(0, 1010)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=14)

    autolabel(rects_8_qubits_max, ax)

    plt.savefig(os.path.join(args.save_path, "8-qubits_experiments_candidates_max.png"), dpi=400)
    plt.close()