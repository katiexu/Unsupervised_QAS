import os
import sys
import json
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
import matplotlib.pyplot as plt

# plot 4-qubits relevant reward data of vqe experiments
def plot_vqe_rewards(dim, num_sample):
    # read REINFORCE data
    rl_all_avg_reward_per_100 = []
    rl_all_regret_energy = []
    rl_final_avg_reward = []
    for seed in range(1, 51):
        f_name = 'saved_logs\\rl\\vqe\\dim{}\\run_{}_vqe-model-circuits_4_qubits.json'.format(dim, seed)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        rl_data = json.load(f)
        if num_sample != rl_data["num_sample"]:
            raise ValueError("Please check if using correct 'num_sample' matching the experiments!")
        rl_all_regret_energy.append(rl_data['regret_energy'])
        rl_all_avg_reward_per_100.append(rl_data['avg_reward_per_100'])
        rl_final_avg_reward.append(rl_data['avg_reward_per_100'][-1])
        f.close()
    rl_all_regret_energy = np.array(rl_all_regret_energy)
    rl_all_avg_reward_per_100 = np.array(rl_all_avg_reward_per_100)
    rl_final_avg_reward = np.array(rl_final_avg_reward)
    
    # read Bayersian Optimization data
    bo_all_avg_reward_per_100 = []
    bo_all_regret_energy = []
    bo_final_avg_reward = []
    for seed in range(1, 51):
        f_name = 'saved_logs\\bo\\vqe\\dim{}\\run_{}_vqe-model-circuits_4_qubits.json'.format(dim, seed)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        bo_data = json.load(f)
        if num_sample != bo_data["num_sample"]:
            raise ValueError("Please check if using correct 'num_sample' matching the experiments!")
        bo_all_regret_energy.append(bo_data['regret_energy'])
        bo_all_avg_reward_per_100.append(bo_data['avg_reward_per_100'])
        bo_final_avg_reward.append(bo_data['avg_reward_per_100'][-1])
        f.close()
    bo_all_regret_energy = np.array(bo_all_regret_energy)
    bo_all_avg_reward_per_100 = np.array(bo_all_avg_reward_per_100)
    bo_final_avg_reward = np.array(bo_final_avg_reward)

    # read Random Search data
    rs_all_avg_reward_per_100 = []
    rs_all_regret_energy = []
    rs_final_avg_reward = []
    for seed in range(1, 51):
        f_name = 'saved_logs\\rs\\vqe\\run_{}_vqe-rs-circuits_4_qubits.json'.format(seed)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        rs_data = json.load(f)
        if num_sample != rs_data["num_sample"]:
            raise ValueError("Please check if using correct 'num_sample' matching the experiments!")
        rs_all_regret_energy.append(rs_data['regret_energy'])
        rs_all_avg_reward_per_100.append(rs_data['avg_reward_per_100'])
        rs_final_avg_reward.append(rs_data['avg_reward_per_100'][-1])
        f.close()
    rs_all_regret_energy = np.array(rs_all_regret_energy)
    rs_all_avg_reward_per_100 = np.array(rs_all_avg_reward_per_100)
    rs_final_avg_reward = np.array(rs_final_avg_reward)

    # Process the data
    rl_avg_regret_energy = np.sum(rl_all_regret_energy, axis=0) / 50
    rl_avg_reward_per_100 = np.sum(rl_all_avg_reward_per_100, axis=0) / 50
    bo_avg_regret_energy = np.sum(bo_all_regret_energy, axis=0) / 50
    bo_avg_reward_per_100 = np.sum(bo_all_avg_reward_per_100, axis=0) / 50
    rs_avg_regret_energy = np.sum(rs_all_regret_energy, axis=0) / 50
    rs_avg_reward_per_100 = np.sum(rs_all_avg_reward_per_100, axis=0) / 50

    rl_avg_regret_energy = np.insert(rl_avg_regret_energy, 0, 1.136, axis=0)
    rl_avg_reward_per_100 = np.insert(rl_avg_reward_per_100, 0, 0., axis=0)
    bo_avg_regret_energy = np.insert(bo_avg_regret_energy, 0, 1.136, axis=0)
    bo_avg_reward_per_100 = np.insert(bo_avg_reward_per_100, 0, 0., axis=0)
    rs_avg_regret_energy = np.insert(rs_avg_regret_energy, 0, 1.136, axis=0)
    rs_avg_reward_per_100 = np.insert(rs_avg_reward_per_100, 0, 0., axis=0)

    rl_max_final_reward = np.max(rl_final_avg_reward)
    bo_max_final_reward = np.max(bo_final_avg_reward)
    rs_max_final_reward = np.max(rs_final_avg_reward)

    rl_min_final_reward = np.min(rl_final_avg_reward)
    bo_min_final_reward = np.min(bo_final_avg_reward)
    rs_min_final_reward = np.min(rs_final_avg_reward)

    sample_slice = np.arange(0, num_sample + 1, dtype=int)
    sample_slice_per_100 = np.arange(0, num_sample + 1, 100, dtype=int)

    plt.rcParams.update({'font.size': 14})

    # plot avg_reward_per_100
    fig = plt.figure(figsize=(8, 8))
    plt.xticks(sample_slice_per_100)
    plt.xlabel("search steps", fontsize=18)
    plt.ylabel("average reward per 100 searches", fontsize=18)
    plt.title(f"4-qubit quantum chemistry with {num_sample} searches")
    plt.plot(sample_slice_per_100, rs_avg_reward_per_100, label="Random Search", color="blue")
    plt.plot(sample_slice_per_100, rl_avg_reward_per_100, label="REINFORCE", color='red')
    plt.plot(sample_slice_per_100, bo_avg_reward_per_100, label="Bayersian Optimization", color='green')
    plt.axhline(rs_max_final_reward, xmin=0.85, xmax=0.95, label="Random Search", linestyle='--', linewidth="1", color="blue")
    plt.axhline(rs_min_final_reward, xmin=0.85, xmax=0.95, label="Random Search", linestyle='--', linewidth="1", color="blue")
    plt.axhline(rl_max_final_reward, xmin=0.85, xmax=0.95, label="REINFORCE", linestyle='--', linewidth="1", color="red")
    plt.axhline(rl_min_final_reward, xmin=0.85, xmax=0.95, label="REINFORCE", linestyle='--', linewidth="1", color="red")
    plt.axhline(bo_max_final_reward, xmin=0.85, xmax=0.95, label="Bayersian Optimization", linestyle='--', linewidth="1", color="green")
    plt.axhline(bo_min_final_reward, xmin=0.85, xmax=0.95, label="Bayersian Optimization", linestyle='--', linewidth="1", color="green")
    plt.legend(["Random Search", "REINFORCE", "Bayersian Optimization"], loc="lower right", fontsize=16)
    plt.savefig(os.path.join(args.save_path, "data", "4-qubits-vqe_avg_reward_per_100.png"), dpi=400)
    plt.close()

    # plot avg_reward_per_100 with full filling
    fig = plt.figure(figsize=(8, 8))
    plt.xticks(sample_slice_per_100)
    plt.xlabel("search steps", fontsize=18)
    plt.ylabel("average reward per 100 searches", fontsize=18)
    plt.title(f"4-qubit quantum chemistry with {num_sample} searches")
    plt.plot(sample_slice_per_100, rs_avg_reward_per_100, label="Random Search", color="blue")
    plt.plot(sample_slice_per_100, rl_avg_reward_per_100, label="REINFORCE", color='red')
    plt.plot(sample_slice_per_100, bo_avg_reward_per_100, label="Bayersian Optimization", color='green')
    for s in range(50):
        rs_temp_avg_reward_per_100 = np.insert(rs_all_avg_reward_per_100[s], 0, 0., axis=0)
        rl_temp_avg_reward_per_100 = np.insert(rl_all_avg_reward_per_100[s], 0, 0., axis=0)
        bo_temp_avg_reward_per_100 = np.insert(bo_all_avg_reward_per_100[s], 0, 0., axis=0)
        plt.fill_between(sample_slice_per_100, rs_temp_avg_reward_per_100,
                         rs_avg_reward_per_100, facecolor='blue', alpha=0.5)
        plt.fill_between(sample_slice_per_100, rl_temp_avg_reward_per_100,
                         rl_avg_reward_per_100, facecolor='red', alpha=0.5)
        plt.fill_between(sample_slice_per_100, bo_temp_avg_reward_per_100,
                         bo_avg_reward_per_100, facecolor='green', alpha=0.5)

    plt.legend(["Random Search", "REINFORCE", "Bayersian Optimization"], loc="lower right", fontsize=16)
    plt.savefig(os.path.join(args.save_path, "data", "4-qubits-vqe_avg_reward_per_100_with_full_filling.png"), dpi=400)
    plt.close()

    # plot avg_reward_per_100 with varianace filling
    fig = plt.figure(figsize=(8, 8))
    plt.xticks(sample_slice_per_100)
    plt.xlabel("search steps", fontsize=18)
    plt.ylabel("average reward per 100 searches", fontsize=18)
    plt.title(f"4-qubit quantum chemistry with {num_sample} searches")
    plt.plot(sample_slice_per_100, rs_avg_reward_per_100, label="Random Search", color="blue")
    plt.plot(sample_slice_per_100, rl_avg_reward_per_100, label="REINFORCE", color='red')
    plt.plot(sample_slice_per_100, bo_avg_reward_per_100, label="Bayersian Optimization", color='green')
    rs_var_reward_per_100 = np.std(rs_all_avg_reward_per_100, axis=0)
    rl_var_reward_per_100 = np.std(rl_all_avg_reward_per_100, axis=0)
    bo_var_reward_per_100 = np.std(bo_all_avg_reward_per_100, axis=0)
    rs_var_reward_per_100 = np.insert(rs_var_reward_per_100, 0, 0., axis=0)
    rl_var_reward_per_100 = np.insert(rl_var_reward_per_100, 0, 0., axis=0)
    bo_var_reward_per_100 = np.insert(bo_var_reward_per_100, 0, 0., axis=0)
    plt.fill_between(sample_slice_per_100, rs_avg_reward_per_100 + rs_var_reward_per_100,
                         rs_avg_reward_per_100 - rs_var_reward_per_100, facecolor='blue', alpha=0.25)
    plt.fill_between(sample_slice_per_100, rl_avg_reward_per_100 + rl_var_reward_per_100,
                         rl_avg_reward_per_100 - rl_var_reward_per_100 , facecolor='red', alpha=0.25)
    plt.fill_between(sample_slice_per_100, bo_avg_reward_per_100 + bo_var_reward_per_100,
                         bo_avg_reward_per_100 - bo_var_reward_per_100, facecolor='green', alpha=0.25)
    plt.legend(["Random Search", "REINFORCE", "Bayersian Optimization"], loc="lower right", fontsize=16)
    plt.savefig(os.path.join(args.save_path, "data", "4-qubits-vqe_avg_reward_per_100_with_var_filling.png"), dpi=400)
    plt.close()

    # plot regret energy from 0
    fig = plt.figure(figsize=(8, 8))
    plt.xticks(np.arange(0, num_sample + 1, 100, dtype=int))
    plt.xlabel("search steps", fontsize=18)
    plt.ylabel("regret energy", fontsize=18)
    plt.title(f"4-qubit quantum chemistry with {num_sample} searches")
    plt.plot(sample_slice, rs_avg_regret_energy, label="Random Search", color="blue")
    plt.plot(sample_slice, rl_avg_regret_energy, label="REINFORCE", color="red")
    plt.plot(sample_slice, bo_avg_regret_energy, label="Bayersian Optimization", color="green")
    plt.legend(["Random Search", "REINFORCE", "Bayersian Optimization"], fontsize=16)
    plt.savefig(os.path.join(args.save_path, "data", "4-qubits-vqe_regret_energy_from_0.png"), dpi=400)
    plt.close()

    # plot regret energy from 0 to 100
    fig = plt.figure(figsize=(8, 8))
    plt.xticks(np.arange(0, 101, 10, dtype=int))
    plt.yticks(np.arange(0, 1.2, 0.1, dtype=float))
    plt.xlabel("search steps", fontsize=18)
    plt.ylabel("regret energy", fontsize=18)
    plt.title(f"4-qubit quantum chemistry with {num_sample} searches")
    plt.plot(sample_slice[0:101], rs_avg_regret_energy[0:101], label="Random Search", color="blue")
    plt.plot(sample_slice[0:101], rl_avg_regret_energy[0:101], label="REINFORCE", color="red")
    plt.plot(sample_slice[0:101], bo_avg_regret_energy[0:101], label="Bayersian Optimization", color="green")
    plt.legend(["Random Search", "REINFORCE", "Bayersian Optimization"], fontsize=16)
    plt.savefig(os.path.join(args.save_path, "data", "4-qubits-vqe_regret_energy_from_0_to_100.png"), dpi=400)
    plt.close()

    # plot regret energy from 100
    fig = plt.figure(figsize=(8, 8))
    plt.xticks(np.arange(100, num_sample + 1, 100, dtype=int))
    plt.xlabel("search steps", fontsize=18)
    plt.ylabel("regret energy", fontsize=18)
    plt.title(f"4-qubit quantum chemistry with {num_sample} searches")
    plt.plot(sample_slice[100:], rs_avg_regret_energy[100:], label="Random Search", color="blue")
    plt.plot(sample_slice[100:], rl_avg_regret_energy[100:], label="REINFORCE", color="red")
    plt.plot(sample_slice[100:], bo_avg_regret_energy[100:], label="Bayersian Optimization", color="green")
    plt.legend(["Random Search", "REINFORCE", "Bayersian Optimization"], fontsize=16)
    plt.savefig(os.path.join(args.save_path, "data", "4-qubits-vqe_regret_energy_from_100.png"), dpi=400)
    plt.close()

    # plot regret energy from 100 with variance filling
    fig = plt.figure(figsize=(8, 8))
    plt.xticks(np.arange(100, num_sample + 1, 100, dtype=int))
    plt.xlabel("search steps", fontsize=18)
    plt.ylabel("regret energy", fontsize=18)
    plt.title(f"4-qubit quantum chemistry with {num_sample} searches")
    plt.plot(sample_slice[100:], rs_avg_regret_energy[100:], label="Random Search", color="blue")
    plt.plot(sample_slice[100:], rl_avg_regret_energy[100:], label="REINFORCE", color="red")
    plt.plot(sample_slice[100:], bo_avg_regret_energy[100:], label="Bayersian Optimization", color="green")
    rl_var_regret_fidelity = np.std(rl_all_regret_energy, axis=0)
    rs_var_regret_fidelity = np.std(rs_all_regret_energy, axis=0)
    bo_var_regret_fidelity = np.std(bo_all_regret_energy, axis=0)
    rl_var_regret_fidelity = np.insert(rl_var_regret_fidelity, 0, 0., axis=0)
    rs_var_regret_fidelity = np.insert(rs_var_regret_fidelity, 0, 0., axis=0)
    bo_var_regret_fidelity = np.insert(bo_var_regret_fidelity, 0, 0., axis=0)
    plt.fill_between(sample_slice[100:], (rs_avg_regret_energy + rs_var_regret_fidelity)[100:],
                         (rs_avg_regret_energy - rs_var_regret_fidelity)[100:], facecolor='blue', alpha=0.25)
    plt.fill_between(sample_slice[100:], (rl_avg_regret_energy + rl_var_regret_fidelity)[100:],
                         (rl_avg_regret_energy - rl_var_regret_fidelity)[100:], facecolor='red', alpha=0.25)
    plt.fill_between(sample_slice[100:], (bo_avg_regret_energy + bo_var_regret_fidelity)[100:],
                         (bo_avg_regret_energy - bo_var_regret_fidelity)[100:], facecolor='green', alpha=0.25)
    plt.legend(["Random Search", "REINFORCE", "Bayersian Optimization"], fontsize=16)
    plt.savefig(os.path.join(args.save_path, "data", "4-qubits-vqe_regret_energy_from_100_with_var_willing.png"), dpi=400)
    plt.close()

    # plot regret energy from 200
    fig = plt.figure(figsize=(8, 8))
    plt.xticks(np.arange(200, num_sample + 1, 100, dtype=int))
    plt.xlabel("search steps", fontsize=18)
    plt.ylabel("regret energy", fontsize=18)
    plt.title(f"4-qubit quantum chemistry with {num_sample} searches")
    plt.plot(sample_slice[200:], rs_avg_regret_energy[200:], label="Random Search", color="blue")
    plt.plot(sample_slice[200:], rl_avg_regret_energy[200:], label="REINFORCE", color="red")
    plt.plot(sample_slice[200:], bo_avg_regret_energy[200:], label="Bayersian Optimization", color="green")
    plt.legend(["Random Search", "REINFORCE", "Bayersian Optimization"], fontsize=16)
    plt.savefig(os.path.join(args.save_path, "data", "4-qubits-vqe_regret_energy_from_200.png"), dpi=400)
    plt.close()

    # plot regret energy from 300
    fig = plt.figure(figsize=(8, 8))
    plt.xticks(np.arange(300, num_sample + 1, 100, dtype=int))
    plt.xlabel("search steps", fontsize=18)
    plt.ylabel("regret energy", fontsize=18)
    plt.title(f"4-qubit quantum chemistry with {num_sample} searches")
    plt.plot(sample_slice[300:], rs_avg_regret_energy[300:], label="Random Search", color="blue")
    plt.plot(sample_slice[300:], rl_avg_regret_energy[300:], label="REINFORCE", color="red")
    plt.plot(sample_slice[300:], bo_avg_regret_energy[300:], label="Bayersian Optimization", color="green")
    plt.legend(["Random Search", "REINFORCE", "Bayersian Optimization"], fontsize=16)
    plt.savefig(os.path.join(args.save_path, "data", "4-qubits-vqe_regret_energy_from_300.png"), dpi=400)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-qubits-vqe_experiments_reward")
    parser.add_argument('--dim', type=int, default=16, help='feature dimension')
    parser.add_argument("--num_sample", type=int, default=1000, help="The number of sample circuits")
    parser.add_argument('--save_path', type=str, default='saved_figs\\vqe')

    args = parser.parse_args()
    plot_vqe_rewards(args.dim, args.num_sample)