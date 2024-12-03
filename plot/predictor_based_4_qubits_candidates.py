import os
import sys
sys.path.insert(0, os.getcwd())
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# get the number of candidate circuits of 4-qubits application experiments with corresponding search methods
def get_4_qubits_predictor_num_candidates(app: str, num_sample: int, search_method: str, filter_threshold: float, dim: int = None):
    '''
        app: So far, it can be "fidelity", "maxcut", "vqe"
        num_sample: the number of sample circuits
        dim: latent dimension, which should be claimed for gnn_predictor and gsqas_predictor
        search_method: So far, it can be "gnn_predictor", "gsqas_predictor"
    '''
    if search_method in ["gnn_predictor", "gsqas_predictor"] and dim == None:
        raise ValueError("when using gsqas or bo, the dim must be claimed!")
    
    num_candidates = []
    filtered_embedding_num = []
    filtered_candidate_num = []
    for seed in range(1, 51):
        f_name = 'saved_logs\\{}\\{}\\dim{}\\run_{}_{}_{}-{}-circuits_4_qubits.json'.format(
            search_method, app, dim, seed, filter_threshold, app, search_method)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        data = json.load(f)
        if num_sample != data["num_sample"]:
            raise ValueError("Please check if using correct 'num_sample' matching the experiments!")
        num_candidates.append(data['num_candidates'])
        filtered_embedding_num.append(data['filtered_embedding_num'])
        filtered_candidate_num.append(data['filtered_candidate_num'])
        f.close()

    avg_num_candidates = sum(num_candidates) // 50
    max_num_candidates = max(num_candidates)
    min_num_candidates = min(num_candidates)

    avg_filtered_embedding_num = sum(filtered_embedding_num) // 50
    max_filtered_embedding_num = max(filtered_embedding_num)
    min_filtered_embedding_num = min(filtered_embedding_num)

    avg_filtered_candidate_num = sum(filtered_candidate_num) // 50
    max_filtered_candidate_num = max(filtered_candidate_num)
    min_filtered_candidate_num = min(filtered_candidate_num)

    num_candidates_triple = (avg_num_candidates, max_num_candidates, min_num_candidates)
    filtered_embedding_num_triple = (avg_filtered_embedding_num, max_filtered_embedding_num, min_filtered_embedding_num)
    filtered_candidate_num_triple = (avg_filtered_candidate_num, max_filtered_candidate_num, min_filtered_candidate_num)

    return num_candidates_triple, filtered_embedding_num_triple, filtered_candidate_num_triple

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="4-qubits_app_experiments_num_candidates")
    parser.add_argument('--dim', type=int, default=16, help='feature dimension')
    parser.add_argument("--num_sample", type=int, default=1000, help="The number of sample circuits")
    parser.add_argument('--save_path', type=str, default='saved_figs')

    args = parser.parse_args()

    (fidelity_4_gnn_avg, fidelity_4_gnn_max, fidelity_4_gnn_min), \
    (fidelity_4_gnn_avg_filtered_embedding_num, fidelity_4_gnn_max_filtered_embedding_num, fidelity_4_gnn_min_filtered_embedding_num), \
    (fidelity_4_gnn_avg_filtered_candidate_num, fidelity_4_gnn_max_filtered_candidate_num, fidelity_4_gnn_min_filtered_candidate_num) \
        = get_4_qubits_predictor_num_candidates("fidelity", args.num_sample, "gnn_predictor", 0.5, args.dim)
    (maxcut_4_gnn_avg, maxcut_4_gnn_max, maxcut_4_gnn_min), \
    (maxcut_4_gnn_avg_filtered_embedding_num, maxcut_4_gnn_max_filtered_embedding_num, maxcut_4_gnn_min_filtered_embedding_num), \
    (maxcut_4_gnn_avg_filtered_candidate_num, maxcut_4_gnn_max_filtered_candidate_num, maxcut_4_gnn_min_filtered_candidate_num) \
        = get_4_qubits_predictor_num_candidates("maxcut", args.num_sample, "gnn_predictor", 0.9, args.dim)
    (vqe_4_gnn_avg, vqe_4_gnn_max, vqe_4_gnn_min), \
    (vqe_4_gnn_avg_filtered_embedding_num, vqe_4_gnn_max_filtered_embedding_num, vqe_4_gnn_min_filtered_embedding_num), \
    (vqe_4_gnn_avg_filtered_candidate_num, vqe_4_gnn_max_filtered_candidate_num, vqe_4_gnn_min_filtered_candidate_num) \
        = get_4_qubits_predictor_num_candidates("vqe", args.num_sample, "gnn_predictor", 0.675, args.dim)

    fidelity_4_gnn_lower_err = fidelity_4_gnn_avg - fidelity_4_gnn_min
    fidelity_4_gnn_upper_err = fidelity_4_gnn_max - fidelity_4_gnn_avg
    maxcut_4_gnn_lower_err = maxcut_4_gnn_avg - maxcut_4_gnn_min
    maxcut_4_gnn_upper_err = maxcut_4_gnn_max - maxcut_4_gnn_avg
    vqe_4_gnn_lower_err = vqe_4_gnn_avg - vqe_4_gnn_min
    vqe_4_gnn_upper_err = vqe_4_gnn_max - vqe_4_gnn_avg

    (fidelity_4_gsqas_avg, fidelity_4_gsqas_max, fidelity_4_gsqas_min), \
    (fidelity_4_gsqas_avg_filtered_embedding_num, fidelity_4_gsqas_max_filtered_embedding_num, fidelity_4_gsqas_min_filtered_embedding_num), \
    (fidelity_4_gsqas_avg_filtered_candidate_num, fidelity_4_gsqas_max_filtered_candidate_num, fidelity_4_gsqas_min_filtered_candidate_num) \
        = get_4_qubits_predictor_num_candidates("fidelity", args.num_sample, "gsqas_predictor", 0.5, args.dim)
    (maxcut_4_gsqas_avg, maxcut_4_gsqas_max, maxcut_4_gsqas_min), \
    (maxcut_4_gsqas_avg_filtered_embedding_num, maxcut_4_gsqas_max_filtered_embedding_num, maxcut_4_gsqas_min_filtered_embedding_num), \
    (maxcut_4_gsqas_avg_filtered_candidate_num, maxcut_4_gsqas_max_filtered_candidate_num, maxcut_4_gsqas_min_filtered_candidate_num) \
        = get_4_qubits_predictor_num_candidates("maxcut", args.num_sample, "gsqas_predictor", 0.9, args.dim)
    (vqe_4_gsqas_avg, vqe_4_gsqas_max, vqe_4_gsqas_min), \
    (vqe_4_gsqas_avg_filtered_embedding_num, vqe_4_gsqas_max_filtered_embedding_num, vqe_4_gsqas_min_filtered_embedding_num), \
    (vqe_4_gsqas_avg_filtered_candidate_num, vqe_4_gsqas_max_filtered_candidate_num, vqe_4_gsqas_min_filtered_candidate_num) \
        = get_4_qubits_predictor_num_candidates("vqe", args.num_sample, "gsqas_predictor", 0.675, args.dim)

    fidelity_4_gsqas_lower_err = fidelity_4_gsqas_avg - fidelity_4_gsqas_min
    fidelity_4_gsqas_upper_err = fidelity_4_gsqas_max - fidelity_4_gsqas_avg
    maxcut_4_gsqas_lower_err = maxcut_4_gsqas_avg - maxcut_4_gsqas_min
    maxcut_4_gsqas_upper_err = maxcut_4_gsqas_max - maxcut_4_gsqas_avg
    vqe_4_gsqas_lower_err = vqe_4_gsqas_avg - vqe_4_gsqas_min
    vqe_4_gsqas_upper_err = vqe_4_gsqas_max - vqe_4_gsqas_avg

    gnn_4_avg = [fidelity_4_gnn_avg, maxcut_4_gnn_avg, vqe_4_gnn_avg]
    gsqas_4_avg = [fidelity_4_gsqas_avg, maxcut_4_gsqas_avg, vqe_4_gsqas_avg]

    gnn_4_min = [fidelity_4_gnn_min, maxcut_4_gnn_min, vqe_4_gnn_min]
    gsqas_4_min = [fidelity_4_gsqas_min, maxcut_4_gsqas_min, vqe_4_gsqas_min]

    gnn_4_max = [fidelity_4_gnn_max, maxcut_4_gnn_max, vqe_4_gnn_max]
    gsqas_4_max = [fidelity_4_gsqas_max, maxcut_4_gsqas_max, vqe_4_gsqas_max]

    gnn_4_err = [[fidelity_4_gnn_lower_err, maxcut_4_gnn_lower_err, vqe_4_gnn_lower_err], 
                [fidelity_4_gnn_upper_err, maxcut_4_gnn_upper_err, vqe_4_gnn_upper_err]]
    gsqas_4_err = [[fidelity_4_gsqas_lower_err, maxcut_4_gsqas_lower_err, vqe_4_gsqas_lower_err], 
                [fidelity_4_gsqas_upper_err, maxcut_4_gsqas_upper_err, vqe_4_gsqas_upper_err]]
    
    print("gnn_predictor based search candidate information:")
    print(f"average fidelity filtered embedding quantity: {fidelity_4_gnn_avg_filtered_embedding_num}")
    print(f"maximal fidelity filtered embedding quantity: {fidelity_4_gnn_max_filtered_embedding_num}")
    print(f"minimal fidelity filtered embedding quantity: {fidelity_4_gnn_min_filtered_embedding_num}")
    print(f"average maxcut filtered embedding quantity: {maxcut_4_gnn_avg_filtered_embedding_num}")
    print(f"maximal maxcut filtered embedding quantity: {maxcut_4_gnn_max_filtered_embedding_num}")
    print(f"minimal maxcut filtered embedding quantity: {maxcut_4_gnn_min_filtered_embedding_num}")
    print(f"average vqe filtered embedding quantity: {vqe_4_gnn_avg_filtered_embedding_num}")
    print(f"maximal vqe filtered embedding quantity: {vqe_4_gnn_max_filtered_embedding_num}")
    print(f"minimal vqe filtered embedding quantity: {vqe_4_gnn_min_filtered_embedding_num}")
    print(f"average fidelity filtered candidate quantity: {fidelity_4_gnn_avg_filtered_candidate_num}")
    print(f"maximal fidelity filtered candidate quantity: {fidelity_4_gnn_max_filtered_candidate_num}")
    print(f"minimal fidelity filtered candidate quantity: {fidelity_4_gnn_min_filtered_candidate_num}")
    print(f"average maxcut filtered candidate quantity: {maxcut_4_gnn_avg_filtered_candidate_num}")
    print(f"maximal maxcut filtered candidate quantity: {maxcut_4_gnn_max_filtered_candidate_num}")
    print(f"minimal maxcut filtered candidate quantity: {maxcut_4_gnn_min_filtered_candidate_num}")
    print(f"average vqe filtered candidate quantity: {vqe_4_gnn_avg_filtered_candidate_num}")
    print(f"maximal vqe filtered candidate quantity: {vqe_4_gnn_max_filtered_candidate_num}")
    print(f"minimal vqe filtered candidate quantity: {vqe_4_gnn_min_filtered_candidate_num}")
    print(f"average fidelity searched candidate quantity: {fidelity_4_gnn_avg}")
    print(f"maximal fidelity searched candidate quantity: {fidelity_4_gnn_max}")
    print(f"minimal fidelity searched candidate quantity: {fidelity_4_gnn_min}")
    print(f"average maxcut searched candidate quantity: {maxcut_4_gnn_avg}")
    print(f"maximal maxcut searched candidate quantity: {maxcut_4_gnn_max}")
    print(f"minimal maxcut searched candidate quantity: {maxcut_4_gnn_min}")
    print(f"average vqe searched candidate quantity: {vqe_4_gnn_avg}")
    print(f"maximal vqe searched candidate quantity: {vqe_4_gnn_max}")
    print(f"minimal vqe searched candidate quantity: {vqe_4_gnn_min}")

    print("###################")

    print("gsqas_predictor based search candidate information:")
    print(f"average fidelity filtered embedding quantity: {fidelity_4_gsqas_avg_filtered_embedding_num}")
    print(f"maximal fidelity filtered embedding quantity: {fidelity_4_gsqas_max_filtered_embedding_num}")
    print(f"minimal fidelity filtered embedding quantity: {fidelity_4_gsqas_min_filtered_embedding_num}")
    print(f"average maxcut filtered embedding quantity: {maxcut_4_gsqas_avg_filtered_embedding_num}")
    print(f"maximal maxcut filtered embedding quantity: {maxcut_4_gsqas_max_filtered_embedding_num}")
    print(f"minimal maxcut filtered embedding quantity: {maxcut_4_gsqas_min_filtered_embedding_num}")
    print(f"average vqe filtered embedding quantity: {vqe_4_gsqas_avg_filtered_embedding_num}")
    print(f"maximal vqe filtered embedding quantity: {vqe_4_gsqas_max_filtered_embedding_num}")
    print(f"minimal vqe filtered embedding quantity: {vqe_4_gsqas_min_filtered_embedding_num}")
    print(f"average fidelity filtered candidate quantity: {fidelity_4_gsqas_avg_filtered_candidate_num}")
    print(f"maximal fidelity filtered candidate quantity: {fidelity_4_gsqas_max_filtered_candidate_num}")
    print(f"minimal fidelity filtered candidate quantity: {fidelity_4_gsqas_min_filtered_candidate_num}")
    print(f"average maxcut filtered candidate quantity: {maxcut_4_gsqas_avg_filtered_candidate_num}")
    print(f"maximal maxcut filtered candidate quantity: {maxcut_4_gsqas_max_filtered_candidate_num}")
    print(f"minimal maxcut filtered candidate quantity: {maxcut_4_gsqas_min_filtered_candidate_num}")
    print(f"average vqe filtered candidate quantity: {vqe_4_gsqas_avg_filtered_candidate_num}")
    print(f"maximal vqe filtered candidate quantity: {vqe_4_gsqas_max_filtered_candidate_num}")
    print(f"minimal vqe filtered candidate quantity: {vqe_4_gsqas_min_filtered_candidate_num}")
    print(f"average fidelity searched candidate quantity: {fidelity_4_gsqas_avg}")
    print(f"maximal fidelity searched candidate quantity: {fidelity_4_gsqas_max}")
    print(f"minimal fidelity searched candidate quantity: {fidelity_4_gsqas_min}")
    print(f"average maxcut searched candidate quantity: {maxcut_4_gsqas_avg}")
    print(f"maximal maxcut searched candidate quantity: {maxcut_4_gsqas_max}")
    print(f"minimal maxcut searched candidate quantity: {maxcut_4_gsqas_min}")
    print(f"average vqe searched candidate quantity: {vqe_4_gsqas_avg}")
    print(f"maximal vqe searched candidate quantity: {vqe_4_gsqas_max}")
    print(f"minimal vqe searched candidate quantity: {vqe_4_gsqas_min}")