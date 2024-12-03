import os
import sys
sys.path.insert(0, os.getcwd())
import time
import networkx as nx
import pennylane as qml
import matplotlib.pyplot as plt
import circuit.var_config as vc

from pennylane.qaoa import maxcut
from pennylane import numpy as np
from utils.utils import load_json
from circuit.fidelity import update_circuit
from circuit.circuit_manager import circuit_qnode, CircuitManager


# Transform sample bitstring into binary integer
def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)

# optmize parameters in circuits for solving MAXCUT
def vqe_maxcut(graph, circuit_list, max_iterations=50, conv_tol = 1e-06, sample = True):
    # get the hamiltonians of the graph
    graph = nx.Graph(graph)
    hamiltonian_maxcut, hamiltonian_mixer = maxcut(graph)

    # get the initial parameters in the circuit
    params = []
    for op in circuit_list:
        if op[0] in ['RX', 'RY', 'RZ', 'U3']:
            for param in op[2:]:
                params.append(param)
    params = np.array(params, requires_grad=True)

    # objective function of MAXCUT
    def objective(params):
        neg_obj = 0
        # obtain new circuit list after updating the parameters
        new_circuit_list = update_circuit(circuit_list, params)
        for edge in graph:
            # objective for the MaxCut problem
            neg_obj -= 0.5 * (1 - circuit_qnode(new_circuit_list, app=2, hamiltonian=hamiltonian_maxcut, edge=edge))
        return neg_obj

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # record energy and parameters every step
    obj = [objective(params)]
    params_memory = [params]

    # optimize parameters in objective
    T1 = time.time()
    for i in range(max_iterations):
        params, prev_obj = opt.step_and_cost(objective, params)
        obj.append(objective(params))
        params_memory.append(params)
        conv = np.abs(obj[-1] - prev_obj)

        if i % 5 == 0:
            print("Objective after step {: 5d}: {: .8f}".format(i+1, obj[-1]))
        if conv <= conv_tol:
            print(f"It is converged in Step {i+1}, Objective = {obj[-1] : .8f}")
            break
    T2 = time.time()
    time_cost = (T2 - T1)*1000 # get the optimization time(ms)

    # get the circuit after optimization
    new_circuit_list = update_circuit(circuit_list, params_memory[-1])

    # sample measured bitstrings 200 times
    bit_strings = []

    if sample:
        bit_strings = []
        n_samples = 1000
        for i in range(0, n_samples):
            bit_strings.append(bitstring_to_int(circuit_qnode(new_circuit_list, app=2, shots=1)))

        # print optimal parameters and most frequently sampled bitstring
        counts = np.bincount(np.array(bit_strings))
        most_freq_bit_string = np.argmax(counts)
        print("Most frequently sampled bit string is: {: 04b}".format(most_freq_bit_string))

        return obj, time_cost, params_memory, bit_strings
    else:
        return obj, time_cost, params_memory

# A simple example
if __name__ == '__main__':
    #graph = nx.Graph([(0,1), (0,6), (0,7), (1,2), (1,6), (2,4), (2,7), (3,7), (3,6), (4,5), (4,6), (5,1), (5,6), (5,7), (6,7)])
    graph = [(0, 1), (0, 3), (1, 2), (2, 3)]
    circuit_manager = CircuitManager(8, 2, 30, 10, vc.allowed_gates)
    circuits = circuit_manager.generate_circuits()
    #dataset_4 = load_json("circuit\\data\\data_4_qubits.json")
    #circuit = dataset_4[5000]['op_list']
    obj1, time_cost1, _, bitstrings1 = vqe_maxcut(graph, circuits[0])
    obj2, time_cost2, _, bitstrings2 = vqe_maxcut(graph, circuits[1])
    print(f"Final Energy = {obj1[-1]}, Time Cost = {time_cost1} ms")
    print(f"Final Energy = {obj2[-1]}, Time Cost = {time_cost2} ms")

    xticks = range(0, 16)
    xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
    bins = np.arange(0, 17) - 0.5

    fig, ax = plt.subplots(1, 2, figsize=(11, 6))

    plt.subplot(1, 2, 1)
    plt.title("4-qubit max-cut")
    plt.xlabel("steps")
    plt.ylabel("energy")
    plt.plot(range(len(obj1)), obj1)

    plt.subplot(1, 2, 2)
    plt.title("4-qubit max-cut")
    plt.xlabel("bitstrings")
    plt.ylabel("frequency")
    plt.xticks(xticks, xtick_labels, rotation="vertical")
    plt.hist(bitstrings1, bins=bins)
    plt.show()

    '''
    #### to show a certain circuit and its performance ####

    fig, ax = qml.draw_mpl(circuit_qnode, wire_order=range(4), show_all_wires=True)(circuit)
    plt.show()
    
    fig = plt.figure()
    plt.title("4-qubit max-cut")
    plt.xlabel("steps")
    plt.ylabel("energy")
    plt.plot(range(len(obj1)), obj1)
    plt.show()

    fig = plt.figure(figsize=[6, 6])
    plt.title("4-qubit max-cut")
    plt.xlabel("bitstrings")
    plt.ylabel("frequency")
    plt.xticks(xticks, xtick_labels, rotation="vertical")
    plt.hist(bitstrings1, bins=bins)
    plt.show()
    '''