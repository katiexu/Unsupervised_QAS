import os
import sys

sys.path.insert(0, os.getcwd())
import json
import tqdm
import torch
import numpy as np
import pennylane as qml
import var_config as vc

from tqdm import tqdm
from pennylane import CircuitGraph
from pennylane import numpy as pnp
from torch.nn import functional as F
import random
import argparse
import matplotlib.pyplot as plt

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

current_path = os.getcwd()

dev = qml.device("default.qubit", wires=vc.num_qubits)


# app=1: Fidelity Task; app=2: MAXCUT; app=3: VQE
@qml.qnode(dev)
def circuit_qnode(circuit_list, app=1, hamiltonian=None, edge=None):
    for params in list(circuit_list):
        if params == 'START':
            continue
        elif params[0] == 'Identity':
            qml.Identity(wires=params[1])
        elif params[0] == 'PauliX':
            qml.PauliX(wires=params[1])
        elif params[0] == 'PauliY':
            qml.PauliY(wires=params[1])
        elif params[0] == 'PauliZ':
            qml.PauliZ(wires=params[1])
        elif params[0] == 'Hadamard':
            qml.Hadamard(wires=params[1])
        elif params[0] == 'RX':
            # param = pnp.array(params[2], requires_grad=True)
            param = pnp.array(params[2], requires_grad=False)
            qml.RX(param, wires=params[1])
        elif params[0] == 'RY':
            # param = pnp.array(params[2], requires_grad=True)
            param = pnp.array(params[2], requires_grad=False)
            qml.RY(param, wires=params[1])
        elif params[0] == 'RZ':
            # param = pnp.array(params[2], requires_grad=True)
            param = pnp.array(params[2], requires_grad=False)
            qml.RZ(param, wires=params[1])
        elif params[0] == 'CNOT':
            qml.CNOT(wires=[params[1], params[2]])
        elif params[0] == 'CZ':
            qml.CZ(wires=[params[1], params[2]])
        elif params[0] == 'U3':
            theta = pnp.array(params[2], requires_grad=True)
            phi = pnp.array(params[3], requires_grad=True)
            delta = pnp.array(params[4], requires_grad=True)
            qml.U3(theta, phi, delta, wires=params[1])
        elif params[0] == 'C(U3)':
            theta = pnp.array(params[2], requires_grad=True)
            phi = pnp.array(params[3], requires_grad=True)
            delta = pnp.array(params[4], requires_grad=True)
            qml.ctrl(qml.U3, control=params[1])(theta, phi, delta, wires=params[2])
        elif params[0] == 'SWAP':
            qml.SWAP(wires=[params[1], params[2]])
        elif params == 'END':
            break
        else:
            print(params)
            raise ValueError("There exists operations not in the allowed operation pool!")

    if app == 1:
        return qml.state()
    elif app == 2:
        if edge is None:
            return qml.sample()
        if hamiltonian != None:
            return qml.expval(hamiltonian)
        else:
            raise ValueError("Please pass a hamiltonian as an observation for QAOA_MAXCUT!")
    elif app == 3:
        if hamiltonian != None:
            return qml.expval(hamiltonian)
        else:
            raise ValueError("Please pass a hamiltonian as an observation for VQE!")
    else:
        print("Note: Currently, there are no correspoding appllications!")


def translator(self, selected_single, selected_enta):
    # Translate single-qubit gates into lists
    single_columns = {i: [] for i in range(len(selected_single[0]) - 1)}
    for col_index in range(1, len(selected_single[0]), 2):
        for row_index in range(len(selected_single)):
            if selected_single[row_index][col_index] == 0:
                single_columns[col_index - 1].append(('Identity', row_index))
            else:
                gate = random.choice(['RX', 'RY', 'RZ'])
                angle = np.random.uniform(0, 2 * np.pi)
                single_columns[col_index - 1].append((gate, row_index, angle))
    for col_index in range(2, len(selected_single[0]), 2):
        for row_index in range(len(selected_single)):
            if selected_single[row_index][col_index] == 0:
                single_columns[col_index - 1].append(('Identity', row_index))
            else:
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                delta = np.random.uniform(0, 2 * np.pi)
                single_columns[col_index - 1].append(('U3', row_index, theta, phi, delta))

    # Translate entangled gates into lists
    enta_columns = {i: [] for i in range(len(selected_enta[0]) - 1)}
    for col_index in range(1, len(selected_enta[0])):
        for row_index in range(len(selected_enta)):
            control = row_index
            target = selected_enta[row_index][col_index] - 1
            if control == target:
                enta_columns[col_index - 1].append(('Identity', target))
            else:
                if col_index - 1 in enta_columns:
                    theta = np.random.uniform(0, 2 * np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    delta = np.random.uniform(0, 2 * np.pi)
                    enta_columns[col_index - 1].append(('C(U3)', control, target, theta, phi, delta))

    # # Remove empty lists and re-index
    # non_empty_single_columns = {new_index: v for new_index, (k, v) in
    #                           enumerate(item for item in single_columns.items() if item[1])}
    # non_empty_enta_columns = {new_index: v for new_index, (k, v) in
    #                           enumerate(item for item in enta_columns.items() if item[1])}

    # Re-order gate lists and generate final design
    single_index = 0
    enta_index = 0
    circuit_ops = []
    for layer in range(self.num_layers):
        circuit_ops.extend(single_columns[single_index])
        circuit_ops.extend(single_columns[single_index + 1])
        circuit_ops.extend(enta_columns[enta_index])
        single_index += 2
        enta_index += 1

    return circuit_ops


class CircuitManager:

    # class constructor
    def __init__(self, num_qubits, num_circuits, num_layers, allowed_gates):
        self.num_qubits = num_qubits
        self.num_circuits = num_circuits
        self.num_layers = num_layers
        self.allowed_gates = allowed_gates
        self.pbar = tqdm(range(self.num_circuits), desc="generated_num_circuits")

    # encode allowed gates in one-hot encoding
    def encode_gate_type(self):
        gate_dict = {}
        ops = self.allowed_gates.copy()
        ops.insert(0, 'START')
        ops.append('END')
        ops_len = len(ops)
        ops_index = torch.tensor(range(ops_len))
        type_onehot = F.one_hot(ops_index, num_classes=ops_len)
        for i in range(ops_len):
            gate_dict[ops[i]] = type_onehot[i]
        return gate_dict

    # Circuit generator function
    def generate_circuits(self, single, enta):
        unique_circuits = []

        # def filter_start_with(search_space, start_value):
        #     return [lst for lst in search_space if lst[0] == start_value]

        def generate_QWAS_circuits():
            selected_single = single
            selected_enta = enta

            # for start_value in range(1, self.num_qubits + 1):
            #     candidates_single = filter_start_with(single, start_value)
            #     selected_single.append(random.sample(candidates_single, 1)[0])
            #
            #     candidates_enta = filter_start_with(enta, start_value)
            #     selected_enta.append(random.sample(candidates_enta, 1)[0])

            circuit_ops = translator(self, selected_single, selected_enta)

            return circuit_ops

        while len(unique_circuits) < self.num_circuits:
            circuit_ops = generate_QWAS_circuits()
            if circuit_ops == None:
                continue
            if not set(circuit_ops).issubset(set(unique_circuits)):
                unique_circuits.append(tuple(circuit_ops))
                self.pbar.update(1)

        return unique_circuits


    # transform a circuit into a circuit graph
    def get_circuit_graph(self, circuit_list):
        circuit_qnode(circuit_list)
        tape = circuit_qnode.qtape
        ops = tape.operations
        obs = tape.observables
        return CircuitGraph(ops, obs, tape.wires)

    # get the gate and adjacent matrix of a circuit
    def get_gate_and_adj_matrix(self, circuit_list):
        gate_matrix = []
        op_list = []
        cl = list(circuit_list).copy()
        if cl[0] != 'START':
            cl.insert(0, 'START')
        if cl[-1] != 'END':
            cl.append('END')
        cg = self.get_circuit_graph(circuit_list)
        gate_dict = self.encode_gate_type()
        gate_matrix.append(gate_dict['START'].tolist() + [1] * self.num_qubits)
        op_list.append('START')
        for op in cg.operations:
            op_list.append(op)
            op_qubits = [0] * self.num_qubits
            for i in op.wires:
                op_qubits[i] = 1
            op_vector = gate_dict[op.name].tolist() + op_qubits
            gate_matrix.append(op_vector)
        gate_matrix.append(gate_dict['END'].tolist() + [1] * self.num_qubits)
        op_list.append('END')

        op_len = len(op_list)
        adj_matrix = np.zeros((op_len, op_len), dtype=int)
        for op in cg.operations:
            ancestors = cg.ancestors_in_order([op])
            descendants = cg.descendants_in_order([op])
            if len(ancestors) == 0:
                adj_matrix[0][op_list.index(op)] = 1
            else:
                if op.name in ['CNOT', 'CZ', 'SWAP']:
                    count = 0
                    wires = set()
                    for ancestor in ancestors[::-1]:
                        wires.update(set(ancestor.wires) & set(op.wires))
                        if not len(wires) == count:
                            adj_matrix[op_list.index(ancestor)][op_list.index(op)] = 1
                            count += 1
                        if count == 2:
                            break
                    if count == 1:
                        adj_matrix[0][op_list.index(op)] = 1
                else:
                    direct_ancestor = ancestors[-1]
                    adj_matrix[op_list.index(direct_ancestor)][op_list.index(op)] = 1
            if op.name in ['CNOT', 'CZ', 'SWAP']:
                wires = set()
                for descendant in descendants:
                    wires.update(set(descendant.wires) & set(op.wires))
                    if isinstance(descendant, qml.measurements.StateMP):
                        adj_matrix[op_list.index(op)][op_len - 1] = 1
                    if len(wires) == 2:
                        break
            else:
                if isinstance(descendants[0], qml.measurements.StateMP):
                    adj_matrix[op_list.index(op)][op_len - 1] = 1

        return cl, gate_matrix, adj_matrix

    @property
    def get_num_qubits(self):
        return self.num_qubits

    @property
    def get_num_circuits(self):
        return self.num_circuits

    @property
    def get_num_gates(self):
        return self.num_gates

    @property
    def get_max_depth(self):
        return self.max_depth


# dump circuit features in json file
def data_dumper(circuit_manager: CircuitManager, f_name: str = 'data.json'):
    """dump circuit DAG features."""
    circuit_features = []
    # file_path = os.path.join(current_path, f'circuit\\data\\{f_name}')
    # file_path = os.path.join(current_path, f'data/{f_name}')
    file_path = os.path.join('data_selected_circuits.json')
    for i in range(circuit_manager.get_num_circuits):
        op_list, gate_matrix, adj_matrix = circuit_manager.get_gate_and_adj_matrix(circuits[i])
        circuit_features.append({'op_list': op_list, 'gate_matrix': gate_matrix, 'adj_matrix': adj_matrix.tolist()})
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(circuit_features, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run script with input parameter.')
    parser.add_argument('input', type=str, help='Input value')
    parser.add_argument('circuit_index', type=str, help='Circuit index value')
    args = parser.parse_args()

    input_circuit = eval(args.input)
    circuit_index = args.circuit_index
    print(f"Running python file with input = {input_circuit}")

    # input_circuit = [[[1, 1, 1, 1, 1, 0, 1, 1, 0], [4, 0, 1, 0, 1, 1, 0, 0, 1], [3, 1, 1, 0, 1, 0, 1, 1, 0], [2, 1, 1, 1, 1, 1, 1, 1, 1]], [[3, 3, 4, 3, 2], [2, 3, 3, 3, 1], [1, 2, 2, 2, 2], [4, 1, 1, 1, 1]]]
    # # input_circuit = [[[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1, 1, 1, 1], [3, 1, 1, 1, 1, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1, 1, 1, 1]], [[1, 2, 2, 2, 2], [2, 3, 3, 3, 3], [3, 4, 4, 4, 4], [4, 1, 1, 1, 1]]]
    # circuit_index = 0

    circuit_manager = CircuitManager(vc.num_qubits, 1, vc.num_layers, vc.allowed_gates)

    sorted_input_circuit = [sorted(sub_list, key=lambda x: x[0]) for sub_list in input_circuit]

    single = sorted_input_circuit[0]
    enta = sorted_input_circuit[1]

    circuits = circuit_manager.generate_circuits(single, enta)
    # print("Number of unique circuits generated:", len(circuits))
    # print("The first curcuit list: ", circuits[0])
    op_list, gate_matrix, adj_matrix = circuit_manager.get_gate_and_adj_matrix(circuits[0])
    print("The first curcuit info: ")
    print("op_list: ", op_list)
    print("gate_matrix\n", gate_matrix)
    print("adj_matrx: \n", adj_matrix)
    fig, ax = qml.draw_mpl(circuit_qnode)(circuits[0])
    plt.show()
    data_dumper(circuit_manager, f_name=f'data_selected_circuits.json')

    diagram_dir = f"circuit_diagrams/circuit_index_{circuit_index}"
    if not os.path.exists(diagram_dir):
        os.makedirs(diagram_dir)
    fig.savefig(f"circuit_diagrams/circuit_index_{circuit_index}/original.jpg")