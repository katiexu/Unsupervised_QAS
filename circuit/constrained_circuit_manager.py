import os
import sys
sys.path.insert(0, os.getcwd())
import json
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import circuit.var_config as vc

from circuit_manager import CircuitManager, circuit_qnode


current_path = os.getcwd()


class ConstrainedCircuitManager(CircuitManager):
    def __init__(self, num_qubits, num_circuits, num_gates, max_depth, allowed_gates, qubit_graph: np.ndarray):
        super().__init__(num_qubits, num_circuits, num_gates, max_depth, allowed_gates)
        self.qubit_graph = qubit_graph
    
    # Circuit constrained generator function
    def generate_circuits(self):
        unique_circuits = []

        def random_circuit__constrained_generator():
            circuit_ops = []
            for i in range(self.num_gates):
                gate = np.random.choice(self.allowed_gates).tolist()
                one_gate_qubit = np.random.choice(self.num_qubits)
                if gate in ['CNOT']:
                    control_qubit = np.random.choice(self.num_qubits, 1).item()
                    target_qubit_choices = self.qubit_graph[control_qubit].copy()
                    while np.sum(target_qubit_choices == 1) <= 1:
                        control_qubit = np.random.choice(self.num_qubits)
                        target_qubit_choices = self.qubit_graph[control_qubit].copy()
                    target_qubit_choices[control_qubit] = 0
                    p = target_qubit_choices / np.sum(target_qubit_choices == 1)
                    target_qubit = np.random.choice(len(target_qubit_choices), 1, replace=False, p=p).item()
                    if target_qubit < control_qubit:
                        control_qubit, target_qubit = target_qubit, control_qubit
                    circuit_ops.append((gate, control_qubit, target_qubit))
                elif gate in ['SWAP', 'CZ']:
                    control_qubit = np.random.choice(self.num_qubits, 1).item()
                    target_qubit_choices = self.qubit_graph[control_qubit].copy()
                    while np.sum(target_qubit_choices == 1) <= 1:
                        control_qubit = np.random.choice(self.num_qubits)
                        target_qubit_choices = self.qubit_graph[control_qubit].copy()
                    target_qubit_choices[control_qubit] = 0
                    p = target_qubit_choices / np.sum(target_qubit_choices == 1)
                    target_qubit = np.random.choice(len(target_qubit_choices), 1, replace=False, p=p).item()
                    circuit_ops.append((gate, control_qubit, target_qubit))
                elif gate in ['RX', 'RY', 'RZ']:
                    angle = np.random.uniform(0, 2 * np.pi)
                    circuit_ops.append((gate, one_gate_qubit, angle))
                elif gate in ['U3']:
                    theta = np.random.uniform(0, 2 * np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    delta = np.random.uniform(0, 2 * np.pi)
                    circuit_ops.append((gate, one_gate_qubit, theta, phi, delta))
                else:
                    circuit_ops.append((gate, one_gate_qubit))
                if qml.specs(circuit_qnode)(circuit_ops)['depth'] > self.max_depth:
                    return
            return circuit_ops
                
        while len(unique_circuits) < self.num_circuits:
            circuit_ops = random_circuit__constrained_generator()
            if circuit_ops == None:
                continue
            if not set(circuit_ops).issubset(set(unique_circuits)):
                unique_circuits.append(tuple(circuit_ops))
                self.pbar.update(1)
        self.pbar.close()
        return unique_circuits
    
    @property
    def get_qubit_graph(self):
        return self.qubit_graph

# randomly generate a valid qubit connection graph
def random_qubit_graph(num_qubits):
    diag = [1]*num_qubits
    qubit_graph = np.diag(diag)
    for i in range(qubit_graph.shape[0]):
        for j in range(i+1, qubit_graph.shape[1]):
            qubit_graph[i, j] = np.random.choice(2, 1, p=[0.25, 0.75]).item()
            qubit_graph[j, i] = qubit_graph[i, j]
    return qubit_graph

# dump circuit features in json file
def data_dumper(circuit_manager: CircuitManager, qubit_graph: np.ndarray, f_name: str ='data.json'):
    """dump circuit DAG features."""
    circuit_dict =  dict()
    circuit_features = []
    file_path = os.path.join(current_path, f'circuit\\data\\{f_name}')
    for i in range(circuit_manager.get_num_circuits):
        op_list, gate_matrix, adj_matrix = circuit_manager.get_gate_and_adj_matrix(circuits[i])
        circuit_features.append({'op_list': op_list, 'gate_matrix': gate_matrix, 'adj_matrix': adj_matrix.tolist()})
    circuit_dict["qubit_graph"] = qubit_graph.tolist()
    circuit_dict["circuit_features"] = circuit_features
    with open(file_path, 'w', encoding='utf-8') as file:  
        json.dump(circuit_dict, file)
    
if __name__ == '__main__':
    qubit_graph = random_qubit_graph(vc.num_qubits)
    print("The randomly generated qubit connection graph", qubit_graph)
    constrained_circuit_manager = ConstrainedCircuitManager(vc.num_qubits, vc.num_circuits, vc.num_gates, vc.max_depth, vc.allowed_gates, qubit_graph)
    circuits = constrained_circuit_manager.generate_circuits()
    print("Number of unique circuits generated:", len(circuits))
    print("The first curcuit list: ", circuits[0])
    op_list, gate_matrix, adj_matrix = constrained_circuit_manager.get_gate_and_adj_matrix(circuits[0])
    print("The first curcuit info: ")
    print("op_list: ", op_list)
    print("gate_matrix", gate_matrix)
    print("adj_matrx: ", adj_matrix)
    fig, ax = qml.draw_mpl(circuit_qnode)(circuits[0])
    plt.show()
    data_dumper(constrained_circuit_manager, qubit_graph, f'data_{vc.num_qubits}_qubits_constrained.json')