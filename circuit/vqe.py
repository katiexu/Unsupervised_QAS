import os
import sys
sys.path.insert(0, os.getcwd())
import time
import pennylane as qml
import circuit.var_config as vc
import matplotlib.pyplot as plt

from pennylane import numpy as np
from circuit.fidelity import update_circuit
from circuit.circuit_manager import CircuitManager, circuit_qnode


# optmize parameters in circuits for solving VQE       
def vqe(hamiltonian, circuit_list, max_iterations=50, conv_tol = 1e-06):
    # get the initial parameters in the circuit
    params = []
    for op in circuit_list:
        if op[0] in ['RX', 'RY', 'RZ', 'U3']:
            for param in op[2:]:
                params.append(param)
    params = np.array(params, requires_grad=True)

    # objective function of VQE
    def objective(params):
        # obtain new circuit list after updating the parameters
        new_circuit_list = update_circuit(circuit_list, params)
        obj = circuit_qnode(new_circuit_list, app=3, hamiltonian=hamiltonian)
        return obj

    # record energy and parameters every step
    energy = [circuit_qnode(circuit_list, app=3, hamiltonian=hamiltonian)]
    params_memory = [params]

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    T1 = time.time()
    for n in range(max_iterations):
        params, prev_energy = opt.step_and_cost(objective, params)
        energy.append(objective(params))
        params_memory.append(params)
        conv = np.abs(energy[-1] - prev_energy)

        if n % 5 == 0:
            print(f"Step = {n+1},  Energy = {energy[-1]:.8f} Ha")
        
        if conv <= conv_tol:
            print(f"It is converged in Step {n+1}, Energy = {energy[-1]:.8f} Ha")
            break
    T2 = time.time()
    time_cost = (T2 - T1)*1000 # get the optimization time(ms)

    return energy, time_cost, params_memory

# A simple example
if __name__ == '__main__':
    circuit_manager = CircuitManager(8, 1, 20, 5, vc.allowed_gates)
    circuits = circuit_manager.generate_circuits()
    symbols = ["H", "H"]
    coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
    print("Number of qubits = ", qubits)
    print("The Hamiltonian is ", H)
    #coeffs = [1]*16
    #obs = [ qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0),
    #        qml.PauliZ(1) @ qml.PauliZ(2), qml.PauliX(1),
    #        qml.PauliZ(2) @ qml.PauliZ(3), qml.PauliX(2),
    #        qml.PauliZ(3) @ qml.PauliZ(4), qml.PauliX(3),
    #        qml.PauliZ(4) @ qml.PauliZ(5), qml.PauliX(4),
    #        qml.PauliZ(5) @ qml.PauliZ(6), qml.PauliX(5),
    #        qml.PauliZ(6) @ qml.PauliZ(7), qml.PauliX(6),
    #        qml.PauliZ(7) @ qml.PauliZ(0), qml.PauliX(7),]
    #H = qml.Hamiltonian(coeffs, obs)
    #print("Number of qubits = ", 8)
    #print("The Hamiltonian is ", H)
    fig, ax = qml.draw_mpl(circuit_qnode)(circuits[0])
    plt.show()
    energy, time_cost, _ = vqe(H, circuits[0])
    print(f"Final Energy = {energy[-1]}, Time Cost = {time_cost} ms")