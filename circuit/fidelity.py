import os
import sys
sys.path.insert(0, os.getcwd())
import time
import pennylane as qml
import circuit.var_config as vc

from pennylane import numpy as np
from circuit.circuit_manager import circuit_qnode, CircuitManager


# obtain new circuit list after updating the parameters
def update_circuit(circuit_list, params):
    new_circuit_list = []
    count = 0
    for op in circuit_list:
        if op[0] in ['RX', 'RY', 'RZ', 'U3']:
            new_op = [op[0], op[1]]
            for i in range(len(op[2:])):
                new_op.append(params[count])
                count += 1
            new_circuit_list.append(tuple(new_op))
        else:
            new_circuit_list.append(op)
    return new_circuit_list

# optmize parameters in circuits for getting optimal fidelity
def opt_fidelity(target, circuit_list, max_iterations=50, conv_tol=1e-06):
    # get the initial parameters in the circuit
    params = []
    for op in circuit_list:
        if op[0] in ['RX', 'RY', 'RZ', 'U3']:
            for param in op[2:]:
                params.append(param)
    params = np.array(params, requires_grad=True)

    # objective function of Fidelity Task
    def objective(params):
        obj = 0
        # obtain new circuit list after updating the parameters
        new_circuit_list = update_circuit(circuit_list, params)
        # negative objective for the Fidelity Task
        obj = -np.array(qml.math.fidelity(circuit_qnode(new_circuit_list), target), dtype=np.float64)
        return obj

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
            print("Objective after step {:5d}: {: .8f}".format(i+1, -obj[-1]))
        if conv <= conv_tol:
            print(f"It is converged in Step {i+1}, Objective = {-obj[-1] : .8f}")
            break
    T2 = time.time()
    time_cost = (T2 - T1)*1000 # get the optimization time(ms)

    return obj, time_cost, params_memory

# A simple example
if __name__ == '__main__':
    circuit_manager = CircuitManager(4, 2, 10, 5, vc.allowed_gates)
    circuits = circuit_manager.generate_circuits()
    target = circuit_qnode(circuits[1])
    fidelity, time_cost, _ = opt_fidelity(target, circuits[0])
    print(f"Optimized Fidelity = {-fidelity[-1]}, Time Cost = {time_cost} ms")