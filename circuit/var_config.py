### Configure gloabal variables ###

max_depth = 5
num_qubits = 4
num_gates = 10 ### TODO: Can it be not defined in advance? (different-size Tensor), maybe padding with I gate
num_circuits = 100000
allowed_gates = ['PauliX', 'PauliY', 'PauliZ', 'Hadamard', 'RX', 'RY', 'RZ', 'CNOT', 'CZ', 'U3', 'SWAP']