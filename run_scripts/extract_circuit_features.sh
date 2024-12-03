#!/usr/bin/env bash
python search_methods/reinforce_fidelity_4_qubits.py --saved_fidelity False
python search_methods/reinforce_fidelity_8_qubits.py --saved_fidelity False
python search_methods/reinforce_vqe_4_qubits.py --saved_vqe False
python search_methods/reinforce_vqe_8_qubits.py --saved_vqe False
python search_methods/reinforce_maxcut_4_qubits.py --saved_maxcut False
python search_methods/reinforce_maxcut_8_qubits.py --saved_maxcut False

