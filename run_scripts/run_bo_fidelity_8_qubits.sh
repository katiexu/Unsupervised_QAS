#!/usr/bin/env bash
for s in {1..50}
	do
	  python search_methods/dngo_fidelity_8_qubits.py --seed $s
	done