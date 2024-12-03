#!/usr/bin/env bash

for s in {1..50}
	do
	  python search_methods/random_search_vqe_8_qubits.py --seed $s
	done
