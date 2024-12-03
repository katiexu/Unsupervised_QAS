#!/usr/bin/env bash

for s in {1..50}
	do
	  python search_methods/reinforce_maxcut_4_qubits.py --seed $s
	done
