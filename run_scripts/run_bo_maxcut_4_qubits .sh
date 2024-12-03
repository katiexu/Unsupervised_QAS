#!/usr/bin/env bash
for s in {4..50}
	do
	  python search_methods/dngo_maxcut_4_qubits.py --seed $s
	done
