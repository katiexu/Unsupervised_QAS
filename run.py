import subprocess

input_circuits = [
    [[[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1, 1, 1, 1], [3, 1, 1, 1, 1, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1, 1, 1, 1]], [[1, 2, 2, 2, 2], [2, 3, 3, 3, 3], [3, 4, 4, 4, 4], [4, 1, 1, 1, 1]]],
    [[[1, 1, 1, 1, 1, 0, 1, 1, 0], [4, 0, 1, 0, 1, 1, 0, 0, 1], [3, 1, 1, 0, 1, 0, 1, 1, 0], [2, 1, 1, 1, 1, 1, 1, 1, 1]], [[3, 3, 4, 3, 2], [2, 3, 3, 3, 1], [1, 2, 2, 2, 2], [4, 1, 1, 1, 1]]],
    [[[1, 0, 1, 1, 1, 0, 1, 0, 1], [4, 0, 1, 0, 0, 1, 0, 0, 1], [3, 0, 1, 1, 1, 0, 1, 1, 1], [2, 1, 1, 1, 1, 1, 1, 1, 1]], [[3, 3, 2, 3, 2], [1, 2, 2, 2, 1], [4, 3, 3, 3, 2], [2, 3, 3, 3, 1]]],
    [[[1, 0, 1, 0, 1, 0, 0, 0, 1], [2, 1, 1, 0, 1, 1, 0, 1, 1], [4, 0, 1, 0, 0, 1, 0, 0, 1], [3, 0, 1, 1, 1, 0, 1, 1, 1]], [[2, 4, 1, 3, 4], [3, 3, 2, 1, 2], [1, 1, 2, 2, 1], [4, 3, 3, 3, 2]]],
    [[[1, 0, 0, 1, 0, 0, 0, 1, 1], [4, 0, 1, 0, 1, 1, 0, 0, 1], [2, 1, 0, 0, 1, 1, 0, 1, 1], [3, 0, 1, 1, 1, 0, 1, 1, 1]], [[1, 3, 4, 3, 4], [3, 2, 2, 2, 3], [2, 4, 1, 3, 4], [4, 3, 3, 3, 2]]]
]

# Define the list of alpha values for noise
alpha_values = [0.05, 0.5, 1]

for input_data in input_circuits:

    input_str = str(input_data)
    # 1) Run this file to translate QWAS design to desired Pennylane format
    subprocess.run(['python', 'translate_selected_circuits.py', input_str])

    # 2) Run this file to evaluate test acc of the original QWAS circuit (without noise) with random initial weights
    subprocess.run(['python', 'QWAS_original_performance.py'])

    for alpha in alpha_values:
        # 3) Run this file to generate 10 new circuits by adding noises
        subprocess.run(['python', 'gen_circuits_with_noise.py', '--alpha', str(alpha)])

        # 4) Run this file to evaluate test acc of the newly generated 10 circuits
        subprocess.run(['python', 'test_performance.py', '--alpha', str(alpha)])