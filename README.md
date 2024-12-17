# Unsupervised_QAS for QWAS with data uploading function

## Main tasks
### Part 1: Pretraining
- 1) _gen_QWAS_circ_dataset.py_:
     * Generate a certain amount (5w) of unique random VQCs using QWAS search spaces. 
- 2) _pretraining.py_:
     * Train the generated VQCs with GVAE to get the latent distribution 𝑍.
     * Save the frozen 𝑍 and the GVAE model parameters.
### Part 2: Compare the performance of GVAE generated circuits and the original QWAS ones (run.py)
translate_selected_circuits.py:
Translate the selected QWAS circuits into the format required for loading the GVAE model. 
QWAS_original_performance.py:
Train the selected QWAS circuits (for 30 epochs) and obtain test acc on MNIST datasets for comparison.
gen_circuits_with_noise.py:
Generated new circuits by adding noises to latent representations of the selected QWAS circuits.
test_performance.py:
Train the generated new circuits (for 30 epochs) and obtain test acc on MNIST dataset.

