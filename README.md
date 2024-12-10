# Updated code for QWAS

## Run code
1) Run **circuit/circuit_manager.py** to generate random circuits from QWAS search space.
2) Run **models/pretrainning.py** to do GVAE training and to save the frozen z representation and its corresponding model parameters.
3) Run **add_noise.py** to generate new circuits by adding Gaussian noises to the frozen z.
4) Run **test_performance.py** to evaluate the newly generated circuits.
