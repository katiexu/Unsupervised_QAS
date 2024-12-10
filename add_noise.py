import torch
from models.model import GVAE
from models.configs import configs
import circuit.var_config as vc
import numpy as np
from utils.utils import is_valid_circuit
import json


np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


checkpoint = torch.load('../models/pretrained/dim-16/model-circuits_4_qubits.json.pt')
print(checkpoint)

def transform_operations(max_idx):
    transform_dict =  {0:'START', 1:'U3', 2:'C(U3)', 3:'Identity', 4:'END'}
    ops = []
    for idx in max_idx:
        ops.append(transform_dict[idx.item()])
    return ops

model = GVAE((9, 32, 64, 128, 64, 32, 16), normalize=True, dropout=0.3, **configs[4]['GAE']).cuda()
model.load_state_dict(checkpoint['model_state'])
print(model)

z = torch.load('../models/z.pt')

circuits_with_noise = []
while len(circuits_with_noise) < 10:
    # Add Gaussian noise to latent z
    gaussian_noise = torch.randn_like(z)
    z_with_noise = z + gaussian_noise
    full_op, full_ad = model.decoder(z_with_noise.unsqueeze(0))
    full_op = full_op.squeeze(0).cpu()
    ad = full_ad.squeeze(0).cpu()
    # Restore ops
    op = full_op[:, 0:-(vc.num_qubits)]
    max_idx = torch.argmax(op, dim=-1)
    one_hot = torch.zeros_like(op)
    for i in range(one_hot.shape[0]):
        one_hot[i][max_idx[i]] = 1
    op_decode = transform_operations(max_idx)
    # Restore adj matrix
    ad_decode = (ad > 0.5).int().triu(1).numpy()
    ad_decode = np.ndarray.tolist(ad_decode)
    if is_valid_circuit(ad_decode, op_decode):
        op_results = []
        # Restore the qubit choices of ops
        qubit_choices = full_op[:, -(vc.num_qubits):]
        for i in range(qubit_choices.size(0)):
            if op_decode[i] == 'C(U3)':
                # Select the two largest values and sort indices by value
                values, indices = torch.topk(qubit_choices[i], 2)
                indices = indices[values.argsort(descending=True)]
                op_results.append((op_decode[i], indices.tolist()))
            elif op_decode[i] == 'U3':
                values, indices = torch.topk(qubit_choices[i], 1)
                op_results.append((op_decode[i], indices.tolist()))
            else:
                pass    # Skip 'START', 'END', and 'Identity' gates as they don't change the state

        circuits_with_noise.append(op_results)

    with open('circuits_with_noises.json', 'w') as file:
        json.dump(circuits_with_noise, file)

print('end')
