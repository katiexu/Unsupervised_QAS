import torch

if __name__ == '__main__':
    embedding = torch.load('pretrained\\dim-16\\maxcut-model-circuits_4_qubits_full_embedding.pt')
    feature_embedding = torch.load('pretrained\\dim-16\\maxcut-model-circuits_4_qubits_1.pt')
    for ind in range(len(embedding)):
        embedding[ind]['feature'] = feature_embedding[ind]['feature']
    torch.save(embedding, 'pretrained\\dim-16\\maxcut-model-circuits_4_qubits_full_embedding_1.pt')