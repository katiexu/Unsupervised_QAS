import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gin.models.mlp import MLP
from layers import GraphConvolution
from utils.utils import preprocessing, normalize_adj


class GVAE(nn.Module):
    def __init__(self, dims, normalize, dropout, **kwargs):
        super(GVAE, self).__init__()
        self.encoder = VAEncoder(dims, normalize, dropout)
        self.decoder = Decoder(dims[-1], dims[0], dropout, **kwargs)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, ops, adj):
        mu, logvar = self.encoder(ops, adj)
        z = self.reparameterize(mu, logvar)
        ops_recon, adj_recon, adjusted_ops_recon, adjusted_adj_recon = self.decoder(z)
        return ops_recon, adj_recon, mu, logvar, adjusted_ops_recon, adjusted_adj_recon



class VAEncoder(nn.Module):
    def __init__(self, dims, normalize, dropout):
        super(VAEncoder, self).__init__()
        self.gcs = nn.ModuleList(self.get_gcs(dims, dropout))
        self.gc_mu = GraphConvolution(dims[-2], dims[-1], dropout)
        self.gc_logvar = GraphConvolution(dims[-2], dims[-1], dropout)
        self.normalize = normalize
        self.conv_ops = nn.Conv1d(dims[0], dims[0], kernel_size=5, stride=3, padding=0)
        self.conv_adj = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(3, 3), padding=0)

    def get_gcs(self, dims, dropout):
        gcs = []
        for k in range(len(dims)-1):
            gcs.append(GraphConvolution(dims[k],dims[k+1], dropout))
        return gcs

    def forward(self, ops, adj):
        if self.normalize:
            adj = normalize_adj(adj)

        # Adjust ops dimension to (32, 10, 11) and adj dimension to (32, 10, 10)
        adjusted_ops = self.conv_ops(ops.permute(0, 2, 1)).permute(0, 2, 1)
        adjusted_adj = self.conv_adj(adj.unsqueeze(1)).squeeze(1)

        x = adjusted_ops
        for gc in self.gcs[:-1]:
            x = gc(x, adjusted_adj)
        mu = self.gc_mu(x, adjusted_adj)
        logvar = self.gc_logvar(x, adjusted_adj)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, embedding_dim, input_dim, dropout, activation_adj=torch.sigmoid, activation_ops=torch.sigmoid, adj_hidden_dim=None, ops_hidden_dim=None):
        super(Decoder, self).__init__()
        if adj_hidden_dim == None:
            self.adj_hidden_dim = embedding_dim
        if ops_hidden_dim == None:
            self.ops_hidden_dim = embedding_dim
        self.activation_adj = activation_adj
        self.activation_ops = activation_ops
        self.weight = torch.nn.Linear(embedding_dim, input_dim)
        self.dropout = dropout
        self.deconv_ops = nn.ConvTranspose1d(input_dim, input_dim, kernel_size=7, stride=3, padding=1, output_padding=2)
        self.deconv_adj = nn.ConvTranspose2d(1, 1, kernel_size=7, stride=3, padding=1, output_padding=2)

    def forward(self, embedding):
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        adjusted_ops = self.weight(embedding)
        adjusted_adj = torch.matmul(embedding, embedding.permute(0, 2, 1))

        # Use transposed conv to re-adjust ops dimension to (32, 34, 11) and adj dimension to (32, 34, 34)
        ops = self.deconv_ops(adjusted_ops.permute(0, 2, 1)).permute(0, 2, 1)
        adj = self.deconv_adj(adjusted_adj.unsqueeze(1)).squeeze(1)

        ops_recon = self.activation_adj(ops)
        adj_recon = self.activation_adj(adj)
        adjusted_ops_recon = self.activation_adj(adjusted_ops)
        adjusted_adj_recon = self.activation_adj(adjusted_adj)

        return ops_recon, adj_recon, adjusted_ops_recon, adjusted_adj_recon


class VAEReconstructed_Loss(object):
    def __init__(self, w_ops=1.0, w_adj=1.0, loss_ops=None, loss_adj=None):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.loss_ops = loss_ops
        self.loss_adj = loss_adj

    def __call__(self, inputs, targets, mu, logvar):
        ops_recon, adj_recon = inputs[0], inputs[1]
        ops, adj = targets[0], targets[1]
        loss_ops = self.loss_ops(ops_recon, ops)
        loss_adj = self.loss_adj(adj_recon.double(), adj.double())
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj
        KLD = -0.5 / (ops.shape[0] * ops.shape[1]) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 2))
        return loss + KLD

