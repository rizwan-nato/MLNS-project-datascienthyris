from os import path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from dgl import batch, from_networkx
from dgl.data.ppi import LegacyPPIDataset
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader




class BasicGraphModel(nn.Module):

    def __init__(self, g, n_layers, input_size, hidden_size, output_size, nonlinearity):
        super().__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_size, hidden_size, activation=nonlinearity))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_size, hidden_size, activation=nonlinearity))
        self.layers.append(GraphConv(hidden_size, output_size))

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)

        return outputs




class AttGraphModel(nn.Module):

    def __init__(self, g, input_size, hidden_size, output_size, nonlinearity, n_attention=2, heads = 4, d_rate = 0.):
        super().__init__()
        self.n_attention = n_attention
        self.g = g
        self.layers = nn.ModuleList()
        channels = hidden_size//heads

        for i in range(n_attention-1):
            if i ==0:
                self.layers.append(GATConv(input_size, channels, num_heads=heads, attn_drop=d_rate, feat_drop=d_rate, activation = nonlinearity))
            else:
                self.layers.append(GATConv(channels*heads, channels, num_heads=heads, attn_drop=d_rate, feat_drop=d_rate, activation = nonlinearity))
        self.layers.append(GATConv(channels*heads, output_size, 6))


    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = torch.reshape(outputs,(outputs.shape[0],-1))
            outputs = layer(self.g, outputs)
            
        outputs = torch.mean(outputs, 1)
        return outputs

class AttGraphModel_skip(nn.Module):

    def __init__(self, g, input_size, hidden_size, output_size, nonlinearity, n_attention=2, heads = 4, d_rate = 0.):
        super().__init__()
        self.n_attention = n_attention
        self.g = g
        self.layers = nn.ModuleList()
        channels = hidden_size//heads

        for i in range(n_attention-1):
            if i ==0:
                self.layers.append(GATConv(input_size, channels, num_heads=heads, attn_drop=d_rate, feat_drop=d_rate, activation = nonlinearity))
            else:
                self.layers.append(GATConv(channels*heads, channels, num_heads=heads, attn_drop=d_rate, feat_drop=d_rate, activation = nonlinearity))
        self.layers.append(GATConv(channels*heads, output_size, 6))


    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            if i%3 == 1:
                skip = outputs
            outputs = torch.reshape(outputs,(outputs.shape[0],-1))
            outputs = layer(self.g, outputs)
            if i%3 == 2:
                outputs = outputs + skip
            
        outputs = torch.mean(outputs, 1)
        return outputs
