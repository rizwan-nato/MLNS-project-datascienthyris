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
import torch.nn as nn
from dgl.nn import MaxPooling


class BasicGraphModel(nn.Module):

    def __init__(
        self,
        g, 
        n_layers, 
        input_size, 
        hidden_size=32, 
        output_size=16, 
        nonlinearity=F.relu,
        dropout_p=0.5
        ):

        super().__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_size, hidden_size, activation=nonlinearity))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_size, hidden_size, activation=nonlinearity))
        self.layers.append(GraphConv(hidden_size, output_size))

        self.dropout = nn.Dropout(p = dropout_p)
        self.maxpool = MaxPooling()
        self.FC = nn.Linear(output_size, 8)

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)
        outputs = F.relu(self.dropout(outputs))
        outputs = self.maxpool(self.g,outputs)
        outputs = self.FC(outputs)
        return F.softmax(outputs)

class AttGraphModel(nn.Module):

    def __init__(self, g, input_size, hidden_size, output_size, nonlinearity= F.relu, n_layers= 3, heads = 4, dropout_p = 0.):
        super().__init__()
        self.n_layers = n_layers
        self.g = g
        self.layers = nn.ModuleList()
        channels = hidden_size//heads

        for i in range(n_layers-1):
            if i ==0:
                self.layers.append(GATConv(input_size, channels, num_heads=heads, attn_drop=dropout_p, feat_drop=dropout_p, activation = nonlinearity))
            else:
                self.layers.append(GATConv(channels*heads, channels, num_heads=heads, attn_drop=dropout_p, feat_drop=dropout_p, activation = nonlinearity))
        self.layers.append(GATConv(channels*heads, output_size, heads))

        self.dropout = nn.Dropout(p = dropout_p)
        self.maxpool = MaxPooling()
        self.FC = nn.Linear(output_size*heads, 8)

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = torch.reshape(outputs,(outputs.shape[0],-1))
            outputs = layer(self.g, outputs)
        outputs = torch.reshape(outputs,(outputs.shape[0],-1))  
        outputs = F.relu(self.dropout(outputs))
        outputs = self.maxpool(self.g,outputs)
        outputs = self.FC(outputs)
        return F.softmax(outputs)
