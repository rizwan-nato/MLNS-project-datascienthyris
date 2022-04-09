from os import path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from dgl import batch, from_networkx, unbatch
from dgl.data.ppi import LegacyPPIDataset
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from dgl.nn import MaxPooling
from torch.autograd import Variable
from model.rnncells import GRUCell

class BasicGraphModel(nn.Module):

    def __init__(
        self,
        g, 
        n_layers, 
        input_size, 
        hidden_size=32, 
        nonlinearity=F.relu,
        dropout_p=0.5
        ):

        super().__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_size, hidden_size, activation=nonlinearity))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_size, hidden_size, activation=nonlinearity))
        self.layers.append(GraphConv(hidden_size, hidden_size))

        self.dropout = nn.Dropout(p = dropout_p)
        self.maxpool = MaxPooling()
        self.FC = nn.Linear(hidden_size, 8)

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)
        outputs = F.relu(self.dropout(outputs))
        outputs = self.maxpool(self.g,outputs)
        outputs = self.FC(outputs)
        return F.softmax(outputs)

class AttGraphModel(nn.Module):

    def __init__(self, g, input_size, hidden_size, nonlinearity= F.relu, n_layers= 3, heads = 4, dropout_p = 0.):
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
        self.layers.append(GATConv(channels*heads, hidden_size, heads))

        self.dropout = nn.Dropout(p = dropout_p)
        self.maxpool = MaxPooling()
        self.FC = nn.Linear(hidden_size*heads, 8)

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




class DCGRU(nn.Module):
    def __init__(self, g, input_size, hidden_size, bias = 0,  nonlinearity= F.relu, n_layers= 3, dropout_p = 0.):
        super(DCGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bias = bias
        self.output_size = 8

        self.layers = nn.ModuleList()

        self.layers.append(GRUCell(self.input_size,
                                          self.hidden_size,
                                          self.bias))
        for l in range(1, self.n_layers):
            self.layers.append(GRUCell(self.hidden_size,
                                              self.hidden_size,
                                              self.bias))
        self.fc = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, g,input, hx=None):

        # Input of shape (batch_size, seqence length, input_size)
        #
        # Output of shape (batch_size, output_size)
        gs=unbatch(g)
        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.n_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.n_layers, input.size(0), self.hidden_size))
        else:
             h0 = hx
        outs = []

        hidden = list()
        for layer in range(self.n_layers):
            hidden.append(h0[layer, :, :])
        print(input.shape)
        for t in range(input.size(1)):
            for layer in range(self.n_layers):
                if layer == 0:
                    hidden_l = self.layers[layer](gs[t], input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.layers[layer](g[t], hidden[layer - 1],hidden[layer])
                hidden[layer] = hidden_l
            outs.append(hidden_l)
        out = outs[-1].squeeze()
        out = self.fc(out)

        return out