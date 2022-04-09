from cgitb import reset
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from dgl.nn import ChebConv


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.Theta_r_x = ChebConv(input_size, hidden_size, 2)
        self.Theta_r_h = ChebConv(hidden_size, hidden_size, 2)

        self.Theta_u_x = ChebConv(input_size, hidden_size, 2)
        self.Theta_u_h = ChebConv(hidden_size, hidden_size, 2)

        self.Theta_c_x = ChebConv(input_size, hidden_size, 2)
        self.Theta_c_h = ChebConv(hidden_size, hidden_size, 2)

        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(0, std)

    def forward(self, g, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Output:
        #       hy: of shape (batch_size, hidden_size)

        if hx is None:
            hx = Variable(torch.zeros_like(input))

        reset_gate = torch.sigmoid(
            self.Theta_r_x(g, input) + self.Theta_r_h(g, hx)
            )
        update_gate = torch.sigmoid(
            self.Theta_u_x(g, input) + self.Theta_u_h(g, hx)
            )
        new_gate = torch.sigmoid(
            self.Theta_c_x(g, input) + self.Theta_c_h(g, hx) * reset_gate
            )

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy