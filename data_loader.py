from os import path
import os

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
import yaml 
import networkx as nx


with open('config.yaml') as f:
    cfg = yaml.load(f)

ROOT_DIR = cfg['ROOT_DIR']

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir= ROOT_DIR, transform=None, mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.paths = []
        self.labels = []
        for path, subdirs, files in os.walk(root_dir):
            for name in files:
                path_split = path.split("\\")
                if path_split[1] == mode:
                    if not path in self.paths:
                        self.paths.append(path)
                        self.labels.append(path_split[2])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        final_graph = nx.empty_graph()
        for path, subdirs, files in os.walk(self.paths[idx]):
            G = nx.read_gpickle(os.path.join(path,files))
            final_graph = nx.full_join(final_graph, G)
        



        return from_networkx(final_graph)




