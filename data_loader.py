import os

import numpy as np

import torch
from dgl import batch, from_networkx

from torch.utils.data import Dataset
import networkx as nx

from config import ROOT_DIR

def collate_fn(sample) :
    # concatenate graph, features and labels w.r.t batch size
    graphs, features, labels = map(list, zip(*sample))
    graph = batch(graphs)
    features = torch.from_numpy(np.concatenate(features))
    labels = torch.from_numpy(np.concatenate(np.array(labels).reshape(-1,1)))
    return graph, features, labels



class EEGDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir= ROOT_DIR, transform=None, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.paths = []
        for path, subdirs, files in os.walk(os.path.join(root_dir, mode)):
            for name in files:
                self.paths.append(os.path.join(path,name))
        self.dic_label = {
            "ABSZ":0,
            "CPSZ":1,
            "FNSZ":2,
            "GNSZ":3,
            "MYSZ":4,
            "SPSZ":5,
            "TCSZ":6,
            "TNSZ":7,
        }

        np.random.shuffle(self.paths)
        self.paths = self.paths[:]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        label, G = nx.read_gpickle(self.paths[idx])
        features = []
        for u in G.nodes('Signal'):
            features.append(u[1])
        G = from_networkx(G)
        features = np.array(features)
        label = self.dic_label[label]
        return G, features, label




