#!/usr/bin/env python3

import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

datapath = "datasamples/ppi"

def load_ppi():
    num_nodes = 14755
    num_feats = 50

    feat_data = np.load(datapath + "/toy-ppi-feats.npy")
    feat_data = feat_data.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(feat_data)
    feat_data = scaler.transform(feat_data)

    adj_lists = defaultdict(set)
    with open(datapath + "/toy-ppi-walks.txt") as fp:
        for line in fp:
            info = line.strip().split()
            item1 = int(info[0])
            item2 = int(info[1])
            adj_lists[item1].add(item2)
            adj_lists[item2].add(item1)

    adj_lists = {k: np.array(list(v)) for k, v in adj_lists.items()}
 
    return num_nodes, feat_data, adj_lists
