#!/usr/bin/env python3

import numpy as np
from collections import defaultdict

datapath = "datasamples/cora"

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats), dtype=np.float32)
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(datapath + "/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open(datapath + "/cora.cites") as fp:
        for line in fp:
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    adj_lists = {k: np.array(list(v)) for k, v in adj_lists.items()}
    
    return num_nodes, feat_data, labels, len(label_map), adj_lists
