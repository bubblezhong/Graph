#!/usr/bin/env python3

"""
This module computes minibatch (data and auxiliary matrices) for mean aggregator

requirement: neigh_dict is a BIDIRECTIONAL adjacency matrix in dict
"""

import numpy as np
import collections
from functools import reduce

# 无监督学习时：通过边的信息进行负采样
def build_batch_from_edges(edges, nodes, neigh_dict, sample_sizes, neg_size):
    """
    This batch method is used for unsupervised mode. First, it prepares
    auxiliary matrices for the combination of neighbor nodes (read from edges)
    and negative sample nodes. Second, it provides mappings to filter the
    results into three portions for use in the unsupervised loss function.
    
    feed:mini_batch_edges, nodes, adj_mat_dict, sample_sizes= [25, 10], neg_size=20
    
    :param array([(int, int)]) edges: edge with node ids
    :param array([int]) nodes: all node ids
    :param {node:[node]} neigh_dict: BIDIRECTIONAL adjacency matrix in dict
    :param [sample_size]: sample sizes for each layer, lens is the number of layers
    :param int neg_size: size of batchN
    :return namedtuple minibatch (3 more additional elements to supervised mode)
        "src_nodes": node ids to retrieve from raw feature and feed to the first layer
        "dstsrc2dsts": list of dstsrc2dst matrices from last to first layer
        "dstsrc2srcs": list of dstsrc2src matrices from last to first layer
        "dif_mats": list of dif_mat matrices from last to first layer
        "dst2batchA": select batchA nodes from all nodes at the last layer
        "dst2batchB": select batchB nodes from all nodes at the last layer
        "dst2batchN": filter batchN nodes from all nodes at the last layer

    Terms:
    - batchA: just a set of nodes
    - batchB: a set of nodes which are neighbors of batchA
    - batchN: a set of negative sample nodes far away from batchA/batchB
    Notes:
    - batchA and batchB have the same size, and they are row-to-row paired in
      training (u and v in Eq (1) in the GraphSage paper).
    - batchN is randomly selected. The entire set is far from any node in
      batchA/batchB. There is a small chance that a node in batchN is close
      to a node in batchA/batchB.
    """
    
    # edges 数组转置, batchA表示边的起点集合，batchB表示边的终点集合
    batchA, batchB = edges.transpose()
    
    # 找出所有的负采样的样本，在随机选取负采样样本
    # reduce(fun,seq): sequence连续使用function
    #  np.setdiff1d(arr1,arr2)：返回存在arr1中不存在于arr2中的元素(去重)。
    possible_negs = reduce ( np.setdiff1d
                           , ( nodes
                             , batchA
                              # 返回：batchA中的节点的邻居节点的数组(去重)
                             , _get_neighbors(batchA, neigh_dict)
                             , batchB
                            # 返回：batchB中的节点的邻居节点的数组(去重)
                             , _get_neighbors(batchB, neigh_dict)
                             )
                           )
    batchN = np.random.choice ( possible_negs
                              , min(neg_size, len(possible_negs))
                              , replace=False
                              )

    # np.unique sorts the return, required by the following np.searchsorted
    # 所有的节点的集合
    batch_all = np.unique(np.concatenate((batchA, batchB, batchN)))
    # order does matter, in the model, use tf.gather on this
    dst2batchA = np.searchsorted(batch_all, batchA)
    # order does matter, in the model, use tf.gather on this
    dst2batchB = np.searchsorted(batch_all, batchB)
    # order does not matter, in the model, use tf.boolean_mask on this
    # 测试一维数组的每个元素是否也存在于第二个数组中。
    # 返回一个与ar1长度相同的布尔数组，该数组为true，其中ar1的元素位于ar2中，否则为False。
    dst2batchN = np.in1d(batch_all, batchN)

    minibatch_plain = build_batch_from_nodes ( batch_all
                                             , neigh_dict
                                             , sample_sizes
                                             )

    MiniBatchFields = [ "src_nodes", "dstsrc2srcs", "dstsrc2dsts", "dif_mats"
                      , "dst2batchA", "dst2batchB", "dst2batchN" ]
    MiniBatch = collections.namedtuple ("MiniBatch", MiniBatchFields)

    return MiniBatch ( minibatch_plain.src_nodes
                     , minibatch_plain.dstsrc2srcs
                     , minibatch_plain.dstsrc2dsts
                     , minibatch_plain.dif_mats
                     , dst2batchA
                     , dst2batchB
                     , dst2batchN
                     )

def build_batch_from_nodes(nodes, neigh_dict, sample_sizes):
    """
    :param [int] nodes: node ids（size=batch_size）[2,5,78,9]
    :param {node:[node]} neigh_dict: BIDIRECTIONAL adjacency matrix in dict
    :param [sample_size]: sample sizes for each layer, lens is the number of layers  [5,5]
    :return namedtuple minibatch
        "src_nodes": node ids to retrieve from raw feature and feed to the first layer
        "dstsrc2srcs": list of dstsrc2src matrices from last to first layer
        "dstsrc2dsts": list of dstsrc2dst matrices from last to first layer
        "dif_mats": list of dif_mat matrices from last to first layer
    """
    
    dst_nodes = [nodes]
    dstsrc2dsts = []
    dstsrc2srcs = []
    dif_mats = []

    max_node_id = max(list(neigh_dict.keys()))
# 
    for sample_size in reversed(sample_sizes):
        ds, d2s, d2d, dm = _compute_diffusion_matrix ( dst_nodes[-1]
                                                     , neigh_dict
                                                     , sample_size
                                                     , max_node_id
                                                     )
        dst_nodes.append(ds)
        dstsrc2srcs.append(d2s)
        dstsrc2dsts.append(d2d)
        dif_mats.append(dm)

    src_nodes = dst_nodes.pop()
    
    # # 定义一个namedtuple类型MiniBatch，并包含MiniBatchFields中的属性。
    MiniBatchFields = ["src_nodes", "dstsrc2srcs", "dstsrc2dsts", "dif_mats"]
    MiniBatch = collections.namedtuple ("MiniBatch", MiniBatchFields)
    # # 创建一个MiniBatch对象
    return MiniBatch(src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats)

################################################################
#                       Private Functions                      #
################################################################

def _compute_diffusion_matrix(dst_nodes, neigh_dict, sample_size, max_node_id):

    # 随机从边节点中采样出sample_size个节点，若边个数小于sample_size，全部采样
    def sample(ns):
        # ns=[9,100,200,102] 某个节点的边节点数组
        return np.random.choice(ns, min(len(ns), sample_size), replace=False)

    # 生成len=max_node_id + 1的全零vector,将采样的节点的位置标识为 1.
    def vectorize(ns):
        v = np.zeros(max_node_id + 1, dtype=np.float32)
        v[ns] = 1
        return v

    # sample neighbors
    # 生成采样后的节点的邻接矩阵 
    #对每个训练节点的边节点采样，并且生成vector(将采样的节点的位置标识为 1.其余位置=0)
    adj_mat_full = np.stack([vectorize(sample(neigh_dict[n])) for n in dst_nodes])
   
    # np.any 测试沿给定轴（axis=0）的任何数组元素的求值是否为True。
    # 结果为(max_node_id + 1,)的bool数组，True表示该节点被采样过。
    nonzero_cols_mask = np.any(adj_mat_full.astype(np.bool), axis=0)

    # compute diffusion matrix
    # 根据 nonzero_cols_mask 缩减维度
    # 最后的结果shape=(len(dst_node),x),其中x:nonzero_cols_mask求和(表示一共多少个节点被使用)
    adj_mat = adj_mat_full[:, nonzero_cols_mask] # 
    # 每一行除以每一行的总和
    adj_mat_sum = np.sum(adj_mat, axis=1, keepdims=True)  
    dif_mat = adj_mat / adj_mat_sum

    # compute dstsrc mappings
    # src_nodes表示上述采样中被用到的节点的序号(维度缩减)
    src_nodes = np.arange(nonzero_cols_mask.size)[nonzero_cols_mask]
    
    # np.union1d automatic sorts the return, which is required for np.searchsorted
    # 找到两个数组的并集。dst_nodes：所有的训练节点的集合，所有的训练节点的邻居节点的集合src_nodes
    dstsrc = np.union1d(dst_nodes, src_nodes)
    
    # 在 dstsrc 中查找应在其中插入 src_nodes 中的元素以保持顺序的索引。返回长度与src_nodes一致的
    # 因为 dstsrc 包含src_nodes中的元素，所以返回结果可以理解为src_nodes中的元素在dstsrc中的索引。
    dstsrc2src = np.searchsorted(dstsrc, src_nodes)
    dstsrc2dst = np.searchsorted(dstsrc, dst_nodes)

    # 返回：
    # dstsrc==>所有的训练节点,所有的训练节点的邻居节点的数组
    # dstsrc2src ==>训练节点对应的下标
    # dif_mat：采样后的邻居节点的邻接矩阵压缩以后的结果
    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat

def _get_neighbors(nodes, neigh_dict):
    """
    return an array of neighbors of all nodes in the input
    """
    return np.unique(np.concatenate([neigh_dict[n] for n in nodes]))
