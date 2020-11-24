#!/usr/bin/env python3

import tensorflow as tf

init_fn = tf.keras.initializers.GlorotUniform

class GraphSageBase(tf.keras.Model):
    """
    GraphSage base model outputing embeddings of given nodes
    """

    def __init__(self, raw_features, internal_dim, num_layers, last_has_activ):

        assert num_layers > 0, 'illegal parameter "num_layers"'
        assert internal_dim > 0, 'illegal parameter "internal_dim"'

        super().__init__()
        # call主要作用：根据传入的 node 获得相应的特征
        self.input_layer = RawFeature(raw_features, name="raw_feature_layer")

        self.seq_layers = []
        # 采样层数 num_layers = 2
        for i in range (1, num_layers + 1):
            layer_name = "agg_lv" + str(i)
            # i=1,internal_dim=128
            # else internal_dim=1433
            input_dim = internal_dim if i > 1 else raw_features.shape[-1]
            has_activ = last_has_activ if i == num_layers else True
            # 
            aggregator_layer = MeanAggregator ( input_dim
                                              , internal_dim
                                              , name=layer_name
                                              , activ = has_activ
                                              )
            self.seq_layers.append(aggregator_layer)

    def call(self, minibatch):
        """
        :param [node] nodes: target nodes for embedding
        """
        # squeeze: 将原始input中所有维度为1的那些维都删掉的结果
        x = self.input_layer(tf.squeeze(minibatch.src_nodes))
        for aggregator_layer in self.seq_layers:
            x = aggregator_layer ( x
                                 , minibatch.dstsrc2srcs.pop()
                                 , minibatch.dstsrc2dsts.pop()
                                 , minibatch.dif_mats.pop()
                                 )
        return x

class GraphSageUnsupervised(GraphSageBase):
    # # raw_features, 128, 2, 1.0
    def __init__(self, raw_features, internal_dim, num_layers, neg_weight):
        super().__init__(raw_features, internal_dim, num_layers, False)
        self.neg_weight = neg_weight

    def call(self, minibatch):
        embeddingABN = tf.math.l2_normalize(super().call(minibatch), 1)
        self.add_loss (
                compute_uloss ( tf.gather(embeddingABN, minibatch.dst2batchA)
                              , tf.gather(embeddingABN, minibatch.dst2batchB)
                              , tf.boolean_mask(embeddingABN, minibatch.dst2batchN)
                              , self.neg_weight
                              )
                )
        return embeddingABN
    
class GraphSageSupervised(GraphSageBase):
    def __init__(self, raw_features, internal_dim, num_layers, num_classes):
        super().__init__(raw_features, internal_dim, num_layers, True)
        
        # 分类器
        self.classifier = tf.keras.layers.Dense ( num_classes
                                                , activation = tf.nn.softmax
                                                , use_bias = False
                                                , kernel_initializer = init_fn
                                                , name = "classifier"
                                                )

    def call(self, minibatch):
        """
        :param [node] nodes: target nodes for embedding
        """
        return self.classifier( super().call(minibatch) )


################################################################
#                         Custom Layers                        #
################################################################

class RawFeature(tf.keras.layers.Layer):
    def __init__(self, features, **kwargs):
        """
        :param ndarray((#(node), #(feature))) features: a matrix, each row is feature for a node
        """
        super().__init__(trainable=False, **kwargs)
        self.features = tf.constant(features)
        
    def call(self, nodes):
        """
        :param [int] nodes: node ids
        """
        return tf.gather(self.features, nodes)

class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, src_dim, dst_dim, activ=True, **kwargs):
        """
        :param int src_dim: input dimension
        :param int dst_dim: output dimension
        """
        super().__init__(**kwargs)
        self.activ_fn = tf.nn.relu if activ else tf.identity
        self.w = self.add_weight( name = kwargs["name"] + "_weight"
                                , shape = (src_dim*2, dst_dim)
                                , dtype = tf.float32
                                , initializer = init_fn
                                , trainable = True
                                )
    
    def call(self, dstsrc_features, dstsrc2src, dstsrc2dst, dif_mat):
        """
        :param tensor dstsrc_features: the embedding from the previous layer
        :param tensor dstsrc2dst: 1d index mapping (prepraed by minibatch generator)
        :param tensor dstsrc2src: 1d index mapping (prepraed by minibatch generator)
        :param tensor dif_mat: 2d diffusion matrix (prepraed by minibatch generator)
        """
        dst_features = tf.gather(dstsrc_features, dstsrc2dst)
        src_features = tf.gather(dstsrc_features, dstsrc2src)
        aggregated_features = tf.matmul(dif_mat, src_features)
        concatenated_features = tf.concat([aggregated_features, dst_features], 1)
        x = tf.matmul(concatenated_features, self.w)
        return self.activ_fn(x)

################################################################
#               Custom Loss Function (Unsupervised)            #
################################################################

@tf.function 
def compute_uloss(embeddingA, embeddingB, embeddingN, neg_weight):
    """
    compute and return the loss for unspervised model based on Eq (1) in the
    GraphSage paper

    :param 2d-tensor embeddingA: embedding of a list of nodes
    :param 2d-tensor embeddingB: embedding of a list of neighbor nodes
                                 pairwise to embeddingA
    :param 2d-tensor embeddingN: embedding of a list of non-neighbor nodes
                                 (negative samples) to embeddingA
    :param float neg_weight: negative weight
    """
    # positive affinity: pair-wise calculation
    # 边的两端节点对应相乘，求相似度
    pos_affinity = tf.reduce_sum ( tf.multiply ( embeddingA, embeddingB ), axis=1 )
    
    # negative affinity: enumeration of all combinations of (embeddingA, embeddingN)
    # 每个正样本都和负样本求相似度
    neg_affinity = tf.matmul ( embeddingA, tf.transpose ( embeddingN ) )

    pos_xent = tf.nn.sigmoid_cross_entropy_with_logits ( tf.ones_like(pos_affinity)
                                                       , pos_affinity
                                                       , "positive_xent" )
    # p1:[1,2,3],p2:[[0.1,0.5,0.4]]
    neg_xent = tf.nn.sigmoid_cross_entropy_with_logits ( tf.zeros_like(neg_affinity)
                                                       , neg_affinity
                                                       , "negative_xent" )

    weighted_neg = tf.multiply ( neg_weight, tf.reduce_sum(neg_xent) )
    batch_loss = tf.add ( tf.reduce_sum(pos_xent), weighted_neg )

    # per batch loss: GraphSAGE:models.py line 378
    return tf.divide ( batch_loss, embeddingA.shape[0] )
