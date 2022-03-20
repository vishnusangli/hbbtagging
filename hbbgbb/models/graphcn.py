import sonnet as snt
import tensorflow as tf
import graph_nets as gn



class GraphConv(snt.Module):
    """
    A layer for the graph convolution network (GCN) with sparse inputs
    """
    def __init__(self, inp_feat, out_feat,  dropout = 0., activation = tf.nn.relu, sparse_inputs = True, bias = True, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.dropout = 0.
        self.activation = activation

        self.sparse_inputs = sparse_inputs
        self.bias = bias

        


    def __call__(self, data):
        pass

