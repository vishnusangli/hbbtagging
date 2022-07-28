import sonnet as snt
import graph_nets as gn

class INModel(snt.Module):
    """
    A GNN based on `nglayers` of interaction networks.

    The input for all nodes/edges/global is first normalized using
    `snt.LayerNorm` operating in `gn.modules.GraphIndependent`.

    The output is a relation network with `nlabels` global outputs.
    """
    def __init__(self, nlabels, nglayers=0):
        """
        `nglayers`: number of layers of the interaction network
        """
        super(INModel, self).__init__()

        self.glayers = []
        for i in range(nglayers):
            graph_network = gn.modules.InteractionNetwork(
                node_model_fn=lambda: snt.nets.MLP([2]),
                edge_model_fn=lambda: snt.nets.MLP([2]),
            )
            self.glayers.append(graph_network)

        self.olayer = gn.modules.RelationNetwork(
            edge_model_fn=lambda: snt.nets.MLP([2]),
            global_model_fn=lambda: snt.nets.MLP([nlabels])
        )

    def __call__(self,data):
        for glayer in self.glayers:
            data = glayer(data)
        return self.olayer(data)


class GNModel(snt.Module):
    def __init__(self, name = None, OUTPUT_EDGE_SIZE = 2, OUTPUT_NODE_SIZE = 2, nlabels = 3, nlayers = 1):
        super().__init__(name)
        self.norm = gn.modules.GraphIndependent(
            node_model_fn  = None,
            edge_model_fn  = None,
            global_model_fn=lambda: snt.LayerNorm(0, create_scale=True, create_offset=True)
        )

    
        self.layers = []
        for layer in range(nlayers):
            graph_network = gn.modules.GraphNetwork(
                edge_model_fn=lambda: snt.nets.MLP([OUTPUT_EDGE_SIZE]),
                node_model_fn=lambda: snt.nets.MLP([OUTPUT_NODE_SIZE]),
                global_model_fn=lambda: snt.nets.MLP([256, 256, nlabels])
            )
            self.layers.append(graph_network)
    def __call__(self, graph):
        graph = self.norm(graph)
        for layer in self.layers:
            graph = layer(graph)
        return graph
        

class DSModel(snt.Module):
    def __init__(self, name = None, OUTPUT_NODE_MLP = [2], nlabels = 3, nlayers = 1, hid_layers = []):
        super().__init__(name)

        self.norm = gn.modules.GraphIndependent(
            node_model_fn  = None,
            edge_model_fn  = None,
            global_model_fn=lambda: snt.LayerNorm(0, create_scale=True, create_offset=True)
        )

        self.layers = []
        for layer in range(nlayers):
            ds_network = gn.modules.DeepSets(
                node_model_fn=lambda: snt.nets.MLP([3]),
                global_model_fn=lambda: snt.nets.MLP([256, 256, nlabels])
            )
            self.layers.append(ds_network)
            
    def __call__(self, graph):
        graph = self.norm(graph)
        for layer in self.layers:
            graph = layer(graph)
        return graph

class GIModel(snt.Module):
    """
    This is a graph independent network that copies the architecture 
    from the feature network
    """
    def __init__(self, nlabels = 3, hidden_layers = 3, hidden_size = 1024):
        super(GIModel, self).__init__()

        self.norm = gn.modules.GraphIndependent(
            node_model_fn  =lambda: snt.LayerNorm(0, create_scale=True, create_offset=True),
            edge_model_fn  =lambda: snt.LayerNorm(0, create_scale=True, create_offset=True),
            global_model_fn=lambda: snt.LayerNorm(0, create_scale=True, create_offset=True)
        )

        model_arch = []
        for i in range(hidden_layers):
            model_arch.append(hidden_size)
        model_arch.append(nlabels)

        self.layers = []
        gimodel = gn.modules.GraphIndependent(
            global_model_fn= lambda: snt.nets.MLP(model_arch)
        )
        self.layers.append(gimodel)
    
    def __call__(self, graph):
        graph = self.norm(graph)
        for layer in self.layers:
            graph = layer(graph)
        return graph