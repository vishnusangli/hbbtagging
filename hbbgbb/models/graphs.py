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

        self.norm = gn.modules.GraphIndependent(
                node_model_fn  =lambda: snt.LayerNorm(0, create_scale=True, create_offset=True),
                edge_model_fn  =lambda: snt.LayerNorm(0, create_scale=True, create_offset=True),
                global_model_fn=lambda: snt.LayerNorm(0, create_scale=True, create_offset=True)
            )

        self.glayers = []
        for i in range(nglayers):
            graph_network = gn.modules.InteractionNetwork(
                node_model_fn=lambda: snt.nets.MLP([2]),
                edge_model_fn=lambda: snt.nets.MLP([2]),
            )

        self.olayer = gn.modules.RelationNetwork(
            edge_model_fn=lambda: snt.nets.MLP([2]),
            global_model_fn=lambda: snt.nets.MLP([nlabels])
        )

    def __call__(self,data):
        data = self.norm(data)
        for glayer in self.glayers:
            data = glayer(data)
        return self.olayer(data)
