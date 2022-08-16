import os
import hickle as hkl
import pickle as pkl
import sonnet as snt
import graph_nets as gn
import numpy as np

MODELS = ['edge', 'node', 'global']
SAVE_NAMES = ['edge_model', 'node_model', 'global_model']
FILE_EXTENSION = '.pkl'

def create_folder(path):
    if not os.path.exists(path): os.mkdir(path)

def add_path(path, dir):
    return f"{path}/{dir}"

def save_sntmodel(model, path):
    #hkl.dump(model, path + FILE_EXTENSION, mode = 'w')
    file_path = path + FILE_EXTENSION
    with open(file_path, 'wb') as f:
        pkl.dump(model, f)
    print(f"Saved {model} at {path}")

def load_sntmodel(path):
    #return hkl.load(path+ FILE_EXTENSION)
    file_path = path + FILE_EXTENSION
    with open(file_path, 'rb') as f:
        model = pkl.load(f)
    print(f"Loaded {model} at {path}")
    return model

def save_model(models, filepath = 'saved_models/tmp/', norm = None):
    create_folder(filepath)
    for i in range(len(SAVE_NAMES)):
        if models[i] != None:
            temp = add_path(filepath, SAVE_NAMES[i])
            #create_folder(temp)
            save_sntmodel(models[i], temp)
    
    if norm != None:
        temp = add_path(filepath, 'norm')
        save_model(norm.models, filepath = temp)

def load_model(filepath = 'saved_models/tmp/'):
    if not os.path.exists(filepath): return None
    loaded_models = []
    for i in SAVE_NAMES:
        temp = add_path(filepath, i) 
        if os.path.exists(temp + FILE_EXTENSION): 
            temp_model = load_sntmodel(temp)
            loaded_models.append(temp_model)
        else: 
            loaded_models.append(None)
            print(f"{temp} does not exist")
    
    temp = add_path(filepath, 'norm')
    if os.path.exists(temp):
        norm_models = load_model(temp)[0]
        # We make the assumption that a normalization layer
        # doesn't need a normalization layer of it's own
        return loaded_models, myNormModel(pre_models = norm_models)
    else:
        return loaded_models, None
# %%
class myNormModel(snt.Module):
    models = None
    layer = None
    def __init__(self, use_edges = False, use_nodes = False, use_globals = False, pre_models = None) -> None:
        super().__init__()
        allowed_models = [use_edges, use_nodes, use_globals]
        self.models = []
        if pre_models is None:
            for a in allowed_models:
                temp_model = snt.LayerNorm(0, create_scale=True, create_offset=True) if a else None
                self.models.append(temp_model)
        else: self.models = pre_models
        
        norm_model = gn.modules.GraphIndependent(
            edge_model_fn= None if self.models[0] is None else lambda: self.models[0],
            node_model_fn=None if self.models[1] is None else lambda: self.models[1],
            global_model_fn=None if self.models[2] is None else lambda: self.models[2],
        )
        self.layer = norm_model
            
    def __call__(self, graph):
        return self.layer(graph)
def give_norm_model(use_edges = False, use_nodes = False, use_globals = False, pre_models = None):
    """
    This method returns a `gn.modules.GraphIndependent` 
    network Normalization Layer
    """
    allowed_models = [use_edges, use_nodes, use_globals]
    models = []

    if not np.any(allowed_models) and pre_models is None: return None
    
    norm_model = myNormModel(use_edges, use_nodes, use_globals, pre_models)
    return norm_model


def give_pass_through():
    return myNormModel()
# %%

class myGraphNetwork(snt.Module):
    norm = None
    layers = None
    models = None
    modelname = None
    model_submodule_layers = None

    def __init__(self, name, nlabels = 3, edge_layers = None, node_layers = None, global_layers = None, output_func = ['global'], pre_norm = None, pre_models = None):
        """
        
        """
        super().__init__()
        self.modelname = name
        if pre_models != None: 
            self.models = pre_models
        else:
            use_edges = edge_layers is not None
            use_nodes = node_layers is not None
            use_globals = global_layers is not None

            edge_model_layers = edge_layers[:] if use_edges else None
            node_model_layers = node_layers[:] if use_nodes else None
            global_model_layers = global_layers[:] if use_globals else None
            try:
                if 'node' in output_func:
                    print("Node output")
                    node_model_layers.append(nlabels)
                if 'global' in output_func:
                    print("Global output")
                    global_model_layers.append(nlabels)
            except AttributeError as e:
                print(f"Can't create model. Outputs {output_func} cannot be set with no model.")
                print("Set layers as [] for no hidden layers")
                assert False
            self.model_submodule_layers = [edge_model_layers, node_model_layers, global_model_layers]
            self.models = []

            if use_edges: # edge
                self.models.append(snt.nets.MLP(edge_model_layers)) 
            else: self.models.append(None)

            if use_nodes: # node
                self.models.append(snt.nets.MLP(node_model_layers)) 
            else: self.models.append(None)

            if use_globals: # global
                self.models.append(snt.nets.MLP(global_model_layers)) 
            else: self.models.append(None)


        graph_model = self.build()
        self.layers = []
        if pre_norm != None:
            self.layers.append(pre_norm)
            self.norm = pre_norm
        self.layers.append(graph_model)
    
    def __call__(self, graph):
        for layer in self.layers:
            graph = layer(graph)
        return graph

    def build(self) -> None:
        """
        Order of models -> ['edge', 'node', 'global']
        """

        # CHECK
        assert self.models[0] != None, "Graph Network requires an edge model"
        assert self.models[1] != None, "Graph Network requires a node model"
        assert self.models[2] != None, "Graph Network requires a global model"

        # CREATE NETWORK
        graph_model = gn.modules.GraphNetwork(
            edge_model_fn=lambda: self.models[0],
            node_model_fn=lambda: self.models[1],
            global_model_fn=lambda: self.models[2]
        )
        return graph_model

    def save(self, path, verbose = True):
        save_dir = add_path(path, self.modelname)
        print(f"Saving models", self.models)
        print(f"Saving norm", self.norm)
        save_model(self.models, save_dir, self.norm)
        print(f"Saved to {path}")
    
    def load(name, save_dir = 'saved_models'):
        temp = add_path(save_dir, name)
        net_models, norm_models = load_model(temp)
        norm_net = give_norm_model(pre_models=norm_models)
        
        obj = myGraphNetwork(name, pre_models=net_models, pre_norm=norm_net)
        print(f"Loaded model from {temp}")
        print(obj)
        return obj

# %%
class myGraphIndep(myGraphNetwork):
    def __init__(self, name, nlabels=3, edge_layers=None, 
    node_layers=None, global_layers=None, output_func=['global'], pre_norm=None, pre_models=None):
        super().__init__(name, nlabels, edge_layers, 
        node_layers, global_layers, output_func, pre_norm, pre_models)
    
    def build(self) -> None:
        if self.models[0] is None: print("No edge model set")
        if self.models[1] is None: print("No node model set")
        if self.models[2] is None: print("No global model set")
        graph_model = gn.modules.GraphIndependent(
            edge_model_fn= None if self.models[0] is None else lambda: self.models[0],
            node_model_fn=None if self.models[1] is None else lambda: self.models[1],
            global_model_fn=None if self.models[2] is None else lambda: self.models[2],
        )
        return graph_model

    def load(name, save_dir = 'saved_models'):
        temp = add_path(save_dir, name)
        net_models, norm= load_model(temp)
        obj = myGraphIndep(name, pre_models=net_models, pre_norm=norm)
        return obj
    
class myDeepSets(myGraphNetwork):
    def __init__(self, name, nlabels=3, 
    node_layers=None, global_layers=None, output_func=['global'], pre_norm=None, pre_models=None):
        super().__init__(name, nlabels, None, 
        node_layers, global_layers, output_func, pre_norm, pre_models)

    def build(self) -> None:
        """
        """
        # CHECK
        assert self.models[0] == None, "Deep Sets does not require an edge model"
        assert self.models[1] != None, "Deep Sets requires a node model"
        assert self.models[2] != None, "Deep Setsrequires a global model"

        # CREATE NETWORK
        graph_model = gn.modules.DeepSets(
            node_model_fn=lambda: self.models[1],
            global_model_fn=lambda: self.models[2]
        )
        return graph_model