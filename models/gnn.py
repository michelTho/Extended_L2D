import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp import MLP


class GNN(nn.Module):
    """
    This class implements the graph neural network module.
    It must only comprehend a forward method : the backward pass will be handled
    by pytorch, during the learning phase of the RL model, using the autograd property
    """

    def __init__(self, n_mlp_layers, n_layers, input_dim, hidden_dim, device):
        super(GNN, self).__init__()
        self.device = device

        # Number of MLP which are applied (corresponds to K+1 in the paper)
        self.n_layers = n_layers

        # List of MLPs
        self.mlps = torch.nn.ModuleList()

        # List of batchnorms applied to the output of each MLP
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.n_layers - 1):
            if layer == 0:
                self.mlps.append(
                    MLP(
                        n_layers=n_mlp_layers,
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        batch_norm=True,
                    )
                )
            else:
                self.mlps.append(
                    MLP(
                        n_layers=n_mlp_layers,
                        input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        batch_norm=True,
                    )
                )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, features, adjacency_matrix):
        """
        This follows the implementation of "How Powerful are GNN"
        https://arxiv.org/pdf/1810.00826.pdf
        We use sum pooling for neighbourhood pooling, and mean pooling for graph
        pooling.
        At the end of the forward pass, we have a fixed size embedding for each node
        of the graph. A graph embedding is obtained by performing a pooling of the
        embeddings.
        We output the nodes and graph embedding, since actor model may need node
        embeddings too.
        features should be (batch_size, n_nodes, input_dim_feature_extractor) shaped
        adjacency_matrix should be (bacth_size, n_nodes, n_nodes) shaped
        """
        n_nodes = features.shape[1]
        h = features

        for layer in range(self.n_layers - 1):
            h = torch.sparse.mm(adjacency_matrix, h)
            h = self.mlps[layer](h)
            h = self.batch_norms[layer](h)
            h = F.relu(h)

        nodes_embeddings = h.clone()

        graph_pool = torch.ones((1, n_nodes)) / n_nodes
        graph_embedding = torch.sparse.mm(graph_pool, h)

        return graph_embedding, nodes_embeddings


# This is used for unit testing
if __name__ == "__main__":
    gnn = GNN(
        n_mlp_layers=3,
        n_layers=3,
        input_dim=4,
        hidden_dim=16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    adjacency_matrix = torch.sparse.Tensor([[1, 0, 1], [0, 1, 1], [1, 0, 1]])
    features = torch.Tensor(3, 4)
    g_embedding, n_embeddings = gnn.forward(features, adjacency_matrix)
    print(g_embedding)
    print(n_embeddings)
