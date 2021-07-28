import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv

from mlp import MLP


class ActorCritic(nn.Module):
    def __init__(
        self,
        n_jobs,
        n_machines,
        n_mlp_layers_feature_extractor,
        n_layers_feature_extractor,
        input_dim_feature_extractor,
        hidden_dim_feature_extractor,
        n_mlp_layers_actor,
        hidden_dim_actor,
        n_mlp_layers_critic,
        hidden_dim_critic,
        device,
    ):
        super(ActorCritic, self).__init__()
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.n_nodes = n_jobs * n_machines
        self.hidden_dim_feature_extractor = hidden_dim_feature_extractor

        self.n_layers_feature_extractor = n_layers_feature_extractor
        self.feature_extractors = torch.nn.ModuleList()

        for layer in range(self.n_layers_feature_extractor - 1):
            self.feature_extractors.append(
                GINConv(
                    MLP(
                        n_layers=n_mlp_layers_feature_extractor,
                        input_dim=input_dim_feature_extractor
                        if layer == 0
                        else hidden_dim_feature_extractor,
                        hidden_dim=hidden_dim_feature_extractor,
                        output_dim=hidden_dim_feature_extractor,
                        batch_norm=True,
                        device=device,
                    )
                )
            )

        self.actor = MLP(
            n_layers=n_mlp_layers_actor,
            input_dim=hidden_dim_feature_extractor * 3,
            hidden_dim=hidden_dim_actor,
            output_dim=1,
            batch_norm=False,
            device=device,
        )
        self.critic = MLP(
            n_layers=n_mlp_layers_critic,
            input_dim=hidden_dim_feature_extractor,
            hidden_dim=hidden_dim_critic,
            output_dim=1,
            batch_norm=False,
            device=device,
        )

    def forward(self, features, edge_index):
        for layer in range(self.n_layers_feature_extractor - 1):
            features = self.feature_extractors[layer](features, edge_index)

        graph_pooling = torch.ones((1, self.n_nodes)) / self.n_nodes
        graph_embedding = torch.sparse.mm(graph_pooling, features)
        value = self.critic(graph_embedding)

        possible_s_a_pairs = self.compute_possible_s_a_pairs(graph_embedding, features)
        pi = F.softmax(self.actor(possible_s_a_pairs), dim=0)

        return pi, value

    def compute_possible_s_a_pairs(self, graph_embedding, nodes_embedding):
        """
        Compute all possible actions for the state s. Since an action is equivalent to
        bouding 2 nodes, there are n_nodes^2 actions, i.e. (n_jobs * n_machines)^2
        actions.
        The graph_embedding should be (1, hidden_dim_feature_extractor) shaped
        The nodes_embedding should be (n_nodes, hidden_dim_feature_extractor) shaped
        One s_a_pair should be (1, 3 * hidden_dim_feature_extractor) shaped
        The final s_a_pairs sould be (n_nodes^2, 3*hidden_dim_feature_extractor) shaped
        """
        # We create 3 tensors representing state, node 1 and node 2
        # and then stack them together to get all state action pairs
        states = graph_embedding.repeat(self.n_nodes * self.n_nodes, 1)
        nodes1 = nodes_embedding.repeat(self.n_nodes, 1)
        nodes2 = nodes_embedding.repeat_interleave(self.n_nodes, dim=0)
        s_a_pairs = torch.hstack([states, nodes1, nodes2])
        return s_a_pairs


# This is used for unit testing
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = ActorCritic(
        n_jobs=2,
        n_machines=2,
        n_mlp_layers_feature_extractor=3,
        n_layers_feature_extractor=3,
        input_dim_feature_extractor=2,
        hidden_dim_feature_extractor=3,
        n_mlp_layers_actor=3,
        hidden_dim_actor=64,
        n_mlp_layers_critic=3,
        hidden_dim_critic=64,
        device=device,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 2, 2, 2, 3], [0, 3, 0, 1, 2, 3, 0, 2, 3, 3]],
        dtype=torch.long,
    )
    features = torch.rand(4, 2)
    print(actor_critic(features, edge_index))
