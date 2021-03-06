import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, batch_norm, device):
        super(MLP, self).__init__()
        if n_layers < 3:
            raise Exception("Number of layers must be >= 3")
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        self.device = device
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.batch_norms = torch.nn.ModuleList()

        for i in range(self.n_layers - 1):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            elif i < self.n_layers - 2:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
            if self.batch_norm:
                if i < self.n_layers - 2:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                else:
                    self.batch_norms.append(nn.BatchNorm1d(output_dim))
        self.to(self.device)

    def forward(self, x):
        for layer in range(self.n_layers - 1):
            if self.batch_norm:
                x = F.relu(self.batch_norms[layer](self.layers[layer](x)))
            else:
                x = F.relu(self.layers[layer](x))
        return x


# This is used for unit testing
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = MLP(3, 8, 16, 2, False, device)
    tensor = torch.Tensor(4, 8)
    print(tensor)
    print(mlp(tensor))
    mlp = MLP(3, 8, 16, 2, True, device)
    print(mlp(tensor))
