class Memory:
    def __init__(self):
        # The state graphs are stored as torch_geometric.data.Data objects.
        # These objects can be built from a list of node features, and a list of edges
        self.graphs = []
        # The action that was taken. Is of the form (a, b) where a and b are the
        # indices of the nodes we want to connect
        self.action = []
        # The reward for the current state and action
        self.reward = []
        # Wheter the next state is terminal or not
        self.done = []
        # The log of the probability of the action that was taken.
        self.log_probability = []

    def compute_returns(self, gamma):
        returns = []
        discounted_return = 0
        for current_reward, current_done in zip(
            reversed(self.reward), reversed(self.done)
        ):
            if current_done:
                discounted_return = 0
            discounted_return = current_reward + gamma * discounted_return
            returns.append(discounted_return)
        returns = reversed(returns)
        return returns


# This is used for unit testing
if __name__ == "__main__":
    import torch

    memory = Memory()
    for i in range(10):
        memory.features.append(torch.rand(2, 4))
        memory.edge_index.append(
            torch.tensor(
                [[0, 0, 1, 1, 1, 1, 2, 2, 2, 3], [0, 3, 0, 1, 2, 3, 0, 2, 3, 3]],
                dtype=torch.long,
            )
        )
        memory.action.append((0, 1))
        memory.reward.append(-1)
        memory.done.append(False if i < 9 else True)
        memory.log_probability.append(torch.rand(1, 1))

    returns = memory.compute_returns(1)
    print(list(returns))
