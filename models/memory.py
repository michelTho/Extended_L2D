class Memory:
    def __init__(self):
        # Initial features of the nodes of the graph.
        # Should be of size (n_nodes, input_dim_feature_extractor)
        self.features = []
        # The current adjacency_matrix of the nodes of the graph
        # Should be of size (n_nodes, n_nodes)
        self.adjacency_matrix = []
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
        memory.adjacency_matrix.append(
            torch.tensor([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 1]])
        )
        memory.action.append((0, 1))
        memory.reward.append(-1)
        memory.done.append(False if i < 9 else True)
        memory.log_probability.append(torch.rand(1, 1))

    returns = memory.compute_returns(1)
    print(list(returns))
