import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


class GraphAttention(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphAttention, self).__init__()

        self.linear = nn.Linear(in_channels, out_channels)

        self.attention = nn.Sequential(
            nn.Linear((out_channels + out_channels) * 2, 1),
            nn.LeakyReLU()
        )

    def message(self, x_i: Tensor, x_j: Tensor, edge_index_i: Tensor, edge_index_j: Tensor, node_embeddings: Tensor) -> Tensor:
        node_embeddings_i = node_embeddings[edge_index_i]
        node_embeddings_j = node_embeddings[edge_index_j]

        g_i = torch.cat([x_i, node_embeddings_i], dim=-1)
        g_j = torch.cat([x_j, node_embeddings_j], dim=-1)
        g = torch.cat([g_i, g_j], dim=-1)

        pi = self.attention(g)
        alpha = softmax(pi, index=edge_index_i)

        attention_x_j = alpha * x_j

        return attention_x_j

    def forward(self, x: Tensor, edge_index: Tensor, node_embeddings: Tensor) -> Tensor:
        x = self.linear(x)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

        out = self.propagate(edge_index, x=x, node_embeddings=node_embeddings)

        return out
