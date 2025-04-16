from torch import Tensor, nn

from .GraphAttention import GraphAttention


class GraphLayer(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(GraphLayer, self).__init__()

        self.graph_attention = GraphAttention(in_channel, out_channel)

        self.bn = nn.BatchNorm1d(out_channel)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor, edge_index: Tensor, node_embeddings: Tensor) -> Tensor:
        # x: [batch_size * num_nodes, sequence_len]

        out = self.graph_attention(x, edge_index, node_embeddings)

        out = self.bn(out)
        out = self.activation(out)

        return out
