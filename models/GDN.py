import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from .GraphLayer import GraphLayer


class GDN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        sequence_len: int,
        d_hidden: int,
        d_output_hidden: int,
        k: int,
        num_output_layer: int = 1,
        *,
        dtype=None,
        device=None
    ):
        super(GDN, self).__init__()

        self.k = k

        self.dtype = dtype
        self.device = device

        self.embedding = nn.Embedding(num_nodes, d_hidden)
        self.graph_layer = GraphLayer(sequence_len, d_hidden)
        self.process_layer = nn.Sequential(
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.ModuleList()
        if num_output_layer == 1:
            self.output_layer.append(nn.Linear(d_hidden, 1))
        else:
            self.output_layer.append(nn.Linear(d_hidden, d_output_hidden))
            for i in range(num_output_layer - 2):
                self.output_layer.append(nn.Linear(d_output_hidden, d_output_hidden))
                self.output_layer.append(nn.BatchNorm1d(d_output_hidden))
                self.output_layer.append(nn.ReLU())
            self.output_layer.append(nn.Linear(d_output_hidden, 1))

        self.to(device)
        self.to(dtype)

    @staticmethod
    def __cos_similarity(x: Tensor, y: Tensor) -> Tensor:
        x_norm = x.norm(dim=-1).unsqueeze(-1)
        y_norm = y.norm(dim=-1).unsqueeze(-1)

        return (x @ y.T) / (x_norm @ y_norm.T)

    def __get_edges(self, x: Tensor, y: Tensor) -> Tensor:
        similarity = self.__cos_similarity(x, y)

        _, indices = torch.topk(similarity, self.k, dim=-1)

        source_nodes = torch.arange(x.shape[0], device=self.device).repeat_interleave(self.k)
        target_nodes = indices.reshape(-1)

        edges = torch.stack([source_nodes, target_nodes], dim=0)

        return edges

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_nodes, _ = x.shape

        node_embeddings = self.embedding(torch.arange(num_nodes).to(self.device))  # [num_nodes, d_hidden]
        edges = self.__get_edges(node_embeddings, node_embeddings)  # [2, num_nodes * k]

        data_list = [Data(x=x[i], edge_index=edges) for i in range(batch_size)]
        batch = Batch.from_data_list(data_list)

        node_embeddings_batch = node_embeddings.repeat(batch_size, 1)

        z = self.graph_layer(batch['x'], batch['edge_index'], node_embeddings_batch)  # [batch_size * num_nodes, d_hidden]

        s_hat = z * node_embeddings_batch
        s_hat = self.process_layer(s_hat)
        for model in self.output_layer:
            s_hat = model(s_hat)

        output = s_hat.reshape(batch_size, num_nodes)

        return output
