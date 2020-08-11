import torch
import torch.nn as nn


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model=16, head_count=4, bias=True):
        super(MultiHeadAttention, self).__init__()
        self.head_count = head_count
        self.d_model = d_model
        self.d_k = d_model // head_count
        self.d_v = d_model // head_count
        self.nonlinear = nn.ReLU()
        assert d_model % head_count == 0, "d_model: {} should be divisible by head_count: {}".format(d_model, head_count)

        self.q_linear = nn.Linear(self.d_model, self.d_k * self.head_count, bias=bias)
        self.k_linear = nn.Linear(self.d_model, self.d_k * self.head_count, bias=bias)
        self.v_linear = nn.Linear(self.d_model, self.d_v * self.head_count, bias=bias)

        self.attention = ScaledDotProduct(self.d_k, head_count)

    def forward(self, query, key, value, mask=None):
        if len(query.shape) == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

        batch_size, node_size, _ = query.size()

        query = self.nonlinear(self.q_linear(query))
        query = query.reshape(batch_size, node_size, self.head_count, self.d_k).permute(0, 2, 1, 3).reshape(
            batch_size * self.head_count, node_size, self.d_k)
        # query = query.reshape(node_size, -1, self.head_count, self.d_k).transpose(1, 2).transpose(0, 1).transpose(1, 2)

        key = self.nonlinear(self.k_linear(key))
        key = key.reshape(batch_size, node_size, self.head_count, self.d_k).permute(0, 2, 1, 3).reshape(
            batch_size * self.head_count, node_size, self.d_k)
        # key = key.reshape(node_size, -1, self.head_count, self.d_k).transpose(1, 2).transpose(0, 1).transpose(1, 2)

        value = self.nonlinear(self.v_linear(value))
        value = value.reshape(batch_size, node_size, self.head_count, self.d_k).permute(0, 2, 1, 3).reshape(
            batch_size * self.head_count, node_size, self.d_k)
        # value = value.reshape(node_size, -1, self.head_count, self.d_k).transpose(1, 2).transpose(0, 1).transpose(1, 2)

        x, dist = self.attention(query, key, value, mask)
        x = x.reshape(batch_size, self.head_count, node_size, self.d_k).permute(0, 2, 1, 3).reshape(batch_size,
                                                                                                    node_size,
                                                                                                    self.d_model)
        # x = x.reshape(-1, node_size, self.d_model)

        return x, dist


class ScaledDotProduct(torch.nn.Module):

    def __init__(self, d_k, head_count):
        super(ScaledDotProduct, self).__init__()
        self.d_k = d_k
        self.head_count = head_count

    def forward(self, query, key, value, mask=None, dropout=None):
        x = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            mask = mask.repeat(self.head_count, 1, 1)
            x = x.masked_fill(mask == 0, -1e9)
        x = torch.nn.functional.softmax(x, dim=-1)
        dist = x
        if dropout is not None:
            x = torch.nn.Dropout(dropout)

        x = torch.matmul(x, value)
        return x, dist
