import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

class MultiheadCoattention(nn.Module):
    '''
    Implementation of multi-head attention adapted from https://github.com/shichence/AutoInt
    The class implements the multi-head co-attention algorithm.
    '''

    def __init__(self, embedding_dim, num_units, num_heads=2, dropout_keep_prob=1.):
        super(MultiheadCoattention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_units = num_units
        self.key_dim = nn.Parameter(torch.tensor(data=[num_units//num_heads], requires_grad=False, dtype=torch.float32))

        self.Q = torch.nn.Linear(embedding_dim, num_units)
        self.K = torch.nn.Linear(embedding_dim, num_units)
        self.V = torch.nn.Linear(embedding_dim, num_units)
        self.res_k = torch.nn.Linear(embedding_dim, num_units)
        self.res_q = torch.nn.Linear(embedding_dim, num_units)

        self.softmax_q = nn.Softmax(dim=1)
        self.softmax_k = nn.Softmax(dim=2)

        self.dropout = torch.nn.Dropout(1-dropout_keep_prob)
        self.layer_norm = nn.LayerNorm(num_units)

    def forward(self, queries, keys, values, has_residual=True):
        Q = F.relu(self.Q(queries))
        K = F.relu(self.K(keys))
        V = F.relu(self.V(values))
        if has_residual:
            res_k = F.relu(self.res_k(queries))
            res_q = F.relu(self.res_q(values))

        # split heads
        chunk_size = int(self.num_units / self.num_heads)
        Q_ = torch.cat(torch.split(Q, chunk_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, chunk_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, chunk_size, dim=2), dim=0)
        # get scaled similarity
        weights = torch.bmm(Q_, K_.transpose(1, 2))
        weights = weights / torch.sqrt(self.key_dim)
        # save similarities for later inspection
        ret_weights = weights.clone()
        #ret_weights = ret_weights.cpu().detach().numpy()
        # get weights
        weights_k = self.softmax_k(weights) # prob dist over keys
        weights_q = self.softmax_q(weights) # prob dist over queries
        weights_k = self.dropout(weights_k)
        weights_q = self.dropout(weights_q)
        # get outputs
        v_out = torch.bmm(weights_k, V_)
        q_out = torch.bmm(weights_q.transpose(1, 2), Q_)
        # reshuffle for heads
        restore_chunk_size = int(v_out.size(0) / self.num_heads)
        v_out = torch.cat(torch.split(v_out, restore_chunk_size, dim=0), dim=2)
        q_out = torch.cat(torch.split(q_out, restore_chunk_size, dim=0), dim=2)
        # add residual connection
        if has_residual:
            v_out += res_k
            q_out += res_q
        # combine latent spaces through addition and normalise
        outputs = v_out + q_out
        outputs = F.relu(outputs)
        outputs = self.layer_norm(outputs)

        return outputs, ret_weights, weights_k
