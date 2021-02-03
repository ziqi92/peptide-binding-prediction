import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from config import device

class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight_W_w = nn.Parameter(torch.Tensor(input_size, hidden_size).to(device))
        self.bias_w = nn.Parameter(torch.Tensor(hidden_size,1).to(device))
        self.weight_u_w = nn.Parameter(torch.Tensor(hidden_size,1).to(device))
        
        self.softmax = nn.Softmax()
        self.weight_W_w.data.uniform_(-0.1,0.1)
        self.bias_w.data.uniform_(-0.1,0.1)
        self.weight_u_w.data.uniform_(-0.1,0.1)
    
    def forward(self, embed, seq_len=None):
        uit = self.batch_matmul_bias(embed, nonlinearity='tanh')
        ait = self.batch_matmul(uit)
        
        if seq_len is not None:
            idx = torch.arange(ait.size(1)).unsqueeze(0).expand(ait.size()).to(device)
            len_expanded = seq_len.unsqueeze(1).expand(ait.size()).to(device)
            
            mask = idx < len_expanded
            ait[~mask] = float('-inf')
            
        ait_norm = self.softmax(ait)
        output = self.attention_mul(embed, ait_norm)
        return output, ait_norm
        
    def batch_matmul_bias(self, seq, nonlinearity=''):
        s = None
        bias_dim = self.bias_w.size()
        bias = self.bias_w
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], self.weight_W_w)
            _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
            if(nonlinearity=='tanh'):
                _s_bias = torch.tanh(_s_bias)
            _s_bias = _s_bias.unsqueeze(0)
            if(s is None):
                s = _s_bias
            else:
                s = torch.cat((s, _s_bias),0)
        return s.squeeze()
    
    def batch_matmul(self, seq, nonlinearity=''):
        s = None
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], self.weight_u_w)
            if(nonlinearity=='tanh'):
                _s = torch.tanh(_s)
            _s = _s.unsqueeze(0)
            if(s is None):
                s = _s
            else:
                s = torch.cat((s, _s),0)
        return s.squeeze()
        
    
    def attention_mul(self, nn_outputs, att_weights):
        attn_vectors = None
        for i in range(nn_outputs.size(0)):
            h_i = nn_outputs[i]
            a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
            h_i = a_i * h_i
            h_i = h_i.unsqueeze(0)
            if(attn_vectors is None):
                attn_vectors = h_i
            else:
                attn_vectors = torch.cat((attn_vectors,h_i),0)
        
        output = torch.sum(attn_vectors, 1)
        return torch.sum(attn_vectors, 1).to(device)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, key_mask, query_mask):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = attn.masked_fill(key_mask, float('-inf'))

        attn = self.softmax(attn)
        
        attn = attn.masked_fill(query_mask, 0)
        
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
        
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=math.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=math.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=math.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=math.pow(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, seq_len=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        key_mask = None
        query_mask = None
        
        if seq_len is not None:
            idx = torch.arange(len_q).view(1,1,-1).expand((sz_b,len_q,-1)).to(device)
            len_expanded = seq_len.view(-1,1,1).expand(idx.size()).to(device)
            
            key_mask = idx >= len_expanded
            key_mask = key_mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
            
            idx = torch.arange(len_q).view(1,-1,1).expand((sz_b,-1,len_k)).to(device)
            query_mask = idx >= len_expanded
            query_mask = query_mask.repeat(n_head, 1, 1)
            
        residual = q
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn = self.attention(q, k, v, key_mask, query_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn
