import pdb
import torch
import attention
import torch.nn as nn
import torch.nn.functional as F
from sub_layers import LocalConv1d, PositionwiseFeedForward
from pos_embedding import PositionEmbedding
from nnutils import get_activation, aggregate_feature
from config import POS_DIM, AMINO_ACID, LEARNED_DIM

class ConvModel(nn.Module):
    padding = False
    
    def __init__(self, model_part, seq_len, feature_num, conv_layers, layer_sizes, learned, embedding_dim, activation, output_activation, transfer, transfer_dim, dropout, posembed=True, batchnorm=True):
        super().__init__()
        # posembed
        
        self.learned = learned
        if learned:
            self.embed = nn.Embedding(AMINO_ACID, LEARNED_DIM, padding_idx=0)
        
        self.addpos = posembed
        self.batchnorm = batchnorm
        self.feature_num = feature_num
        if posembed == True:
            self.posembed = PositionEmbedding(seq_len, POS_DIM, "Random")
            self.feature_num += POS_DIM*2
        
        self.clayer_num = len(conv_layers)
        self.kmer = 0
        # conv
        kw = self.feature_num
        dw = seq_len
        if self.clayer_num > 0:
            self.convs = nn.ModuleList()
            self.cactive = nn.ModuleList()
            if batchnorm:
                self.bns1 = nn.ModuleList()
            for i, conv_layer in enumerate(conv_layers):
                self.kmer += conv_layer["kernel_size"]-1
                self.convs.append(nn.Conv1d(kw, conv_layer["filters"], conv_layer["kernel_size"]))
                kw = conv_layer["filters"]
                dw = dw - conv_layer["kernel_size"] + 1
                self.cactive.append(get_activation(conv_layer["activation"]))
                if batchnorm:
                    self.bns1.append(nn.BatchNorm1d(conv_layer["filters"]))
        else:
            raise ValueError("The number of convolution layer must be larger than zero.")
            
        self.layer_num = len(layer_sizes)
        
        self.model_part = model_part
        self.attn, output_size = aggregate_feature(model_part, kw, dw)
        
        self.transfer = transfer
        if transfer:
            self.trans_layer = nn.Linear(transfer_dim, int(output_size/2))
            
        if self.layer_num > 0:
            if batchnorm:
                self.bns2 = nn.ModuleList()
            self.denses = nn.ModuleList()
            self.dactive = nn.ModuleList()
            for i, layer_size in enumerate(layer_sizes):
                if transfer and i==0:
                    self.denses.append(nn.Linear(int(output_size*3/2), layer_size))
                    self.layer_norm = nn.LayerNorm(int(output_size*3/2))
                else:
                    self.denses.append(nn.Linear(output_size, layer_size))
                output_size = layer_size
                self.dactive.append(get_activation(activation))
                
                if batchnorm:
                    self.bns2.append(nn.BatchNorm1d(layer_size))
        
        self.output_layer = nn.Linear(output_size, embedding_dim)
        self.output_active = get_activation(output_activation)
        
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.dropout(dropout)
            
    def forward(self, enc_input, lenc_input, seq_len, trans, padded_input=None):
        if self.learned:
            lenc_input = self.embed(lenc_input)
            lenc_input = torch.transpose(lenc_input, 1, 2)
            enc_input = torch.cat((enc_input, lenc_input),1) if enc_input is not None else lenc_input
            
        if self.addpos == True:
            enc_input = self.posembed(enc_input, seq_len)

        enc_output = enc_input
        for i in range(self.clayer_num):
            if self.batchnorm:
                enc_output = self.cactive[i](self.bns1[i](self.convs[i](enc_output)))
            else:
                enc_output = self.cactive[i](self.convs[i](enc_output))
        
        enc_output = torch.transpose(enc_output, 1, 2)
        ait_norm = None
        if self.model_part == "Attention":
            tmp_seq_len = seq_len - self.kmer
            enc_output, ait_norm = self.attn(enc_output, tmp_seq_len)
        else:
            enc_output = enc_output.flatten(1)
        
        hit = enc_output
        if self.transfer:
            trans = F.relu(self.trans_layer(trans))
            hit = self.layer_norm(torch.cat((hit, trans),1))
            
        if self.layer_num > 0:
            for i in range(self.layer_num):
                if self.batchnorm:
                    hit = self.dactive[i](self.bns2[i](self.denses[i](hit)))
                else:
                    hit = self.dactive[i](self.denses[i](hit))
                
                if self.dropout != None:
                    hit = self.dropout(hit)
        
        score = self.output_layer(hit)
        if self.output_active != None:
            score = self.output_active(score)
        
        return score, ait_norm, enc_output
        

class SpannyConvModel(nn.Module):
    padding = True
    
    def __init__(self, model_part, seq_len, feature_num, global_conv, conv_layers, layer_sizes, learned, embedding_dim, activation, output_activation, transfer, transfer_dim, dropout, posembed=True, batchnorm=True):
        super().__init__()
        
        self.learned = learned
        if learned:
            self.embed = nn.Embedding(AMINO_ACID, LEARNED_DIM, padding_idx=0)
                
        self.conv2 = torch.nn.Conv1d(feature_num, global_conv["filters"], seq_len)
        self.active2 = get_activation(global_conv["activation"])
        
        # posembed
        self.addpos = posembed
        self.batchnorm = batchnorm
        if posembed == True:
            self.posembed = PositionEmbedding(seq_len, POS_DIM, "Random")
            feature_num = feature_num + POS_DIM*2
        
        self.clayer_num = len(conv_layers)
        self.kmer = 0
        # conv
        dw = seq_len
        kw = feature_num
        if self.clayer_num > 0:
            self.convs = nn.ModuleList()
            self.cactive = nn.ModuleList()
            
            if batchnorm:
                self.bns1 = nn.ModuleList()
            for i, conv_layer in enumerate(conv_layers):
                self.kmer += conv_layer["kernel_size"]-1
                self.convs.append(nn.Conv1d(kw, conv_layer["filters"], conv_layer["kernel_size"]))
                dw = dw - conv_layer["kernel_size"] + 1
                kw = conv_layer["filters"]
                self.cactive.append(get_activation(conv_layer["activation"]))
                if batchnorm:
                    self.bns1.append(nn.BatchNorm1d(conv_layer["filters"]))
        else:
            raise ValueError("The number of convolution layer must be larger than zero.")
        
        if batchnorm:
            self.bns1.append(nn.BatchNorm1d(global_conv["filters"]))
        
        self.model_part = model_part
        self.attn, output_size = aggregate_feature(model_part, kw, dw)
        
        output_size = output_size + global_conv["filters"]
        self.layer_num = len(layer_sizes)
        
        self.transfer = transfer
        if transfer:
            self.trans_layer = nn.Linear(transfer_dim, int(output_size/2))
            output_size = int(output_size*3/2)
        
        if self.layer_num > 0:
            if batchnorm:
                self.bns2 = nn.ModuleList()
            self.denses = nn.ModuleList()
            self.dactive = nn.ModuleList()
            
            for i, layer_size in enumerate(layer_sizes):
                self.denses.append(nn.Linear(output_size, layer_size))
                output_size = layer_size
                self.dactive.append(get_activation(activation))
                
                if batchnorm:
                    self.bns2.append(nn.BatchNorm1d(layer_size))
        
        self.dropout = None
        
        if dropout > 0:
            self.dropout = nn.dropout(dropout)
            
        self.output_layer = nn.Linear(output_size, embedding_dim)
        self.output_active = get_activation(output_activation)
        
    def forward(self, enc_input, lenc_input, seq_len, trans, padded_input=None):
        global_feature = padded_input[0]
        if self.learned:
            lenc_input = self.embed(lenc_input)
            lenc_input = torch.transpose(lenc_input, 1, 2)
            enc_input = torch.cat((enc_input, lenc_input),1) if enc_input is not None else lenc_input   

            penc_input = self.embed(padded_input[1])
            penc_input = torch.transpose(penc_input, 1, 2)
            global_feature = torch.cat((global_feature, penc_input),1) if global_feature is not None else penc_input   


        # global kernel
        if padded_input is not None:
            res = self.active2(self.bns1[-1](self.conv2(global_feature))).squeeze(1)
            res = res.flatten(1)
        
        if self.addpos == True:
            enc_input = self.posembed(enc_input, seq_len)
        
        enc_output = enc_input
        for i in range(self.clayer_num):
            if self.batchnorm:
                enc_output = self.cactive[i](self.bns1[i](self.convs[i](enc_output)))
            else:
                enc_output = self.cactive[i](self.convs[i](enc_output))
        
        enc_output = torch.transpose(enc_output, 1, 2)
        
        ait_norm = None
        if self.model_part == "Attention":
            tmp_seq_len = seq_len - self.kmer
            enc_output, ait_norm = self.attn(enc_output, tmp_seq_len)
        else:
            enc_output = enc_output.flatten(1)
            
        enc_output = torch.cat((enc_output, res), 1)
        
        hit = enc_output
        if self.transfer:
            trans = F.relu(self.trans_layer(trans))
            hit = torch.cat((hit, trans),1)
        
        if self.layer_num > 0:
            for i in range(self.layer_num):
                if self.batchnorm:
                    hit = self.dactive[i](self.bns2[i](self.denses[i](hit)))
                else:
                    hit = self.dactive[i](self.denses[i](hit))
                
                if self.dropout != None:
                    hit = self.dropout(hit)
        
        score = self.output_layer(hit)
        if self.output_active != None:
            score = self.output_active(score)
        
        return score, ait_norm, enc_output
        
class MHCflurry(nn.Module):
    padding = True
    
    def __init__(self, model_part, seq_len, feature_num, locally_connected_layers, layer_sizes, learned, embedding_dim, activation, output_activation, transfer, transfer_dim, dropout, batchnorm=False):
        """
        Args:
            model_part: use attention or simply flatten
            seq_len: length of input sequence
            feature_num: number of features for each word in the sequence.
            input_channel:
            locally_connected_layers: list of layers which contain attributes of each layer
                { "filters" : ? , "activation" : "tahn"/"relu", "kernel_size": 3 }
            layer_sizes: list of layers' size in dense layers
            activation: activation function of dense layers
            output_activation: activation function of output
        
        
        """
        super().__init__()
        
        self.learned = learned
        if learned:
            self.embed = nn.Embedding(AMINO_ACID, LEARNED_DIM, padding_idx=0)
            
        self.lclayer_num = len(locally_connected_layers)
        ##conv
        kw = feature_num
        dw = seq_len
        if self.lclayer_num > 0:
            self.convs = nn.ModuleList()
            self.lcactive = nn.ModuleList()
            for i, locally_connected_layer in enumerate(locally_connected_layers):
                dw = dw - locally_connected_layer["kernel_size"] + 1
                self.convs.append(LocalConv1d(kw, locally_connected_layer["filters"], dw, locally_connected_layer["kernel_size"]))
                kw = locally_connected_layer["filters"]
                self.lcactive.append(get_activation(locally_connected_layer["activation"]))
        
        self.model_part = model_part
        self.attn, output_size = aggregate_feature(model_part, kw, dw)
        
        self.transfer = transfer
        if transfer:
            self.trans_layer = nn.Linear(transfer_dim, int(output_size/2))
            
            
        self.layer_num = len(layer_sizes)
        if self.layer_num > 0:
            self.denses = nn.ModuleList()
            self.dactive = nn.ModuleList()
            for i, layer_size in enumerate(layer_sizes):
                if transfer:
                    self.denses.append(nn.Linear(int(output_size*3/2), layer_size))
                else:
                    self.denses.append(nn.Linear(output_size, layer_size))
                output_size = layer_size
                self.dactive.append(get_activation(activation))
        
        self.output_layer = nn.Linear(output_size, embedding_dim)
        self.output_active = get_activation(output_activation)
    
    
    def forward(self, enc_input, lenc_input, seq_len, trans, padded_input=None):
        if self.learned:
            lenc_input = self.embed(lenc_input)
            enc_input = torch.cat((enc_input, lenc_input),2) if enc_input is not None else lenc_input
        
        if self.lclayer_num > 0:
            for i in range(self.lclayer_num):
                enc_input = self.lcactive[i](self.convs[i](enc_input))
                enc_input = torch.transpose(enc_input, 1, 2)
        
        enc_output = enc_input
        if self.model_part == "Attention":
            enc_output, _ = self.attn(enc_output, seq_len)
        else:
            enc_output = enc_output.flatten(1)
        
        hit = enc_output
        if self.transfer:
            trans = F.relu(self.trans_layer(trans))
            hit = torch.cat((hit, trans),1)
            
        if self.layer_num > 0:
            for i in range(self.layer_num):
                hit = self.dactive[i](self.denses[i](hit))
                
        score = self.output_layer(hit)
        if self.output_active != None:
            score = self.output_active(score)
        
        return score, None, enc_output

class Transformer(nn.Module):
    ''' Compose with two layers '''
    padding = False
    
    def __init__(self, model_part, seq_len, d_model, d_inner, n_head, d_k, d_v, layer_sizes, learned, embedding_dim, activation, output_activation, transfer, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.learned = learned
        if learned:
            self.embed = nn.Embedding(AMINO_ACID, LEARNED_DIM, padding_idx=0)
        
        self.posembed = PositionEmbedding(seq_len, d_model, "Sinusoid")
        
        self.d_model = d_model
        self.slf_attn = attention.MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        
        self.model_part = model_part
        self.attn, output_size = aggregate_feature(model_part, d_model, seq_len)
        
        self.transfer = transfer
        if transfer:
            output_size = output_size * 2
            
        self.layer_num = len(layer_sizes)
        if self.layer_num > 0:
            self.denses = nn.ModuleList()
            self.dactive = nn.ModuleList()
            for i, layer_size in enumerate(layer_sizes):
                self.denses.append(nn.Linear(output_size, layer_size))
                output_size = layer_size
                self.dactive.append(get_activation(activation))
        
        self.output_layer = nn.Linear(output_size, embedding_dim)
        self.output_active = get_activation(output_activation)
        
    def forward(self, enc_input, lenc_input, seq_len, trans, padded_input=None):
        if self.learned:
            lenc_input = self.embed(lenc_input)
            enc_input = torch.cat((enc_input, lenc_input),2)

        enc_output = self.posembed(enc_input, seq_len)
        
        non_pad_mask = self.get_non_pad_mask(seq_len, enc_output.size()[1], self.d_model)
        
        enc_output, enc_slf_attn = self.slf_attn(
            enc_output, enc_output, enc_output, seq_len=seq_len)
        
        enc_output = enc_output.masked_fill(non_pad_mask, 0)
        
        enc_output = self.pos_ffn(enc_output)
        
        enc_output = enc_output.masked_fill(non_pad_mask, 0)
        
        if self.model_part == "Attention":
            enc_output, _ = self.attn(enc_output, seq_len)
        else:
            enc_output = enc_output.flatten(1)
        
        hit = enc_output
        if self.transfer:
            hit = torch.cat((hit, trans),1)
            
        if self.layer_num > 0:
            for i in range(self.layer_num):
                hit = self.dactive[i](self.denses[i](hit))
                
        score = self.output_layer(hit)
        if self.output_active != None:
            score = self.output_active(score)
        
        return score, enc_slf_attn, enc_output
    
    def get_non_pad_mask(self, seq_len, max_len, d_model):
        idx = torch.arange(max_len).view(1,-1,1).expand((seq_len.size()[0],-1, d_model)).to(device)
        len_expanded = seq_len.view(-1,1,1).expand((-1,max_len, d_model)).to(device)
        
        mask = idx >= len_expanded
        
        return mask
