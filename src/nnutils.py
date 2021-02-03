import torch
import numpy as np
import torch.nn as nn
from attention import SelfAttention

def get_activation(identifier):
    if identifier == None:
        return None
    else:
        try:
            active = eval("nn."+identifier+"()")
        except Exception as e:
            print(e)
            raise ValueError("Wrong identifier for activation function : " + identifier)
        
        return active
        
def create_var(tensor, device=None, dtype=None, idx=None, ridx=None):
    if tensor is None:
        return None
    
    data = tensor
    if idx is not None:
        data = data[idx,...]
    if ridx is not None:
        data = np.delete(tensor, ridx, 0)
    
    if dtype == "float":
        data = torch.FloatTensor(data)
    elif dtype == "long":
        data = torch.LongTensor(data)
        
    if device is not None:
        data = data.to(device)
        
    return data
    
def aggregate_feature(model_part, d_model, seq_len):
    if model_part == "Attention":
        attn = SelfAttention(d_model, d_model)
        output_size = d_model
    elif model_part == "Flatten":
        attn = None
        output_size = d_model * seq_len
    else:
        raise ValueError("Unsupported model choice: "+model_part+ ". Please use \"Attention\" or \"Flatten\".")
        
    return attn, output_size