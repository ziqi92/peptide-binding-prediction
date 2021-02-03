import numpy as np
import datetime
import argparse
import torch
import math
from models import SpannyConvModel, ConvModel, MHCflurry, Transformer
import os
import pandas as pd
from nnutils import create_var
from config import device, FLOAT, LONG
from datautils import generate_batch, convert_idxs, generate_pairs
from loss import NPairLoss
import test

def batch_train_pair(batch_data, batch_padded_data, batch_learned_data, transfer_data, ba, seq_len, pairs, weight, model, optim, criterion):
    '''
        training method for pairs of peptide
    '''
    model.train()
    optim.zero_grad()
    y_pred,_, _ = model(batch_data, batch_learned_data, seq_len, transfer_data, padded_input=batch_padded_data)
    
    mean_pred = torch.mean(y_pred)
    var_pred = torch.var(y_pred)
    max_pred = torch.max(y_pred)
    min_pred = torch.min(y_pred)
    
    if isinstance(criterion, NPairLoss):
        rankloss = criterion(y_pred, pairs)
    else:
        si = torch.index_select(y_pred, 0, pairs[:,0])
        sj = torch.index_select(y_pred, 0, pairs[:,1])
        rankloss = criterion(si, sj, weight)
    
    rankloss.backward(retain_graph=True)
    
    optim.step()
    
    return rankloss.item(), mean_pred, var_pred, max_pred, min_pred

def batch_train_point(batch_data, batch_padded_data, batch_learned_data, transfer_data, ba, ineq, seq_len, model, optim, criterion):
    '''
        training method for peptide
    '''
    model.train()
    
    ba = (1-np.log(ba)/np.log(5e4)).clip(0,1)
    ba = torch.FloatTensor(ba).to(device)
    
    optim.zero_grad()
    y_pred,_,_ = model(batch_data, batch_learned_data, seq_len, transfer_data, padded_input=batch_padded_data)
    
    mean_pred = torch.mean(y_pred)
    var_pred = torch.var(y_pred)
    max_pred = torch.max(y_pred)
    min_pred = torch.min(y_pred)
    
    rankloss = criterion(y_pred, ba, ineq)
    rankloss.backward(retain_graph=True)
    optim.step()
    
    return rankloss.item(), mean_pred, var_pred, max_pred, min_pred

def print_paras(file, model, paras):
    '''
    print the model's gradient and weight difference for debug
    '''
    tmp_paras = [(a.data).cpu().numpy() for a in model.parameters()]

    if paras is not None:
        tmp = "model weights: "
        for i in range(len(paras)):
            tmp += "{:.6f} ".format(np.mean(np.abs(tmp_paras[i]-paras[i])))
        tmp += '\n'
        print(tmp)
        file.write(tmp)
    
    paras = tmp_paras
    
    tmp = "model gradient: "
    for a in model.parameters():
        tmp += "max {:.6f}, min {:.6f}, mean {:.6f}, nonzero rate {:.6f}\n".format(torch.max(a.grad).item(), \
                                        torch.min(a.grad).item(), torch.mean(a.grad).item(), \
                                        (torch.nonzero(a.grad).size()[0]/(a.grad).flatten().size()[0]))        
    print(tmp)
    file.write(tmp)
    file.flush()
    
    return paras
    

def train(model, criterion, optimizer, scheduler, feature, padded_feature, learned_feature, ba, ineq, seq_len, threshold, datatype, batch_size, num_epochs, patience, outputfile, model_path, valid_split=0.1):
    """Define the neural network model
    """
    labeltype = criterion.__class__.labeltype
    if (isinstance(model, ConvModel) or isinstance(model, SpannyConvModel)) and feature is not None:
        feature = np.transpose((feature), (0,2,1))
        tmp = np.transpose((padded_feature[0]), (0,2,1)) if padded_feature is not None else None    
        padded_feature = (tmp, padded_feature[1]) if padded_feature is not None else None

    ## ==================== validation set =====================================
    # for early stopping and learning rate annealing
    num = int(len(ba) * valid_split)
    
    while True:
        valid_idx = np.random.choice(len(ba), num)
        valid_ba = ba[valid_idx]
        valid_ineq = ineq[valid_idx]
        
        if datatype != "point":
            _, valid_pairs, valid_weight = generate_pairs(valid_ba, valid_ineq, threshold, datatype, labeltype)
            if len(valid_pairs) > 0:
                break
        else:
            break
   
    if padded_feature is None: padded_feature = [None, None] 
    valid_feature = create_var(feature, device=device, dtype=FLOAT, idx=valid_idx)
    valid_padded_feature = create_var(padded_feature[0], device=device, dtype=FLOAT, idx=valid_idx)
    valid_padded_learned_feature = create_var(padded_feature[1], device=device, dtype=LONG, idx=valid_idx)
    valid_learned_feature = create_var(learned_feature, device=device, dtype=LONG, idx=valid_idx)
    valid_seq_len = create_var(seq_len, device=device, dtype=LONG, idx=valid_idx)
    valid_padded_feature = (valid_padded_feature, valid_padded_learned_feature)        

    feature = create_var(feature, device=device, dtype=FLOAT, ridx=valid_idx)
    tmp = create_var(padded_feature[0], device=device, dtype=FLOAT, ridx=valid_idx)
    padded_learned_feature = create_var(padded_feature[1], device=device, dtype=LONG, ridx=valid_idx)
    padded_feature = (tmp, padded_learned_feature)
    learned_feature = create_var(learned_feature, device=device, dtype=LONG, ridx=valid_idx)
    seq_len = create_var(seq_len, device=device, dtype=LONG, ridx=valid_idx)
   
    
    ba = np.delete(ba, valid_idx, 0)
    ineq = np.delete(ineq, valid_idx, 0)
    ## =========================================================================
    
    if model_path != "None":
        transfer_model = torch.load(model_path, map_location=device)
        for param in transfer_model.parameters():
            param.requires_grad = False
        
        feature_ = feature
        padded_feature_ = padded_feature
        valid_feature_ = valid_feature
        valid_padded_feature_ = valid_padded_feature
        if (isinstance(transfer_model, MHCflurry) and not isinstance(model, MHCflurry)) \
           or (not isinstance(transfer_model, MHCflurry) and isinstance(model, MHCflurry)):
            feature_ = torch.transpose(feature, 1, 2)
            valid_feature_ = torch.transpose(valid_feature, 1, 2)
            padded_feature_ = torch.transpose(padded_feature, 1, 2)
            valid_padded_feature_ = torch.transpose(valid_padded_feature, 1, 2)
        
        _,_,transfer_feature = transfer_model(feature_, learned_feature, seq_len, None, padded_input=padded_feature_)
        _,_,transfer_valid_feature = transfer_model(valid_feature_, valid_learned_feature, valid_seq_len, None, padded_input=valid_padded_feature_)
    else:
        transfer_feature = None
        transfer_valid_feature = None
    
    seq_num = len(ba)
    num_batch = math.ceil(seq_num / batch_size)
    
    if datatype != "point":
        train_generator = generate_batch(ba, ineq, threshold, batch_size, num_batch, num_epochs, datatype, labeltype)
        
    paras = None
    
    min_val_loss = None
    min_val_iteration = 0
    
    for j in range(num_epochs):
        for k in range(num_batch):
            time1 = datetime.datetime.now()
            if datatype != "point":
                batch_idx, pairs, weight = next(train_generator)
                pairs = convert_idxs(batch_idx, pairs)
                pairs = torch.LongTensor(pairs).to(device)
                weight = torch.FloatTensor(weight).to(device)
                
                if len(pairs) == 0:
                    continue
            else:
                batch_idx = np.random.choice(seq_num, batch_size)
                
            batch_data = create_var(feature, idx=batch_idx)
            batch_padded_data = create_var(padded_feature[0], idx=batch_idx)
            batch_padded_learn_data = create_var(padded_feature[1], idx=batch_idx)
            batch_padded_data = (batch_padded_data, batch_padded_learn_data)
            batch_learned_data = create_var(learned_feature, idx=batch_idx)
            transfer_batch_data = create_var(transfer_feature, idx=batch_idx)
            ba_batch = ba[batch_idx]
            ineq_batch = ineq[batch_idx]
            seq_len_batch = seq_len[batch_idx]
            
            if datatype != "point":
                loss, mean_pred, var_pred, max_pred, min_pred = batch_train_pair(batch_data, batch_padded_data, batch_learned_data, transfer_batch_data, ba_batch, seq_len_batch, pairs, weight, model, optimizer, criterion)
            else:
                loss, mean_pred, var_pred, max_pred, min_pred = batch_train_point(batch_data, batch_padded_data, batch_learned_data, transfer_batch_data, ba_batch, ineq_batch, seq_len_batch, model, optimizer, criterion)
                
            s = 'epoch {:d}, minibatch {:d}, loss {:.4f} |score mean {:.4f}, var {:.4f}, max {:.4f}, min {:.4f}\n'.format(j, k, loss, mean_pred, var_pred, max_pred, min_pred)
            outputfile.write(s)
            outputfile.flush()
        
        if datatype != "point":
            _, valid_loss, _ = test.test_pair(valid_feature, valid_padded_feature, valid_learned_feature, transfer_valid_feature, valid_seq_len, valid_pairs, valid_weight, model, criterion)
        else:
            _, valid_loss, _ = test.test_point(valid_feature, valid_padded_feature, valid_learned_feature, transfer_valid_feature, valid_ba, valid_ineq, valid_seq_len, model, criterion)
        
        s = 'valid loss {:.4f} \n'.format(valid_loss)
        print(s)
        outputfile.write(s)
        outputfile.flush()
        
        if min_val_loss == None or valid_loss < min_val_loss:
            min_val_loss = valid_loss
            min_val_iteration = j
    
        
        if j > min_val_iteration + patience and j > 50:
            s = 'Stopping at epoch {:d}/{:d}, loss {:.4f}, min_val_loss {:.4f} at epoch {:d}'.format(j, num_epochs, loss, min_val_loss, min_val_iteration)
            outputfile.write(s)
            outputfile.flush()
            break
        
        scheduler.step(valid_loss)
    
    train_output, train_ait, _ = model(feature, learned_feature, seq_len, transfer_feature, padded_input=padded_feature)
    
    if torch.cuda.is_available():
        train_output = train_output.cpu().detach().numpy()
    else:
        train_output = train_output.detach().numpy()
            
    return model, train_output, ba, train_ait
