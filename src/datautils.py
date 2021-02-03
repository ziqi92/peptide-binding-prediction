import torch
import numpy as np
import pandas as pd
import math
import config
from config import blosum_path, PEPTIDE_LENGTH, ENCODING_SIZE
import pdb

__blosum_mat = pd.read_csv(blosum_path, sep='\t')
__add_keys = lambda x, b, c : x.update({b: c})
__keys = {}
[__add_keys(__keys, idx, acid) for acid, idx in enumerate(__blosum_mat.keys())]

def extractFeature(data, feature_type, padding, peptide_idx=0, ba_idx=2, ineq_idx=1):
    '''
    extract Feature for specific model
    Args:
        feature_type : 
            if feature_type == 1; then embedding with blosum62
            if feature_type == 0; then return directly
    Output:
        feature: (num_peptides, , *) 
    '''
    peptides = data[:,peptide_idx]
    ba = np.array(data[:,ba_idx]).astype(np.float)
    ineq = data[:,ineq_idx]
    seq_len = np.array([len(p) for p in peptides])
    max_len = PEPTIDE_LENGTH
    if padding:
        tmp = [pad_sequence(peptide) for peptide in peptides]
        seq_len = np.ones_like(ba) * PEPTIDE_LENGTH
        peptides = tmp
    
    feature = None
    if 'Blosum_One-hot' in feature_type:
        feature = np.zeros([len(peptides), max_len, 2 * ENCODING_SIZE], dtype=float)
        for i, peptide in enumerate(peptides):
            feature[i,:len(peptide),:ENCODING_SIZE] = transform_blosum(peptide)
            feature[i,:len(peptide),ENCODING_SIZE:] = transform_one_hot(peptide)
                
    elif 'Blosum' in feature_type:
        feature = np.zeros([len(peptides), max_len, ENCODING_SIZE], dtype=float)
        for i, peptide in enumerate(peptides):
            feature[i,:len(peptide),:] = transform_blosum(peptide)
            
    elif 'One-hot' in feature_type:
        feature = np.zeros([len(peptides), max_len, ENCODING_SIZE], dtype=float)
        for i, peptide in enumerate(peptides):
            feature[i,:len(peptide),:] = transform_one_hot(peptide)
    
    
    learned_feature = None
    if 'Learned' in feature_type:
        learned_feature = np.zeros([len(peptides), max_len], dtype=float)
        for i, peptide in enumerate(peptides):
            for j, acid in enumerate(peptide):
                if acid in __keys and acid != 'X':
                    learned_feature[i, j] = __keys[acid]
                elif acid == 'X':
                    learned_feature[i, j] = 0
    
    if feature is None and learned_feature is None:
        raise ValueError("Unsupported Feature Type")
    else:
        return max_len, feature, learned_feature, ba, ineq, seq_len

def pad_sequence(seq):
    """Padding the sequence to 15
    """
    pad_len = 15-len(seq)
    
    if pad_len == 0:
        return seq
    else:
        # number of X needing to be added
        new_seq = seq[:4]+'X'*7+seq[-4:]
        if pad_len == 7:
            return new_seq
        else:
            pos = 4+math.ceil(pad_len/2)
            new_seq = new_seq[:pos]+seq[4:-4]+new_seq[pos+len(seq)-8:]
            return new_seq

def transform_blosum(seq):
    '''
    Output:
        seq: (kmer, 20)
    '''
    new_seq = np.zeros([len(seq), ENCODING_SIZE], dtype=float)
    for i, acid in enumerate(seq):
        new_seq[i,:] = __blosum_mat[acid][:ENCODING_SIZE]
        
    return new_seq

def transform_one_hot(seq):
    new_seq = np.zeros([len(seq), ENCODING_SIZE], dtype=float)
    for i, acid in enumerate(seq):
        if acid != 'X':
            new_seq[i, __keys[acid]-1] = 1
        
    return new_seq
    
def generate_pairs(ba, ineq, threshold, datatype, labeltype):
    generator = generate_batch(ba, ineq, threshold, np.size(ba), 1, 1, datatype, labeltype)
    idxs, pairs, weight = next(generator)
    return idxs, pairs, weight

def generate_batch(ba, ineq, threshold, batch_size, num_batch, num_epoch, datatype, labletype):
    if datatype == config.PAIR:
        return _generate_batch(ba, ineq, batch_size, num_batch, num_epoch, labletype)
    else:
        return _generate_batch_level(ba, ineq, threshold, batch_size, num_batch, num_epoch, datatype, labletype)

def _generate_batch(bas, ineq, batch_size, num_batch, num_epoch, labeltype):
    for k in range(num_epoch):
        for i in range(num_batch):
            idxs = np.random.choice(len(bas),batch_size,replace=False)
            
            pairs = []
            weight = []
            
            idxs1 = np.argsort(bas[idxs])
            idxs = idxs[idxs1]
            
            for i, idx in enumerate(idxs):
                ba = bas[idx]
                for j, idx_ in enumerate(idxs[i+1:]):
                    ba_ = bas[idx_]
                    # in order to strenthen the relationship between two peptides,  only consider 2*  
                    w = max(0, 1-math.log(ba, 5e4)) - max(0, 1-math.log(ba_, 5e4))
                    if w > 0.1:
                        pairs.append([idx, idx_])
                        weight.append(w)
            
            weight = np.array(weight)
            pairs = np.array(pairs)
            yield idxs, pairs, weight
    

def _generate_batch_level(ba, ineq, threshold, batch_size, num_batch, num_epoch, datatype, labletype):
    ncls = np.size(threshold)+1
    # the number of instance in each class we need to select
    posidx = []
    
    for i in range(len(threshold)):
        if i == 0:
            tmp1 = np.where(ba<threshold[i])[0]
        elif i < len(threshold):
            tmp1 = np.where((threshold[i-1]<ba) & (ba<threshold[i]))[0]
        
        tmp2 = np.where((ba==threshold[i-1]) & (ineq=='<'))[0]
        tmp = np.concatenate((tmp1, tmp2), axis=0)
        
        if datatype != config.NPAIR and np.size(tmp, 0) > 0:
            posidx.append(tmp)
        elif datatype == config.NPAIR and np.size(tmp, 0) > 1:
            posidx.append(tmp)
        else:
            ncls = ncls - 1
    
    tmp1 = np.where(ba>threshold[-1])[0]
    tmp2 = np.where((ba==threshold[-1]) & ((ineq=='=') | (ineq=='>')))[0]
    tmp = np.concatenate((tmp1, tmp2), axis=0)
    
    if np.size(tmp, 0) >= 2:
        posidx.append(tmp)
    else:
        ncls = ncls - 1
    
    ninst = int(batch_size / (ncls+1))
    
    cls_npairs = int(ninst / 2)
    npairs = cls_npairs * ncls
    
    for k in range(num_epoch):
        for i in range(num_batch):
            pairs = []
            weight = []
            if datatype == config.NPAIR:
                for _ in range(cls_npairs):
                    positives = [np.random.choice(posidx[j],2,replace=False) for j in range(ncls)]
                    for j in range(ncls):
                        npair = np.concatenate((positives[j], np.delete(positives,j, axis=0)[:,1]), axis=0)
                        pairs.append(npair)
                        
            elif datatype == config.PAIR_LEVEL or datatype == config.PAIR_LEVEL_IN:
                anchors = [np.random.choice(posidx[j],ninst,replace=False) if len(posidx[j]) > ninst else posidx[j] for j in range(ncls)]
                for j in range(ncls):
                    if datatype == config.PAIR_LEVEL_IN:
                        sj = j
                    else:
                        sj = j+1
                    for p, val in enumerate(anchors[j]):
                        ba1 = ba[val]
                        pairs.extend([[val, val_] for n in range(sj, ncls) for val_ in anchors[n]])
                        if labletype == 'Discrete':
                            weight.extend([n-j for n in range(sj, ncls) for _ in anchors[n]])
                        elif labletype == 'Continuous':
                            try:
                                if threshold[0] == 100:
                                    weight.extend([max(0, 1-math.log(ba1, 5e4)) - max(0, 1-math.log(ba[val_], 5e4)) for n in range(sj, ncls) for val_ in anchors[n]])
                                else:
                                    weight.extend([abs(ba1-ba[val_]) for n in range(sj, ncls) for val_ in anchors[n]])
                            except:
                                pdb.set_trace()
            else:
                raise ValueError("Unsupported datatype: "+datatype)
            
            pairs = np.array(pairs)
            weight = np.array(weight)
            idxs = np.unique(pairs.flatten())
            yield idxs, pairs, weight
            
def convert_idxs(idxs, pairs):
    '''
    convert 
    '''
    idx_dict = {}
    for i, idx in enumerate(idxs):
        idx_dict[idx] = i
    
    new_pairs = np.zeros_like(pairs)
    for i, pair in enumerate(pairs):
        for j, inst in enumerate(pair):
            new_pairs[i,j] = idx_dict[inst]
            
    return new_pairs
