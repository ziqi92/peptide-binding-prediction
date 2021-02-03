import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from config import device, THRESHOLDS

class NPairLoss(nn.Module):
    datatype = ["npair"]
    labeltype = None
    defaulttype = "npair"
    
    def __init__(self):
        super().__init__()
        self.l2 = 0.001
        return
    
    def npair_distance(self, anchor, positive, negatives):
        """
        Compute basic N-Pair loss.
        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            n-pair loss (torch.Tensor())
        """
        return torch.log(1+torch.sum(torch.exp(anchor.mm((negatives-positive).transpose(0,1)))))
    
    def weightsum(self, anchor, positive):
        """
        Compute weight penalty.
        NOTE: Only need to penalize anchor and positive since the negatives are created based on these.
        Args:
            anchor, positive: torch.Tensor(), resp. embeddings for anchor and positive samples.
        Returns:
            torch.Tensor(), Weight penalty
        """
        return torch.sum(anchor**2+positive**2)
        
    def forward(self, y_pred, pairs):
        # prob size : batchsize * 
        loss = torch.stack([self.npair_distance(y_pred[npair[0]:npair[0]+1,:],y_pred[npair[1]:npair[1]+1,:],y_pred[npair[2:],:]) for npair in pairs])
        loss = loss + self.l2*torch.mean(torch.stack([self.weightsum(y_pred[npair[0],:], y_pred[npair[1],:]) for npair in pairs]))
        
        return torch.mean(loss)


class HingeLoss1(nn.Module):
    datatype = ["pair", "pair_level"]
    labeltype = "Continuous"
    defaulttype = "pair_level"
    
    def __init__(self):
        super().__init__()
        return
    
    def forward(self, si, sj, label):
        # prob size : batchsize *
        tmp = torch.zeros(si.size()).to(device)
 
        if THRESHOLDS[0] == 100:
            cost = torch.mean(torch.max(tmp, 0.2+label.unsqueeze(1) + sj - si))
        else:
            cost = torch.mean(torch.max(tmp, 1.0+label.unsqueeze(1) + sj - si))
        
        return cost

class HingeLoss2(nn.Module):
    datatype = ["pair", "pair_level"]
    labeltype = "Discrete"
    defaulttype = "pair_level"
    
    def __init__(self):
        super().__init__()
        return
    
    def forward(self, si, sj, label):
        # prob size : batchsize * 
        tmp = torch.zeros(si.size()).to(device)
        cost = torch.mean(torch.max(tmp, 0.2*label.unsqueeze(1) + sj - si))
        return cost
        
class HingeLoss3(nn.Module):
    datatype = ["pair_level_in"]
    labeltype = "Discrete"
    defaulttype = "pair_level_in"
    
    def __init__(self):
        super().__init__()
        return
        
    def forward(self, si, sj, label):
        # prob size : batchsize * 
        tmp = torch.zeros(si.size()).to(device)
        label = label.unsqueeze(1)
        sign = torch.sign(label)
        loss1 = sign*torch.max(tmp, 0.2*label + sj - si)
        loss2 = (1-sign)*torch.max(tmp, torch.abs(sj-si)-0.2)
        loss = torch.mean(loss1+loss2)
        
        return loss


class HingeLoss4(nn.Module):
    datatype = ["pair", "pair_level"]
    labeltype = "Continuous"
    defaulttype = "pair"
    
    def __init__(self):
        super().__init__()
        return
    
    def forward(self, si, sj, label):
        # prob size : batchsize *
        tmp = torch.zeros(si.size()).to(device)
 
        if THRESHOLDS[0] == 100:
            cost = torch.mean(torch.max(tmp, 0.2+label.unsqueeze(1) + sj - si))
        else:
            cost = torch.mean(torch.max(tmp, 1.0+label.unsqueeze(1) + sj - si))
        
        return cost
        
class LogLoss(nn.Module):
    datatype = ["pair", "pair_level"]
    labeltype = "Discrete"
    
    def __init__(self):
        super().__init__()
        return
    
    def forward(self, si, sj, label):
        # prob size : batchsize * 
        m = nn.Sigmoid()
        prob = m(si-sj).flatten()
        cost = torch.mean(-(label * torch.log(prob)))
        
        return cost

class MeanSquare(nn.Module):
    datatype = ["point"]
    labeltype = None
    defaulttype = "point"
    
    def __init__(self):
        super().__init__()
        return
    
    def forward(self, si, bi, ineq):
        # prob size : batchsize * 
        eq = torch.LongTensor(np.where(ineq=="=")[0]).to(device)
        leq = torch.LongTensor(np.where(ineq=="<")[0]).to(device)
        tmp1 = torch.zeros(leq.size()).to(device)
        geq = torch.LongTensor(np.where(ineq==">")[0]).to(device)
        tmp2 = torch.zeros(geq.size()).to(device)
        si = torch.transpose(si, 1, 0)
        
        try:
            cost = torch.mean((torch.index_select(bi, 0, eq)-torch.index_select(si, 1, eq))**2) if eq.shape[0] > 0 else 0
            cost += torch.mean(torch.max((torch.index_select(si, 1, leq)-torch.index_select(bi, 0, leq)),tmp1)**2) if leq.shape[0] > 0 else 0
            cost += torch.mean(torch.max((torch.index_select(bi, 0, geq)-torch.index_select(si, 1, geq)),tmp2)**2) if geq.shape[0] > 0 else 0
        except:
            pdb.set_trace()
        return cost
