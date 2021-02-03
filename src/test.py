import os
import torch
import numpy as np
import pandas as pd
import argparse
import metrics
from config import device
from models import ConvModel, SpannyConvModel, MHCflurry
import loss
from datautils import generate_pairs
from config import device, THRESHOLDS, codepath
from datautils import extractFeature
 
def test_with_print(feature, padded_feature, learned_feature, ba, ineq, seq_len, model, kmer, criterion, datatype, feature_type, model_path, file=None):
    if (isinstance(model, ConvModel) or isinstance(model, SpannyConvModel)) and feature is not None:
        feature = np.transpose((feature), (0,2,1))
        tmp = np.transpose((padded_feature[0]), (0,2,1)) if padded_feature is not None and padded_feature[0] is not None else None
        padded_feature = (tmp, padded_feature[1]) if padded_feature is not None and padded_feature[1] is not None else None
    
    feature = torch.FloatTensor(feature).to(device) if feature is not None else None
    tmp = torch.FloatTensor(padded_feature[0]).to(device) if padded_feature is not None and padded_feature[0] is not None else None
    padded_learned_feature = torch.LongTensor(padded_feature[1]).to(device) if padded_feature is not None and padded_feature[1] is not None else None
    padded_feature = (tmp, padded_learned_feature)
    seq_len = torch.LongTensor(seq_len).to(device)
    learned_feature = torch.LongTensor(learned_feature).to(device) if learned_feature is not None else None
    
    if model_path != "None":
        transfer_model = torch.load(model_path, map_location=device)
        
        feature_ = feature
        padded_feature_ = padded_feature
        if (isinstance(transfer_model, MHCflurry) and not isinstance(model, MHCflurry)) \
           or (not isinstance(transfer_model, MHCflurry) and isinstance(model, MHCflurry)):
            feature_ = torch.transpose(feature_, 1, 2)
            tmp = np.transpose((padded_feature_[0]), (0,2,1)) if padded_feature[0] is not None else None
            padded_feature_ = (tmp, padded_feature[1])
            
        _,_,transfer_feature = transfer_model(feature_, learned_feature, seq_len, None, padded_input=padded_feature_)
    else:
        transfer_feature = None
    
    pairs = None 
    if datatype != "point":
        try:
            idxs, pairs, weight = generate_pairs(ba, ineq, THRESHOLDS, datatype, criterion.__class__.labeltype)
        except:
            pairs = None
            weight = None
        y_pred, loss, ait = test_pair(feature, padded_feature, learned_feature, transfer_feature, seq_len, pairs, weight, model, criterion)
    else:
        y_pred, loss, ait = test_point(feature, padded_feature, learned_feature, transfer_feature, ba, ineq, seq_len, model, criterion)
        
    if file != None and pairs is not None and datatype != "npair":
        num, avgrank, hr, whr, ndcg = metrics.toprank_metrics(ba, y_pred, -1)
        
        s = 'Accuracy Loss {:.4f}\n'.format(loss)
        
        file.write(s)
        print(s)
        for i in num:
            s = 'num#{}: {}, avg_rankl#{}: {:.4f}, hr#{}: {:.4f}, whr#{}: {:.4f}, ndcg#{}: {:.4f}\n'.format(
                i, int(num[i]), i, avgrank[i], i, hr[i], i, whr[i], i, ndcg[i])
            print(s)
            file.write(s)
            file.flush()
        
    return y_pred, ait
    
def test_pair(feature, padded_feature, learned_feature, transfer_feature, seq_len, pairs, weight, model, criterion):
    model.eval()
    y_pred, ait, _ = model(feature, learned_feature, seq_len, transfer_feature, padded_input=padded_feature)
    
    if pairs is None:
        if torch.cuda.is_available():
            if ait is not None:
                ait = ait.cpu().detach().numpy()
            return y_pred.cpu().detach().numpy(), None, ait
        else:
            if ait is not None:
                ait = ait.detach().numpy()
            return y_pred.detach().numpy(), None, ait
        
    pairs = torch.LongTensor(pairs).to(device)
    weight = torch.FloatTensor(weight).to(device)
    try:
        if isinstance(criterion, loss.NPairLoss):
            rankloss = criterion(y_pred, pairs)
        else:
            si = torch.index_select(y_pred, 0, pairs[:,0])
            sj = torch.index_select(y_pred, 0, pairs[:,1])
            rankloss = criterion(si, sj, weight)
    except:
        rankloss = torch.tensor(0)
    
    if torch.cuda.is_available():
        if ait is not None:
            ait = ait.cpu().detach().numpy()
        return y_pred.cpu().detach().numpy(), rankloss.item(), ait 
        
    else:
        if ait is not None:
            ait = ait.detach().numpy()
        return y_pred.detach().numpy(), rankloss.item(), ait
    
def test_point(feature, padded_feature, learned_feature, transfer_feature, ba, ineq, seq_len, model, criterion):
    model.eval()
    y_pred, ait, _ = model(feature, learned_feature, seq_len, transfer_feature, padded_input=padded_feature)
        
    ba = (1-np.log(ba)/np.log(5e4)).clip(0,1)
    ba = torch.FloatTensor(ba).to(device)
    
    rankloss = criterion(y_pred, ba, ineq)
    
    if torch.cuda.is_available():
        if ait is not None:
            ait = ait.cpu().detach().numpy()
        return y_pred.cpu().detach().numpy(), rankloss.item(), ait 
        
    else:
        if ait is not None:
            ait = ait.detach().numpy()
        return y_pred.detach().numpy(), rankloss.item(), ait

def parse_data(data, feature_type, padding, modeltype):
    if modeltype != "SpannyConvModel":
        test_max_len, test_feature, test_learned_feature, test_ba, test_ineq, test_seq_len = extractFeature(data, feature_type, padding)
        return test_max_len, test_feature, None, test_learned_feature, test_ba, test_ineq, test_seq_len
    else:
        test_max_len, test_feature, test_learned_feature, test_ba, test_ineq, test_seq_len = extractFeature(test_data, feature_type, False)
        _, padded_test_feature, _, _, _, _ = extractFeature(test_data, feature_type, True)
        return test_max_len, test_feature, padded_test_feature, test_learned_feature, test_ba, test_ineq, test_seq_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify parameters of baseline for peptide prediction')
    
    parser.add_argument('-a', '--allele', default='None', type=str, help='allele name')
    parser.add_argument('-b', '--batch_size', default='32', type=str, help="batch size")
    parser.add_argument('-d', '--datapath', default=codepath+"/data/fold/", type=str, help='data path')
    parser.add_argument('-dt', '--datatype', default=None, type=str, help='data type of criterion (npair, pair, point, etc)')
    parser.add_argument('-dr', '--dropout', default='0.2', type=str, help='dropout in fully connected layer')
    parser.add_argument('-e', '--embedding_dim', default='1', type=str, help='dimension of the model\'s output (>=1 for npair loss)')
    parser.add_argument('-f','--filters', default='128', type=str, help='number of filters in convolution layers or locally connected layers')
    parser.add_argument('-fd','--fold', default='5', type=str, help='number of folds')
    parser.add_argument('-ft','--feature_type', default='Blosum', type=str, help='feature type')
    parser.add_argument('-g','--gamma', default='0.1', type=str, help='gamma of learning rate decay')
    parser.add_argument('-k','--kmer', default='3', type=str, help='kmer or length of kernels')
    parser.add_argument('-l', '--layer_size', default='[64]', help="layer size of fully connected layer")
    parser.add_argument('-ln', '--layer_num', default="1", type=str, help='number of convolution layers or locally connected layers')
    parser.add_argument('-lr', '--learning_rate', default="0.1", type=str, help='learning_rate')
    parser.add_argument('-ls', '--loss_type', default='HingeLoss1', type=str, help='type of loss function')
    parser.add_argument('-m', '--model_type', default='MHCflurry', type=str, help='name of Model')
    parser.add_argument('-ma', '--model_part', default='Attention', type=str, help='name of Model')
    parser.add_argument('-mo', '--model_path', default='None', type=str, help='name of Transfered Model')
    parser.add_argument('-n', '--nepoch', default=150, type=int, help='maximum number of epoches')
    parser.add_argument('-o', '--output', default=codepath+"/output/", type=str, help='output path')
    parser.add_argument('-p', '--patience', default='20', type=str, help='patience for early stopping with validation loss')
    parser.add_argument('-r', '--regular', default="0.00001", type=str, help='regular')
    parser.add_argument('-s', '--save_model', default='y', type=str, help='whether save model or not? y(save) n(not save)')
    parser.add_argument('-ss', '--step_size', default='5', type=str, help='step size in learning rate decay')
    parser.add_argument('-t', '--threshold', default='[100,500,1000,5000]', type=str, help='threshold')
    parser.add_argument('-tp', '--test_path', required=True, type=str)
    
    args = parser.parse_args()
       
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FOLD = int(args.fold)
    path = args.output
    if args.allele == "None":
        alleles = np.loadtxt(codepath+"src/allele.txt", dtype=str)
    else:
        alleles = [args.allele]
    
    model = torch.load(args.model_path, map_location=torch.device(device))
    
    try:
        padding = model.padding
    except:
        pdb.set_trace()
        raise ValueError("Unexpected Model Type: "+args.model_type)
    
    test_data = np.loadtxt(args.test_path, dtype=str)
    test_max_len, test_feature, test_padded_feature, test_learned_feature, test_ba, test_ineq, test_seq_len = parse_data(test_data, args.feature_type, padding, args.model_type)
    score, ait = test_with_print(test_feature, test_padded_feature, test_learned_feature, test_ba, test_ineq, test_seq_len, model, int(args.kmer), None, args.datatype, args.feature_type, "None", None)
    
    np.savetxt(args.test_path[:-4]+"_"+args.model_type+"_scores.txt", score)
