import numpy as np
import datetime
import time
import argparse
import torch
import math
import metrics
import models
import loss
import os
import train
import test
import warnings
import config
from torchsummary import summary
from config import codepath, device, THRESHOLDS, LEARNED_DIM
from datautils import extractFeature
from nnutils import create_var

def prepare_data_validate(modeltype, path, allele, feature_type, fold, ratio, outpath):
    # padding = ConvModel.SpannyConvModel.padding
    # data process
    try:
        padding = eval("models."+modeltype+".padding")
    except:
        raise ValueError("Unexpected Model Type: "+modeltype)
    
    data = np.loadtxt(path+allele+'/fold0_train.txt_', dtype=str)
    
    padded_feature = None
    if modeltype != "SpannyConvModel":
        max_len, feature, learned_feature, ba, ineq, seq_len = extractFeature(data, feature_type, padding)
    else:
        max_len, feature, learned_feature, ba, ineq, seq_len = extractFeature(data, feature_type, False)
        _, padded_feature, _, _, _, _ = extractFeature(data, feature_type, True)
    
    num = int(len(ba) * ratio)
    idxs = np.arange(len(ba))
    for i in range(fold):
        test_idxs = idxs[num*i:min(len(ba), num*(i+1))]
        test_feature = create_var(feature, idx=test_idxs)
        test_padded_feature = create_var(padded_feature, idx=test_idxs)
        test_learned_feature = create_var(learned_feature, idx=test_idxs)
        test_seq_len = seq_len[test_idxs]
        test_ba = ba[test_idxs]
        test_ineq = ineq[test_idxs]
        
        train_idxs = np.delete(idxs, test_idxs, 0)
        train_feature = create_var(feature, ridx=test_idxs)
        train_padded_feature = create_var(padded_feature, ridx=test_idxs)
        train_learned_feature = create_var(learned_feature, ridx=test_idxs)
        train_ba = np.delete(ba, test_idxs, 0)
        train_ineq = np.delete(ineq, test_idxs, 0)
        train_seq_len = np.delete(seq_len, test_idxs, 0)
        
        if not os.path.exists(outpath+"fold0_" +str(i) + "_valid.txt"):
            np.savetxt(outpath+"fold0_" +str(i) + "_valid.txt", data[test_idxs], fmt='%s')
        
        if not os.path.exists(outpath+"fold0_" +str(i) + "_train.txt"):
            np.savetxt(outpath+"fold0_" +str(i) + "_train.txt", data[train_idxs], fmt='%s')
        
        yield max_len, train_feature, train_padded_feature, train_learned_feature, train_ba, train_ineq, train_seq_len, test_feature, test_padded_feature, test_learned_feature, test_ba, test_ineq, test_seq_len

def prepare_data(modeltype, path, allele, feature_type, fold):
    # padding = ConvModel.SpannyConvModel.padding
    # data process
    try:
        padding = eval("models."+modeltype+".padding")
    except:
        raise ValueError("Unexpected Model Type: "+modeltype)
    
    for i in range(fold):
        train_data = np.loadtxt(path+allele+'/fold'+str(i)+'_train.txt', dtype=str)
        test_data = np.loadtxt(path+allele+'/fold'+str(i)+'_test.txt', dtype=str)
        
        if modeltype != "SpannyConvModel":
            train_max_len, train_feature, train_learned_feature, train_ba, train_ineq, train_seq_len = extractFeature(train_data, feature_type, padding)
            test_max_len, test_feature, test_learned_feature, test_ba, test_ineq, test_seq_len = extractFeature(test_data, feature_type, padding)
            max_len = max(train_max_len, test_max_len)
            yield max_len, train_feature, None, train_learned_feature, train_ba, train_ineq, train_seq_len, test_feature, None, test_learned_feature, test_ba, test_ineq, test_seq_len
        else:
            train_max_len, train_feature, train_learned_feature, train_ba, train_ineq, train_seq_len = extractFeature(train_data, feature_type, False)
            _, padded_train_feature, padded_train_learned_feature, _, _, _ = extractFeature(train_data, feature_type, True)
            test_max_len, test_feature, test_learned_feature, test_ba, test_ineq, test_seq_len = extractFeature(test_data, feature_type, False)
            _, padded_test_feature, padded_test_learned_feature, _, _, _ = extractFeature(test_data, feature_type, True)
            max_len = max(train_max_len, test_max_len)
            padded_train_feature = (padded_train_feature, padded_train_learned_feature)
            padded_test_feature = (padded_test_feature, padded_test_learned_feature)
            yield max_len, train_feature, padded_train_feature, train_learned_feature, train_ba, train_ineq, train_seq_len, test_feature, padded_test_feature, test_learned_feature, test_ba, test_ineq, test_seq_len
        

def build_model(model_type, model_part, learned, seq_len, feature_num, kmer, clayer_num, filters, layer_sizes, embedding_dim, activation, output_activation, transfer, transfer_dim, dropout=0.1):
    
    if model_type == "ConvModel":
        convlayers = [{"kernel_size": kmer, "filters": filters, "activation": "ReLU"} for _ in range(clayer_num)]
        model = models.ConvModel(model_part, seq_len, feature_num, convlayers, layer_sizes, learned, embedding_dim, activation, output_activation, transfer, transfer_dim, dropout, posembed=False)
    elif model_type == "SpannyConvModel":
        convlayers = [{"kernel_size": kmer, "filters": filters, "activation": "ReLU"} for _ in range(clayer_num)]
        global_kernel = {"kernel_size": kmer, "filters": filters, "activation": "ReLU"}
        
        model = models.SpannyConvModel(model_part, seq_len, feature_num, global_kernel, convlayers, layer_sizes, learned, embedding_dim, activation, output_activation, transfer, transfer_dim, dropout, posembed=False)
    elif model_type == "MHCflurry":
        locally_connected_layers = [{"kernel_size": kmer, "filters": filters, "activation": "Tanh"} for _ in range(clayer_num)]
        model = models.MHCflurry(model_part, seq_len, feature_num, locally_connected_layers, layer_sizes, learned, embedding_dim, activation, output_activation, transfer, transfer_dim, dropout)
    elif model_type == "Transformer":
        # d_model : 
        model = models.Transformer(model_part, seq_len, feature_num, feature_num*2, filters, int(feature_num/filters), int(feature_num/filters), layer_sizes, learned, embedding_dim, activation, output_activation, transfer, transfer_dim)
    else:
        raise ValueError("Unsupported model type : "+model_type)
   
    if torch.cuda.is_available():
        model.cuda()
    return model
    

if __name__ == '__main__':
    time1 = datetime.datetime.now()
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
    parser.add_argument('-g','--gamma', default='0.2', type=str, help='gamma will be used to control learning rate decay')
    parser.add_argument('-k','--kmer', default='3', type=str, help='k-mer (size of kernels)')
    parser.add_argument('-l', '--layer_size', default='[64]', help="number of fully connected layer")
    parser.add_argument('-ln', '--layer_num', default="1", type=str, help='number of convolution layers or locally connected layers')
    parser.add_argument('-lr', '--learning_rate', default="0.05", type=str, help='learning_rate')
    parser.add_argument('-ls', '--loss_type', default='HingeLoss1', type=str, help='type of loss function (HingeLoss1, HingeLoss2, HingeLoss3, MeanSquare)')
    parser.add_argument('-m', '--model_type', default='MHCflurry', type=str, help='name of Model (MHCflurry, ConvModel, SpannyConvModel)')
    parser.add_argument('-ma', '--model_part', default='Attention', type=str, help='use attention or not (Attention or Flat')

    parser.add_argument('-n', '--nepoch', default=150, type=int, help='maximum number of epoches')
    parser.add_argument('-o', '--output', default=codepath+"/output/", type=str, help='output path')
    parser.add_argument('-p', '--patience', default='20', type=str, help='patience for early stopping with validation loss')
    parser.add_argument('-r', '--regular', default="0.00001", type=str, help='the parameter for regularization term in loss function')
    parser.add_argument('-s', '--save_model', default='y', type=str, help='whether save model or not? y(save) n(not save)')
    parser.add_argument('-ss', '--step_size', default='5', type=str, help='step size in learning rate decay')
    parser.add_argument('-t', '--threshold', default='[100,500,1000,5000]', type=str, help='threshold')

    parser.add_argument('-mo', '--model_path', default='None', type=str, help='name of Transfered Model')
    parser.add_argument('-td', '--transfer_dim', default='0', type=str, help='dimension of transfer embeddings')
    parser.add_argument('-tt', '--train_test', default='0.0', type=str, help='ratio between training dataset and test dataset. '
                                 +'(>0 means that test dataset is splited from dataset; =0 means that test dataset and training dataset are two individual datasets')
    
    args = parser.parse_args()
    
    FOLD = int(args.fold)
    path = args.output
    if args.allele == "None":
        alleles = np.loadtxt(codepath+"/src/allele.txt", dtype=str)
    else:
        alleles = [args.allele]
    
    transfer = True
    if args.model_path == "None":
        transfer = False
    
    # convert threshold into list
    threshold = args.threshold[1:-1]
    thresholds = threshold.split(',')
    thresholds = [float(t) for t in thresholds if t.strip()]

    if THRESHOLDS != thresholds: config.THRESHOLDS = thresholds

    # convert layers
    layers = args.layer_size[1:-1]
    layers = layers.split(',')
    layers = [int(layer) for layer in layers if layer.strip()]
    
    for allele in alleles:
            # directory
        path = args.output+allele+'/'
        
        if not os.path.exists(path):
            os.mkdir(path)
        
        transallele = "True" if args.model_path != "None" else "False"
        path = path + args.model_type+'_'+args.feature_type+'_'+args.layer_size+'_'+args.loss_type+'_'+args.batch_size+'_'+args.filters+'_'+args.kmer+'_'+args.layer_num
        if not os.path.exists(path):
            os.mkdir(path)
        
        # save parameters
        para = open(path+'/para.txt','w')
        
        for name, val in vars(args).items():
            para.write("%s : %s\n" % (name, val))
         
        if float(args.train_test) > 0:
            generator = prepare_data_validate(args.model_type, args.datapath, allele, args.feature_type, FOLD, float(args.train_test), args.output+allele+'/')
        else:
            generator = prepare_data(args.model_type, args.datapath, allele, args.feature_type, FOLD)
        
        skip = False
        
        for i in range(FOLD):
            max_len, feature, padded_feature, learned_feature, ba, ineq, seq_len, test_feature, test_padded_feature, test_learned_feature, test_ba, test_ineq, test_seq_len = next(generator)
            f = open(path+'/fold'+str(i)+'_loss.txt','w')
            
            # get feature and data
            try:
                criterion = eval("loss."+args.loss_type+"()")
            except:
                raise ValueError("Unsupported loss type: "+args.loss_type)
            
            if args.datatype is None:
                datatype = criterion.__class__.defaulttype
            elif args.datatype not in criterion.__class__.datatype:
                raise ValueError("Unsupported data type: "+args.datatype + " for loss type "+args.loss_type)
            else:
                datatype = args.datatype
                
            output_activation = None
            embedding_dim = int(args.embedding_dim)
            if isinstance(criterion, loss.MeanSquare):
                output_activation = "Sigmoid"
            
            if not isinstance(criterion, loss.NPairLoss) and embedding_dim != 1:
                warnings.warn("For loss type <" + args.loss_type + ">, embedding_dim = " + args.embedding_dim + " is unsupported. The program has changed embedding dim to 1 automatically.")
                embedding_dim = 1
                
            feature_num = 0
            learned = False
            if "Blosum" in args.feature_type:
                feature_num += 20
            if "One-hot" in args.feature_type:
                feature_num += 20
            if "Learned" in args.feature_type:
                feature_num += LEARNED_DIM
                learned = True
                
            model = build_model(args.model_type, args.model_part, learned, max_len, feature_num, int(args.kmer), int(args.layer_num), int(args.filters), layers, embedding_dim, "ReLU", output_activation, transfer, int(args.transfer_dim), dropout=0)
            
            optimizer = torch.optim.SGD(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.regular))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=float(args.gamma), patience=int(args.step_size), threshold=1e-4)
        
            # train the model
            model, tr_output, ba, _ = train.train(model, criterion, optimizer, scheduler, 
                                feature, padded_feature, learned_feature, ba, ineq, seq_len, thresholds, datatype,
                                int(args.batch_size), int(args.nepoch), int(args.patience), f, args.model_path
                          )
            # save the result of training
            np.savetxt(path+'/fold'+str(i)+'_train_scores.txt', tr_output)
            np.savetxt(path+'/fold'+str(i)+'_train_bass.txt', ba)
            if args.save_model == 'y':
                 torch.save(model, path+'/fold'+str(i)+'_model.pt') 
            
            score, ait = test.test_with_print(test_feature, test_padded_feature, test_learned_feature, test_ba, test_ineq, test_seq_len, model, int(args.kmer), criterion, datatype, args.feature_type, args.model_path, f)
            
            np.savetxt(path+'/fold'+str(i)+'_scores.txt', score)
            f.close()
            if ait is not None and len(np.shape(ait))<3:
                np.savetxt(path+'/fold'+str(i)+'_att.txt', ait)
            
        time2 = datetime.datetime.now()
        
        para.write('costtime for allele '+ allele +' is '+ str(time2-time1))
        para.close()
