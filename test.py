
import torchvision.models as models
from dataloader import ChickenskinBags_newdata
from model import Vgg13MIL
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import functional as F 
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import torch.utils.data as data_utils
import pickle
from einops import rearrange
import os
from tqdm import tqdm
import time
import psutil

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if __name__ == "__main__":
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser(description='Thyroid bags')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        print('\nGPU is ON!')

    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    data_path = './data' 
    test_label_path = './test.csv'
    test_loader = data_utils.DataLoader(ChickenskinBags_newdata(target_number='Thyroid', 
                                                    root=data_path, 
                                                    label_path = test_label_path, 
                                                    train=False),
                                            batch_size=1,
                                            shuffle=False, **loader_kwargs)

    model_paths = [  
        './checkpoints/Thyroid_best.pth'
    ]  
    model_names = ['vgg13']  
    y_true_all = {}
    y_scores_all = {} 
    
    for model_path, model_name in zip(model_paths, model_names):  
        if model_name == 'vgg13':  
            model = Vgg13MIL()  
        else:  
            print('No model could apply!!!')  
            continue 
    
        loaded_model_state_dict = torch.load(model_path, map_location='cuda:0')  
        model.load_state_dict(loaded_model_state_dict, strict=False)  
    
        if args.cuda:  
            model = model.cuda()  
        model.eval()  
    
        y_true = []  
        y_scores = [] 
        y_pred = [] 
        for _, data, label in test_loader:  
            if args.cuda:  
                data, label = data.cuda(), label.cuda()  
            output = model(data)  
            Y_hat = torch.argmax(output, dim=1) 
            y_scores.append(F.softmax(output, dim=1)[:, 1].detach().cpu().numpy())  
            y_true.extend(label.cpu().numpy())  
            y_pred.extend(Y_hat.cpu().numpy())  

        y_scores = np.concatenate(y_scores)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)


    for model_name, scores in y_scores_all.items():  
        fpr, tpr, thresholds = roc_curve(y_true_all[model_name], scores)  
        roc_auc = auc(fpr, tpr)  
        print(roc_auc)
        # plt.plot(fpr, tpr, label='{} (AUC = {:.2f})'.format(model_name, roc_auc))  
    
