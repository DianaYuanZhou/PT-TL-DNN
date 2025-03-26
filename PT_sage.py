# -*- codig: utf-8 -*-
"""
Created on Wed Apr  3 15:07:36 2024

This file is the realization of feature selection part of the paper:
    Liu, L., Meng, Q., Weng, C., Lu, Q., Wang, T., & Wen, Y. (2022). 
    Explainable deep transfer learning model for disease risk prediction using high-dimensional genomic data. 
    PLOS Computational Biology, 18(7), e1010328.
    
@author: yuanzhou1
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.autograd import Variable

from sklearn.model_selection import KFold
#from scipy.stats import norm
from statistics import NormalDist

from DNN import DeepNeuralNetwork, init_weights, NN_model
from General_functions import SimpleDataset, permute_input_tensor, permute_input_tensor2, Normalize

seed = 100
gene_name_ls = ['CHRNA3', 'CHRNA5', 'CHRNA6', 'CHRNB3', 'CHRNB4'] #{'CHRNA3', 'CHRNA5', 'CHRNA6', 'CHRNB3', 'CHRNB4'}
stt_file = './res/031625/imp_score_2hl_sage.csv'
pval_file = './res/031625/pval_imp_2hl_sage.csv'

### Simulating Dataset
k_folds = 3

### Tuning Parameters
learning_rate = 1e-4
num_iterations = 50000
print_level = 5000
permute_epoch = 5000

pen_type = 'Group Lasso' #{'Group Lasso', 'L2'}
SGL_type = 'all'  #{'all', 'input'}
L2_lambda_range = torch.FloatTensor([0., 0.001, 0.01, 0.05, 0.1, 0.5, 1., 5.])
lambda_best_file = './res/031625/L2_lambda_best_sage.csv'

def main():

    for gene_name in gene_name_ls:
        print("Seed %6d" % seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        ### Import sage data
        gene_data = pd.read_csv('./sage/' + gene_name + '.csv')
        gene_data.rename(columns = {'Unnamed: 0':'IID'}, inplace=True)
        pheno_data = pd.read_csv('./sage/Y.csv')
        merged_dat = pd.merge(gene_data, pheno_data, on='IID', how='inner')
        merged_tensor = torch.tensor(merged_dat.values)

        X = merged_tensor[:,range(1,(merged_tensor.shape[-1]-3))]
        y = merged_tensor[:,-1].reshape([merged_tensor.shape[0], 1])
        
        ### Normalization
        X = Normalize(X)
        y = Normalize(y)
        n = X.shape[0]
        
        ### Define splitting technique
        kfold = KFold(n_splits = k_folds, shuffle = True, random_state = seed)
        simple_dataset = SimpleDataset(X, y)

        returnscore = []
        returnscorelist = []
        
        for fold, (train_ids, valid_ids) in enumerate(kfold.split(simple_dataset)):
            print(f'FOLD {fold}')
            print('-----------------------------')

            ### Split datasets
            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(valid_ids)

            train_loader = DataLoader(simple_dataset, batch_size = len(train_ids), sampler = train_subsampler)
            valid_loader = DataLoader(simple_dataset, batch_size = len(valid_ids), sampler = valid_subsampler)
            
            ### Criteria of training the DNN model
            criterion_dnn = nn.MSELoss()
            
            valid_loss = np.zeros((L2_lambda_range.shape[0]))
            lambda_idx = int(0)
            
            for l2_lambda in L2_lambda_range:
            
                ### Define DNN model out of the DeepNeuralNetwork class
                DNN = DeepNeuralNetwork(input_size = X.shape[-1], hidden_size1 = 16, hidden_size2 = 4, output_size = 1)
                DNN.apply(init_weights)  ### Initialize weights
                if pen_type == "Group Lasso":
                    DNN_optimizer = torch.optim.SGD(DNN.parameters(), lr = learning_rate)
                else:
                    DNN_optimizer = torch.optim.SGD(DNN.parameters(), lr = learning_rate, weight_decay = l2_lambda)
                lambda1 = lambda epoch: 1/(1+np.log(epoch+2))
                DNN_scheduler = torch.optim.lr_scheduler.LambdaLR(DNN_optimizer, lr_lambda = lambda1)
                        
                for X_train, y_train in train_loader:
                    inputs = X_train.float()
                    labels = y_train.float().reshape(-1, 1)
                
                    for epoch in range(num_iterations):
                        DNN_optimizer.zero_grad()
                        outputs = DNN.forward(inputs)
                        DNN_loss = criterion_dnn(outputs, labels)
                    
                        if pen_type == "Group Lasso":
                            sgl_pen = DNN.SGLpenalty(SGL_type, l2_lambda, 0.001)
                            DNN_loss = DNN_loss + sgl_pen
                    
                        DNN_loss.backward()
                        DNN_optimizer.step()
                        DNN_scheduler.step()
                    
                        train_loss = DNN_loss.item()
                    
                        if (epoch % print_level) == print_level-1:
                            print('Epoch {}/{} \t NN cost = {:.4f}'.format(epoch+1, num_iterations, train_loss))
                
                #### Save checkpoint            
                torch.save({
                    'model_state_dict': DNN.state_dict(),
                    }, './Parameters/nn_checkpoints_'+gene_name+'_sage_'+str(lambda_idx)+'.pth')
                
                for X_test, y_test in valid_loader:
                    inputs_test = X_test.float()
                    labels_test = y_test.float().reshape(-1, 1)
                    with torch.no_grad():
                        DNN.eval()
                    test_outputs = DNN.forward(inputs_test)
                    test_loss = criterion_dnn(test_outputs, labels_test)
                    valid_loss[lambda_idx] = test_loss.item()
                    lambda_idx = int(lambda_idx + 1)
                    
            l2_lambda_best_idx = np.array(valid_loss).argmin()
            l2_lambda_best = L2_lambda_range[l2_lambda_best_idx]
            print(l2_lambda_best)
        
            DNN_best = DeepNeuralNetwork(input_size = X.shape[-1], hidden_size1 = 16, hidden_size2 = 4, output_size = 1)
            checkpoint = torch.load('./Parameters/nn_checkpoints_'+gene_name+'_sage_'+str(l2_lambda_best_idx)+'.pth')
            DNN_best.load_state_dict(checkpoint['model_state_dict'])
            
            torch.save({
                    'model_state_dict': DNN_best.state_dict(),
                    }, './Parameters/nn_checkpoints_'+gene_name+'_sage.pth')
            
            for X_test, y_test in valid_loader:
                inputs_test = X_test.float()
                labels_test = y_test.float().reshape(-1, 1)
                
                with torch.no_grad():
                    DNN_best.eval()
                    
                predictors = DNN_best.forward(Variable(inputs_test, requires_grad = False))
                pred_loss = np.square((predictors - Variable(labels_test, requires_grad = False)).detach().numpy())
                
                pred_loss_perm = np.zeros((inputs_test.shape[0], 0))
                for epoch in range(permute_epoch):
                    inputs_perm = permute_input_tensor2(inputs_test, cov_idx = list(range(X.shape[-1])), seed = epoch)
                    predictors_perm = DNN_best.forward(Variable(inputs_perm, requires_grad = False))
                    pred_loss_perm_i = np.square((predictors_perm - Variable(labels_test, requires_grad = False)).detach().numpy())
                    pred_loss_perm = np.column_stack((pred_loss_perm, pred_loss_perm_i))
                
                #print(pred_loss_perm.shape)
                pred_loss_perm_avg = np.mean(pred_loss_perm, axis = 1).reshape(pred_loss.shape)
                score_k = pred_loss - pred_loss_perm_avg
                scorelist_k = pred_loss - pred_loss_perm
                returnscore.append(score_k)
                returnscorelist.append(scorelist_k)
                
        importance_score_all = np.copy(returnscore[0])
        importance_score_list_all = np.copy(returnscorelist[0])
        
        for i in range(1,k_folds):
            importance_score_all = np.concatenate((importance_score_all, returnscore[i]), axis = 0)
            importance_score_list_all = np.concatenate((importance_score_list_all, returnscorelist[i]), axis = 0)
        
        stt = np.mean(importance_score_all)
        stt_std = np.lib.scimath.sqrt(sum(np.var(importance_score_list_all, axis=1)))/n
        stt_np = np.array([seed, gene_name, stt, stt_std])
        
        f = open(stt_file, 'a')
        stt_np.tofile(f, sep = ',', format = '%s')
        f.write('\n')
        f.close()


        ### Calculating p values
        standard_stt = stt/stt_std
        rv = NormalDist(mu = 0., sigma = 1.)
        pval = rv.cdf(standard_stt)
        pval_np = np.append(np.array([seed, gene_name]), pval)
        
        f = open(pval_file, 'a')
        pval_np.tofile(f, sep = ',', format = '%s')
        f.write('\n')
        f.close()     
        
if __name__ == '__main__':
    main()                
