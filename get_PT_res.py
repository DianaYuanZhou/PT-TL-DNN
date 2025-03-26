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
from General_functions import SimpleDataset, permute_input_tensor2, Normalize

seed = 1000
gene_name_ls = ['CHRNA3', 'CHRNA5', 'CHRNA6', 'CHRNB3', 'CHRNB4'] #{'CHRNA3', 'CHRNA5', 'CHRNA6', 'CHRNB3', 'CHRNB4'}
stt_file = './res/031625/imp_score_2hl_tl_sage_gp.csv'
pval_file = './res/031625/pval_imp_2hl_tl_sage_gp.csv'

k_folds = 3
hidden_size1 = 16
hidden_size2 = 4
permute_epoch = 5000

def main():

    for gene_name in gene_name_ls:
        print("Seed %6d" % seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        torch.set_default_dtype(torch.float64)
        
        ### Import UKB data
#         gene_data = pd.read_csv('./ukb/' + gene_name + '.csv')
#         gene_data.rename(columns = {'Unnamed: 0':'iid'}, inplace=True)
#         pheno_data = pd.read_csv('./ukb/Y_1001.csv')
#         ## Transfering on race
#         merged_dat = pd.merge(gene_data, pheno_data, on='iid', how='inner')
#         merged_tensor = torch.tensor(merged_dat.values)

#         X = merged_tensor[:,range(1,(merged_tensor.shape[-1]-4))]
#         y = merged_tensor[:,(merged_tensor.shape[-1]-2)].reshape([merged_tensor.shape[0], 1])
        
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
        
        ### Train initial model%run get_PT_res.py
        criterion_dnn = nn.MSELoss()

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
            
            ### Define DNN for SAGE and load parameters
            DNN_best = DeepNeuralNetwork(input_size = X.shape[-1], hidden_size1 = hidden_size1, hidden_size2 = hidden_size2, output_size = 1)
            checkpoint0 = torch.load('./Parameters/nn_checkpoints_'+gene_name+'_TL_sage.pth')
            DNN_best.load_state_dict(checkpoint0['model_state_dict'])
            
            for X_test, y_test in valid_loader:
                
                inputs_test = X_test.double()
                labels_test = y_test.double().reshape(-1, 1)
                
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
                
                pred_loss_perm_avg = np.mean(pred_loss_perm, axis = 1).reshape(pred_loss.shape)
                print(pred_loss[range(10), 0])
                print(pred_loss_perm_avg[range(10), 0])
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
        #stt = np.mean(importance_score)
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
        print(pval_np)
        
        f = open(pval_file, 'a')
        pval_np.tofile(f, sep = ',', format = '%s')
        f.write('\n')
        f.close()
        
        
if __name__ == '__main__':
    main()                
