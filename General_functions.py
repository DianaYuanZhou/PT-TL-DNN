# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:49:32 2024

This file defines several general functions used in main functions

@author: yuanzhou1
"""

import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

'''
def permute_input_tensor(input_tensor, cov_idx):
    
    x = input_tensor[:, cov_idx]
    #x_flat = torch.flatten(x)
    permute_idx = torch.randperm(x.shape[0]).tolist()
    x_permute = x[permute_idx]
    input_tensor[:, cov_idx] = x_permute
    
    return input_tensor
'''
def permute_input_tensor(input_tensor, cov_idx):
    
    
    #x_flat = torch.flatten(x)
    if torch.IntTensor(cov_idx).shape[0] >1:
        for i in cov_idx:
            torch.manual_seed(i)
            permute_idx = torch.randperm(input_tensor.shape[0]).tolist()
            x_permute = input_tensor[permute_idx, i]
            input_tensor[:, i] = x_permute
    else:
        x = input_tensor[:, cov_idx]
        permute_idx = torch.randperm(x.shape[0]).tolist()
        x_permute = x[permute_idx]
        input_tensor[:, cov_idx] = x_permute
    
    return input_tensor

def permute_input_tensor2(input_tensor, cov_idx, seed):
    
    torch.manual_seed(seed)
    permute_idx = torch.randperm(input_tensor.shape[0]).tolist()    
    
    #x_flat = torch.flatten(x)
    if torch.IntTensor(cov_idx).shape[0] >1:
        for i in cov_idx:
            x_permute = input_tensor[permute_idx, i]
            input_tensor[:, i] = x_permute
    else:
        x = input_tensor[:, cov_idx]
        #permute_idx = torch.randperm(x.shape[0]).tolist()
        x_permute = x[permute_idx]
        input_tensor[:, cov_idx] = x_permute
    
    return input_tensor

def Normalize(import_tensor):
    if torch.all(torch.count_nonzero(import_tensor, dim=0) > 0.):
        mean = torch.mean(import_tensor,0,True)
        std = torch.std(import_tensor,0,True,True)
        if torch.any(std==0.):
            nonzero_idx = torch.flatten(torch.nonzero(std[0,]))
            import_tensor[:,nonzero_idx] = torch.add(import_tensor[:,nonzero_idx], -1*mean[:,nonzero_idx])/std[:,nonzero_idx]
            output_tensor = import_tensor
        else:
            output_tensor = torch.add(import_tensor, -1*mean)/std
    else:
        output_tensor = torch.zeros_like(import_tensor)
        norm_idx = torch.where(torch.count_nonzero(import_tensor, dim=0) > 0.)

        mean = torch.mean(import_tensor[:,norm_idx[0]],0,True)
        std = torch.std(import_tensor[:,norm_idx[0]],0,True,True)
        
        if torch.any(std==0.):
            nonzero_idx = torch.flatten(torch.nonzero(std[0,]))
            output_tensor[:,norm_idx[0]][:,nonzero_idx] = torch.add(import_tensor[:,norm_idx[0]][:,nonzero_idx], -1*mean[:,nonzero_idx])/std[:,nonzero_idx]
        else:
            output_tensor[:,norm_idx[0]] = torch.add(import_tensor[:,norm_idx[0]], -1*mean)/std
    return output_tensor