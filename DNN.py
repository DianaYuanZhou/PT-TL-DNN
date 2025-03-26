# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:30:48 2024

This file contains the Deep Neural Network (DNN) class

@author: yuanzhou1
"""

import torch
from torch import nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, a=-1., b=1.)
        m.bias.data.fill_(0.)

class NN_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN_model, self).__init__()
        
        self.hl1 = nn.Linear(input_size, hidden_size)
        self.activate = nn.Sigmoid()
        self.hl2 = nn.Linear(hidden_size, output_size)
        
        self.W = [self.hl1.weight, self.hl2.weight]
        self.b = [self.hl1.bias, self.hl2.bias]

    def forward(self, input_tensor):
        
        Z1 = self.hl1(input_tensor)
        A1 = self.activate(Z1)
        Z2 = self.hl2(A1)
        
        cache = (Z1, A1)
        Z2.retain_grad()
        for c in cache:
            c.retain_grad()
        
        return Z2
    
    def SGLpenalty(self, SGL_type, lambda_best, thres):
        
        sgl_value = 0.
        if SGL_type == 'all':
            for W1 in self.W:
                W_p_norm = torch.norm(W1, dim = 0, p = 2)
                h_W_p = torch.where(W_p_norm < thres, torch.pow(W_p_norm,2)/(2*thres)+thres/2, W_p_norm).sum()
                sgl_value += lambda_best*h_W_p
        
        if SGL_type == 'input':
            W1 = self.W[0]
            W_p_norm = torch.norm(W1, dim = 0, p = 2)
            h_W_p = torch.where(W_p_norm < thres, torch.pow(W_p_norm,2)/(2*thres)+thres/2, W_p_norm).sum()
            sgl_value += lambda_best*h_W_p
            
        return sgl_value

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(DeepNeuralNetwork, self).__init__()
       
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.Sigmoid(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Sigmoid()
            )
        self.last_fc = nn.Linear(hidden_size2, output_size)
        self.W = [self.linear_sigmoid_stack[0].weight, self.linear_sigmoid_stack[2].weight, self.last_fc.weight]

    def forward(self, input_tensor):
        mid_tensor = self.linear_sigmoid_stack(input_tensor)
        output_tensor = self.last_fc(mid_tensor)
        return output_tensor
    
    def SGLpenalty(self, SGL_type, lambda_best, thres):
        
        sgl_value = 0.
        if SGL_type == 'all':
            for W1 in self.W:
                W_p_norm = torch.norm(W1, dim = 0, p = 2)
                h_W_p = torch.where(W_p_norm < thres, torch.pow(W_p_norm,2)/(2*thres)+thres/2, W_p_norm).sum()
                sgl_value += lambda_best*h_W_p
        
        if SGL_type == 'input':
            W1 = self.W[0]
            W_p_norm = torch.norm(W1, dim = 0, p = 2)
            h_W_p = torch.where(W_p_norm < thres, torch.pow(W_p_norm,2)/(2*thres)+thres/2, W_p_norm).sum()
            sgl_value += lambda_best*h_W_p
            
        return sgl_value
    
class DeepNeuralNetwork_Alter(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(DeepNeuralNetwork_Alter, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size1)
        self.hl1 = nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)
        
        self.activate = nn.Sigmoid()
        
        self.W = [self.input_layer.weight, self.hl1.weight, self.output_layer.weight]
        self.b = [self.input_layer.bias, self.hl1.bias, self.output_layer.bias]        

    def forward(self, input_tensor):
        
        Z1 = self.input_layer(input_tensor)
        A1 = self.activate(Z1)
        Z2 = self.hl1(A1)
        A2 = self.activate(Z2)
        output_tensor = self.output_layer(A2)

        return output_tensor