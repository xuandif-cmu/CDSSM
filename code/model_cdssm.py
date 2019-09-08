# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CDSSM(nn.Module):
    def __init__(self, config):
        super(CDSSM, self).__init__()
        
        self.s_cnum = config['s_cnum']
        self.u_cnum = config['u_cnum']
        self.all_cnum = self.s_cnum + self.u_cnum 
        self.emb_len = config['emb_len']
        self.st_len = config['st_len']
        self.K = 1000 # dimension of Convolutional Layer: lc
        self.L = 300 # dimension of semantic layer: y 
        self.batch_size = config['batch_size']
        self.kernal = 3
        self.conv = nn.Conv1d(self.emb_len, self.K, self.kernal)
        self.linear = nn.Linear(self.K, self.L,bias = False) 
        self.max = nn.MaxPool1d(self.st_len - 2)
        self.cossim = nn.CosineSimilarity(eps=1e-6)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(config['dropout'])
        
        self.in_conv = nn.Conv1d(self.emb_len, self.K, self.kernal)
        self.in_linear = nn.Linear(self.K, self.L,bias = False) 
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.testmode = config['test_mode']
        
    def forward(self, utter, intents, embedding):
        #print("forward")
        
        if (embedding.nelement() != 0): 
            self.word_embedding = nn.Embedding.from_pretrained(embedding)
            
        utter = self.word_embedding(utter)      
        intents = self.word_embedding(intents)
 
        utter = utter.transpose(1,2)
        utter_conv = F.tanh(self.conv(utter))
        utter_conv_max = self.max(utter_conv)
        utter_conv_max_linear = F.tanh(self.linear(utter_conv_max.permute(0,2,1)))
        utter_conv_max_linear = utter_conv_max_linear.transpose(1,2)
                
        intents = intents.transpose(1,2).unsqueeze(2)
        intents = intents.repeat(1,1,self.kernal,1)
        class_num = list(intents.shape)
        
        int_convs = [F.tanh(self.in_conv(intents[:,:,:,i])) for i in range(class_num[3])]    
        int_conv_linear = [F.tanh(self.in_linear(int_conv.permute(0,2,1))) for int_conv in int_convs]
       
        # ==== compute cossim
        sim = [self.cossim(utter_conv_max_linear, yi.transpose(1,2)) for yi in int_conv_linear]
        sim = torch.stack(sim)
        sim = sim.transpose(0,1).squeeze(2)
        #print(sim[0])
        y_pred = [self.softmax(r) for r in sim]
        y_pred = torch.stack(y_pred)
        
        return y_pred
  
    def loss(self, y_pred, y_true): #y_pred result y: target intent
        loss = self.criterion(y_pred, y_true)
        return loss
    
