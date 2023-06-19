# import pandas as pd
import numpy as np
import os, glob, torch, random
# from torch_geometric.data import Data, InMemoryDataset, DataLoader
# from torch_geometric.utils import degree, add_self_loops, convert
# import torch
# from BiGCN.Process.dataset import GraphDataset,BiGraphDataset,UdGraphDataset
# os.chdir('/home/rita/111/111-2MLG/project/BiGCN')
# from BiGCN.Process.process import *
# from BiGCN.Process.rand5fold import *

def pertubation(data, method, rho = 0.1, BU = False) :
    # data.to('cpu')
    n_nodes, n_feats = data.x.shape
    choice = np.arange(n_nodes)
    choice = choice[choice != np.array(data.rootindex)]
    selected = np.random.choice(choice, np.around(n_nodes * rho).astype(int), replace=False)
    if method == 'CN' : # Comments contain noise
        noise = np.zeros((n_nodes, n_feats))
        noise[selected] = np.random.randn(1, n_feats)
        noise = torch.from_numpy(noise)
        data.x += noise
        
    elif method == 'CD' : # Comments are deleted
        data.x[selected] = 0
        
    elif method == 'CE' and selected.shape[0] != 1: # Comments are exchangeable
        exchange = selected[torch.randperm(selected.shape[0]).clone()]
        data.x[selected] = data.x[exchange]
        
    elif method == 'PR' : # Propagation sub-structure is removed
        choice = data.edge_index[0].unique()
        choice = choice[choice != data.rootindex]
        if choice.nelement() != 0 :
            selected = np.random.choice(choice, 1)
            idx = (data.edge_index[0] == selected[0]).unsqueeze(0)
            idx = idx.repeat(2, 1)
            data.edge_index = torch.masked_select(data.edge_index, ~idx).reshape(2, -1)
            
            # BU
            if BU :
                idx = (data.BU_edge_index[1] == selected[0]).unsqueeze(0)
                idx = idx.repeat(2, 1)
                data.BU_edge_index = torch.masked_select(data.BU_edge_index, ~idx).reshape(2, -1)
        
    elif method == 'PU' : # Propagation structure is uncertain 
        n = data.edge_index.shape[1]
        choice = np.random.choice(n, np.around(n * rho).astype(int), replace = False)
        idx = np.arange(n) == -1
        idx[choice] = True
        # del_idx = [[i, j] for i, j in zip(data.edge_index[0][choice], data.edge_index[1][choice])]
        # print(del_idx)
        idx = torch.from_numpy(idx).unsqueeze(0)
        idx = idx.repeat(2, 1)
        data.edge_index = torch.masked_select(data.edge_index, ~idx).reshape(2, -1)
        
        # BU
        if BU:
            n = data.BU_edge_index.shape[1]
            choice = np.random.choice(n, np.around(n * rho).astype(int), replace = False)
            idx = np.arange(n) == -1
            idx[choice] = True
            
            # del_idx = [[i, j] for i, j in zip(data.edge_index[0][choice], data.edge_index[1][choice])]
            # print(del_idx)
            idx = torch.from_numpy(idx).unsqueeze(0)
            idx = idx.repeat(2, 1)
            data.BU_edge_index = torch.masked_select(data.BU_edge_index, ~idx).reshape(2, -1)
        
    elif method == 'PI' : # Propagation structure is incorrect
        choice = torch.from_numpy(choice)
        if choice.nelement() > 1 :
            selected = np.random.choice(choice, 2, replace=False)
            # print(selected)
            # del i's parent
            idx = (data.edge_index[1] == selected[0]).unsqueeze(0)
            idx = idx.repeat(2, 1)
            data.edge_index = torch.masked_select(data.edge_index, ~idx).reshape(2, -1)
            # connect i and j
            temp1 = data.edge_index[0].reshape(1, -1)
            temp1 = torch.cat((temp1, torch.tensor(selected[0]).reshape(1, -1)), 1)
            
            temp2 = data.edge_index[1].reshape(1, -1)
            temp2 = torch.cat((temp2, torch.tensor(selected[1]).reshape(1, -1)), 1)
            
            data.edge_index = torch.cat((temp1, temp2), 0)
            
            # BU
            if BU :
                selected = np.random.choice(choice, 2, replace=False)
                # print(selected)
                # del i's parent
                idx = (data.BU_edge_index[1] == selected[0]).unsqueeze(0)
                idx = idx.repeat(2, 1)
                data.BU_edge_index = torch.masked_select(data.BU_edge_index, ~idx).reshape(2, -1)
                # connect i and j
                temp1 = data.BU_edge_index[0].reshape(1, -1)
                temp1 = torch.cat((temp1, torch.tensor(selected[0]).reshape(1, -1)), 1)
                
                temp2 = data.BU_edge_index[1].reshape(1, -1)
                temp2 = torch.cat((temp2, torch.tensor(selected[1]).reshape(1, -1)), 1)
                
                data.BU_edge_index = torch.cat((temp1, temp2), 0)
    # data.to('cuda:0')
    return data

def change_label(data, p) :
    p = np.random.choice([0, 1], p = [1-p, p])
    temp = data
    # temp.to('cpu')
    if p :
        a = np.array(temp.y.cpu())
        b = np.array([0, 1, 2, 3])
        c = np.random.choice(b[a != b], 1)
        temp.y = torch.LongTensor(c)
    # temp.to('cuda:0')
    return temp